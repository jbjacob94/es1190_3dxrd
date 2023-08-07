import os, sys, numpy as np, pylab as pl, math as m, h5py
import tqdm, timeit
import fast_histogram
from multiprocessing import Pool
import skimage.transform
import scipy.spatial
from scipy.sparse import csr_matrix

from ImageD11 import unitcell, blobcorrector, columnfile, transform, sparseframe, cImageD11
from ImageD11.grain import grain, write_grain_file, read_grain_file

from id11_utils import peakfiles


# functions to identify Fridel pairs in a dataset and perform operations using these pairs (tth correction, find xy, ys, reflection coordinates in the sample)


def fix_ybins(ds):
    """check that dty central scan is at position dty = 0. If not, update ymin, ymax and dty in dataset and recomputes ybins """
    
    central_bin = len(ds.ybincens)//2
    c0 = ds.ybincens[central_bin]
    
    print('dty_center = ', '%.2f' %c0, ' on bin n°', central_bin)
    
    if abs(c0) > 0.01 * ds.ystep:
        print('dty not centered on zero. Updating dty scan positions in dataset...')
        shift  = (ds.ymax + ds.ymin) / 2
        ds.dty = ds.dty - shift
        print('shift=', shift)
    
        ds.guessbins()
    
    return ds


def sort_ypairs(ds, show=True):
    """
    search for symmetric dty scans ("ypairs") in a columnfile, and add them to dataset
    cf : columnfile. must contain a dty column
    ds : dataset h5 file
    """
    central_bin = len(ds.ybincens)//2
    hi_side = ds.ybincens[central_bin:].tolist()
    lo_side = ds.ybincens[:central_bin+1].tolist()
    
    hi_side.sort()
    lo_side.sort(reverse=True)
    
    ypairs = [(y1,y2) for (y1,y2) in zip(hi_side, lo_side)]
    ds.ypairs = ypairs
    
    if show is True:
        ypairs_round = [(round(y1,4),round(y2,4)) for (y1,y2) in zip(hi_side, lo_side)]
        print('dty pairs: \n', ypairs_round)
    return ds
    

def select_ypair(cf, ds, ypair):
    """select peaks from two symmetric scans in a ypair and return them as two columnfiles"""
    y1, y2 = ypair[0], ypair[1]
    
    c1 =  cf.copy()
    c1.filter( abs(c1.dty-y1) < 0.49 * ds.ystep )  # take a bit less than 1/2 * ystep to avoid selecting same peak multiple times
    
    c2 = cf.copy()
    c2.filter( abs(c2.dty-y2) < 0.49 * ds.ystep )
    
    return c1, c2


def check_y_symmetry(cf, ds, saveplot=False):
    """check that dty scan pairs contain about the same number of peaks + total peak intensity. For each pair, computes total intensity + total npeaks and plot them in function of abs(dty). If the dty and -dty plots do not fit, it is likely that the sample was not correctly aligned on the rotation center, or that dty is not centered on zero
    """
    Nrows, Sum_I = [],[]
    
    for i,j in ds.ypairs:
        I1 = cf.sum_intensity[abs(cf.dty-i) < 0.5*ds.ystep] 
        I2 = cf.sum_intensity[abs(cf.dty-j) < 0.5*ds.ystep] 
        Sum_I.append((np.sum(I1) , np.sum(I2)) )
    
        Nrows.append((len(I1), len(I2)))
        print('.', end='')
    
    central_bin = len(ds.ybincens)//2
    hi_side = ds.ybincens[central_bin:]

    f = pl.figure(figsize=(10,5), constrained_layout=True)
    f.add_subplot(121)
    pl.plot(hi_side, np.cbrt(Sum_I),'.', label=['dty','-dty'])
    pl.xlabel('|dty| mm')
    pl.ylabel('I^1/3')
    pl.legend()

    f.add_subplot(122)
    pl.plot(hi_side,Nrows,'.', label=['dty','-dty'])
    pl.xlabel('|dty| mm')
    pl.ylabel('n peaks')
    pl.legend()
    pl.show()
    
    f.suptitle('dty alignment - dset '+str(ds.dset))
    if saveplot is True:
        f.savefig(str(ds.datapath)+('_dty_alignment.png'), format='png')
        


def compute_csr_dist_mat(c1, c2, dist_cutoff, mult_fact_tth, mult_fact_I, verbose=True):
    """
    computes KDTree for c1 and rotated c2 and returns a sparse distance matrix between the two trees, dropping all values
    above dist_cutoff
    """
    
    # mask to select non-paired data
    msk1 = c1.fp_id == -1
    msk2 = c2.fp_id == -1
    
        
    # rescale tan tth + sum_intensity to have a spread comparable to omega and eta 
    tth_1 = 1./(np.tan(np.radians(c1.tth[msk1])) + 0.04) * mult_fact_tth
    tth_2 = 1./(np.tan(np.radians(c2.tth[msk2])) + 0.04) * mult_fact_tth
    
    sI1 = pow( c1.sum_intensity[msk1], 1/3 ) * mult_fact_I
    sI2 = pow( c2.sum_intensity[msk2], 1/3 ) * mult_fact_I
    
    # compute KD Trees and sparse distance matrix between peaks from g1 and g2
    g1 = np.transpose((c1.eta[msk1]%360, c1.omega[msk1]%360, tth_1, sI1))
    a = scipy.spatial.cKDTree( g1 )
    g2 = np.transpose(((180-c2.eta[msk2])%360, (180+c2.omega[msk2])%360, tth_2, sI2))
    b = scipy.spatial.cKDTree( g2 )
    
    dij = csr_matrix( a.sparse_distance_matrix( b, dist_cutoff ) )
                      
    return dij
                      
def find_best_matches(dij_csr, verbose=True):
    """ clean the csr distance matrix to avoid pairing a peak from c1 with multiple peaks from c2 and conversely.
    dij_csr: input sparse matrix of shape M*N, where M = c1.nrows ad M = c2.nrows
    doplot : if True, does some plotting to see how good the match is between pairs. 
             computes distribution of euclidian distance between pairs +  distance along each coordinate (tth_, sI, eta, omega) 
    outputs:
    dij_best.data -> distance for selected friedel pairs
    c1_indx, c2_indx: indices of Friedel pairs in c1[msk] and c2[msk]: pairs are (c1[msk][i], c2[msk][j]) for (i,j) in (c1_indx, c2_indx) """
    
    n_pairs_all = dij_csr.nnz                  
    
    # Work on inverse of distance and find max values. Have to do this because dij.argmin() returns position of zeros of the
    # matrix, which is quite useless, and there is no method implemented in csr to return minimum non-zero values
    dij_best = dij_csr.copy()
    dij_best.data = np.divide(np.ones_like(dij_csr.data), dij_csr.data, out=np.zeros_like(dij_csr.data), where=dij_csr.data!=0)
    
    # Find max values + row and col index of max values for each col, ie: what are the best matches in c2 for data in c1?
    maxvals  = np.max(dij_best, axis=0).toarray()[0]
    row_indx = np.argmax(dij_best, axis=0).A[0]     #m.A[0] to convert matrix to 1D numpy array
    col_indx = np.arange(len(row_indx))
    # update csr matrix to keep only maxvals
    dij_best = csr_matrix((maxvals, (row_indx, col_indx)), shape = dij_best.shape )
    
    # do the same as above, but working on rows, ie: among the remaining values in c2, what are the best matches in c1?
    maxvals = np.max(dij_best, axis=1).toarray()[:,0]
    col_indx = np.argmax(dij_best, axis=1).A[:,0]    
    row_indx = np.arange(len(col_indx))
    dij_best = csr_matrix((maxvals, (row_indx, col_indx)), shape = dij_best.shape )
    
    dij_best.eliminate_zeros()  # drop all zero values
    
    # inverse again data to get real distance
    dij_best.data = np.divide(np.ones_like(dij_best.data), dij_best.data, out=np.zeros_like(dij_best.data), where=dij_best.data!=0)
    n_pairs_cleaned = dij_best.nnz
    
    if verbose:
        print(n_pairs_cleaned, 'pairs kept out of ', n_pairs_all, ' possible matches')
    
    c1_indx, c2_indx = dij_best.nonzero()  # friedel pairs indices in c1 and c2 
    
    return dij_best.data, c1_indx, c2_indx
  


def label_friedel_pairs(c1, c2, dist_max, dist_step, mult_fact_tth = 1/2, mult_fact_I = 1/25, verbose=True, doplot=False):
    """ Find Friedel pairs in symmetric columnfiles c1 and c2.
    Input:
    c1, c2: set of columnfiles corresonding to symmetric scans [dty, -dty]
    dist_max: maximum distance between two peaks of a Friedel pair
    dist_step: identification of Fridel pairs is done iteratively, starting with a low distance threshold, and increasing it a each iteration until dist = dist_max
    dist_step controls how much the threshold is increased at each iteration
    mult_fact_tth / mult_fact_I: scaling parameter to make sure tth and intensity weight approximately the same as eta and omega in the euclidian distance between peaks
    verbose : print information about pairing process
    doplot : maked plots to assess pairing quality
    output : c_merged : merged columnfile containing paired peaks in c1 and c2, with friedel pair labels (fp_id) and distance between paired peaks (fp_dist)
    fp_labels : list of friedel pair labels identified """
    
    # INITIALIZATION
    ###############################################################################################
    # sort c1 and c2 on spot3d_id at the begining. No sure wether it is useful, but does not harm
    c1.sortby('spot3d_id')
    c2.sortby('spot3d_id')
    
    # create new friedel pair label + fp_dist for c1 and c2,and initialize all values to -1 (non paired)
    c1.addcolumn(np.full(c1.spot3d_id.shape, -1, dtype=int), 'fp_id')
    c2.addcolumn(np.full(c2.spot3d_id.shape, -1, dtype=int), 'fp_id')
    
    c1.addcolumn(np.full(c1.spot3d_id.shape, -1, dtype = np.float64), 'fp_dist')
    c2.addcolumn(np.full(c2.spot3d_id.shape, -1, dtype = np.float64), 'fp_dist')
    
   
    fp_labels = []   # friedelpair labels list, updated at each iteration with newly found set of friedel pairs
    paired_spot3d = []  # list of tuples containing spot3d indices of paired peaks: easier than fp_id for quick selection of paired  peaks in cf
    npkstot = min(c1.nrows, c2.nrows)  # maximum number of pairs to find. Used to compute proportion of paired peaks
    
    
    # FRIEDEL PAIR SEARCH LOOP
    ###############################################################################################
    dist_steps = np.arange(dist_step, dist_max+dist_step, dist_step)
    for it,dist_cutoff in enumerate(dist_steps):
        
        # sort colfiles by fp_label, to put all unpaired peaks at the begining of the columnfile
        c1.sortby('fp_id')
        c2.sortby('fp_id')        
        
        # find Friedel pairs. c1_indx / c2_indx: friedel pair indices in c1[msk1] and c2[msk2] (mask in compute_csr_dist_mat),
        # but since we ordered on fp_id first, indices to select are the same in c1 / c2
        
        dij = compute_csr_dist_mat(c1, c2, dist_cutoff, mult_fact_tth = mult_fact_tth, mult_fact_I = mult_fact_I, verbose=verbose)
        dist, c1_indx, c2_indx = find_best_matches(dij, verbose=verbose)
        
    
        # update friedel pair labels
        if not fp_labels:  # list is empty
            newlabels = np.arange(len(c1_indx))
        else:
            newlabels = np.arange(max(fp_labels)+1, max(fp_labels)+len(c1_indx)+1)
        
        fp_labels.extend(newlabels)
        
        # sanity check. make sure we are not overwriting already indexed pairs
        assert np.all([i == -1 for i in c1.fp_id[c1_indx]])
        assert np.all([j == -1 for j in c2.fp_id[c2_indx]])
        
        # update fp_id and fp_dist in c1, c2
        c1.fp_id[c1_indx] = c2.fp_id[c2_indx] = newlabels
        c1.fp_dist[c1_indx] = c2.fp_dist[c2_indx] = dist
        
        # update list of tuples with spot3d_id of paired peaks
        spot3dtuples = [(i,j) for (i,j) in zip(c1.spot3d_id[c1_indx], c2.spot3d_id[c2_indx])]
        paired_spot3d.extend(spot3dtuples)
            
    if doplot:
    # plot figures to see indexing quality at each step
        m1 = c1.fp_id>=0
        m2 = c2.fp_id>=0
        
        print('dstep_max=', dist_steps[-1])
        
        tth_dist = 1./(np.tan(np.radians(c1.tth[m1])) + 0.04) * mult_fact_tth - 1./(np.tan(np.radians(c2.tth[m2])) + 0.04) * mult_fact_tth
        sumI_dist = pow( c1.sum_intensity[m1], 1/3 ) * mult_fact_I - pow( c2.sum_intensity[m2], 1/3 ) * mult_fact_I
        eta_dist = (c1.eta[m1]%360 - (180-c2.eta[m2])%360)
        omega_dist = (c1.omega[m1]%360 - (180+c2.omega[m2])%360)
        
        def x_lim(x):
            return np.percentile(x,5), np.percentile(x,95)
        def x_bins(x):
            return np.linspace(np.percentile(x,5), np.percentile(x,95),200)
        
        fig = pl.figure(figsize=(8,10))
        
        ax1 = fig.add_subplot(311)
        ax1.hist(c1.fp_dist[m1], bins=x_bins(c1.fp_dist[m1]), density=True);
        ax1.set_xlabel('distance')
        ax1.set_ylabel('prop')
        ax1.set_xlim(x_lim(c1.fp_dist[m1]))
        
        ax2 = fig.add_subplot(323)
        ax2.hist(eta_dist, bins=x_bins(eta_dist), density=True);
        ax2.set_xlabel('eta')
        ax2.set_xlim(x_lim(eta_dist))
        
        ax3 = fig.add_subplot(324)
        ax3.hist( omega_dist , bins=x_bins(omega_dist), density=True);
        ax3.set_xlabel('omega')
        ax3.set_xlim(x_lim(omega_dist))
        
        ax4 = fig.add_subplot(325)
        ax4.hist(tth_dist , bins=x_bins(tth_dist), density=True);
        ax4.set_xlabel('1/tan(tth) * mult_fact_tth')
        ax4.set_xlim(x_lim(tth_dist))
        
        ax5 = fig.add_subplot(326)
        ax5.hist(sumI_dist, bins=x_bins(sumI_dist),density=True);
        ax5.set_xlabel('sum_intensity**1/3 * mult_fact_I')
        ax5.set_xlim(x_lim(sumI_dist))
                     
        fig.suptitle('Mismatch between identified pairs')
        
    
    # MERGE PAIRED DATA AND RETURN OUTPUT
    ###############################################################################################
    # keep only paired peaks
    c1.filter(c1.fp_id != -1)
    c2.filter(c2.fp_id != -1)
    
    #merged the two columnfiles and sort again by fp label
    c_merged = peakfiles.merge_colf(c1, c2)
    c_merged.spot3d_fpairs = paired_spot3d
    c_merged.sortby('fp_id')
    
    if verbose: 
        print('Friedel pair identification Completed.')
        print('N pairs = ', len(fp_labels),' out of ',npkstot, 'possible candidates')
        print('Prop_paired = ', '%.2f' %(len(fp_labels)/npkstot) )

    return c_merged



# group processing of ypair in a single function, which takes a list of arguments provided in args. easier for multithreading with Pool
def process_ypair(args):
    cf, ds, yp, dist_max, dist_step, mult_fact_tth, mult_fact_I = args
    c1, c2 = select_ypair(cf, ds, yp)
    c_merged = label_friedel_pairs(c1, c2, dist_max, dist_step, mult_fact_tth, mult_fact_I, verbose=False, doplot=False)
    c_merged.sortby('fp_id')
    return c_merged



def find_all_pairs(cf, ds, dist_max=1., dist_step=0.05, mult_fact_tth = 1, mult_fact_I = 1/20, doplot=True, verbose=True, saveplot=False):
    """
    process successively all pairs of scans [-dty; +dty ] in a columnfile to find friedel pairs, and concatenate all the output into a new columnfile    with friedel pairs index (fp_id) and distance between paired peaks (fp_dist)
    """
    # check if ds contains ypairs. if not, compute them
    if 'ypairs' not in dir(ds):
        sort_ypairs(ds, show=False)
    
    # use multiprocessing Pool to run friedel pair search in parallel 
    ##########################################################################
    # list of arguments to be passed to process_ypair
    args = []
    for yp in sorted(ds.ypairs):
        args.append((cf, ds, yp, dist_max, dist_step, mult_fact_tth, mult_fact_I))
    
   # if verbose:
    print('Friedel pair search...')
    # for know, don't use multithreading since it does not work
    out = []
    for arg in tqdm.tqdm(args):
        o = process_ypair(arg)
        out.append(o)
    
    ###########################
    # pool object for multithreading. Does not split jobs in different threads, I don't know why. To fix
    #t0 = timeit.default_timer()
    #nthreads = os.cpu_count() -1
    
    #if __name__ == '__main__':
    #    with Pool(nthreads) as pool:
    #        out = list(tqdm.tqdm(pool.map(process_ypair, args), total = len(args)))
    ###########################

    if verbose:
        print('Friedel pair search completed. Concatenate outputs to new columnfile...')
    
    
    # group all outputs into one single colfile, and update fp_labels to make sure each pair is unique in merged column file
    ##########################################################################
    #initialization
    c_cor = out[0]
    c_cor.sortby('fp_id')
    fp_labels = np.unique(c_cor.fp_id)
    spot3d_fpairs = c_cor.spot3d_fpairs  # data stored in c_cor, but not in a column: will be lost when merging columnfile 
                                         # > build this list separately and append it to merged_cf in the end
    
    # update fp_labels
    for colf in out[1:]:    
        newlabels = np.arange(max(fp_labels)+1, max(fp_labels)+len(np.unique(colf.fp_id))+1)
        fp_labels = np.concatenate((fp_labels, newlabels))
        colf.sortby('fp_id')
        colf.setcolumn(recast(newlabels), 'fp_id')
        spot3d_fpairs.extend(colf.spot3d_fpairs)
    
    # merge columnfiles
    for colf in out[1:]:
        c_cor = peakfiles.merge_colf(c_cor, colf)
    
    c_cor.spot3d_fpairs = spot3d_fpairs
    
    if verbose:
        print('Friedel pairing Completed.')
        print('N pairs = ', int(c_cor.nrows/2))
        print('Prop_paired = ', '%.2f' %(c_cor.nrows/cf.nrows) )
        
    if doplot:
        fig = pl.figure(figsize=(7,4))
        
        ax1 = fig.add_subplot(111)
        ax1.hist(c_cor.fp_dist, bins = np.linspace(0, np.percentile(c_cor.fp_dist,99),200), density=True);
        ax1.set_xlabel('distance')
        ax1.set_ylabel('prop')
        ax1.set_xlim(0, np.percentile(c_cor.fp_dist,99))            
        fig.suptitle('Mismatch between identified pairs')
        fig.show()
        
        if saveplot:
            fig.savefig(str(ds.datapath)+'_fp_dist.png', format='png')
                         
    return c_cor, fp_labels



def recast(ary):
    # given an array [x1, x2,...,xn] of len n, returns recast_array [x1, x1, x2, x2, ..., xn, xn] of len 2n
    return np.concatenate((ary,ary)).reshape((2,len(ary))).T.reshape((2*len(ary)))        
        
        
def update_fpairs_geometry(cf, detector = 'eiger', update_gvecs=True):
    """ update geometry of columnfile using friedel pair information. Finds corrected tth, d-spacing (d) and xs,ys coordinate of peak origin in the sample
    input: columnfile cf. Must contain columns with Friedel pair labels 'fp_id' and friedel pair distance (fp_dist)
    detector : eiger / frelon. The two different stations on ID11, use different distance units (µm/mm)
    update_gvecs : Bool flag. If True, recompute g-vectors according to corrected tth"""
    
    assert detector in ['eiger', 'frelon']
    
    # check that columnfile is correct, extract fp_ids and fp_dist and reshape them 
    ################################################################################
    assert np.all(['fp_id' in cf.titles, 'fp_dist' in cf.titles])  # does cf contains columns 'fp_id' and 'fp_dist'?
    assert cf.nrows%2==0  #check that cf contains even number of peaks
    assert np.all([cf.fp_id.min() >= 0, cf.fp_dist.min() != -1])  # check that all peaks in cf have been labeled
                   
    cf.sortby('fp_id') # sort by fp label
    
    # define masks to split data into two twin columnfiles, each containing one item of each pair
    m1 = np.arange(0, cf.nrows, 2)
    m2 = np.arange(1, cf.nrows, 2)
    
    # check that splitting is ok
    assert len(m1) == len(m2)
    assert np.all(np.equal(cf.fp_id[m1],cf.fp_id[m2]))
    assert np.all(np.equal(cf.fp_dist[m1], cf.fp_dist[m2]))
    
    # compute new parameters: tth_cor, d_cor, etc.
    #################################################################################
    
    wl = cf.parameters.get('wavelength')
    L = cf.parameters.get('distance')
    
    # tth correction
    tan1  = np.tan(np.radians(cf.tth[m1]))
    tan2  = np.tan(np.radians(cf.tth[m2]))
    tth_cor = np.degrees(np.arctan( (tan1 + tan2)/2 ) )  # tan_cor = (tan1 + tan2)/2
    d_cor = 2 * np.sin(np.radians(tth_cor)/2) / wl
    
    # get dx, dy: distance of peak from rot center along x and y axis
    dy = (cf.dty[m1] - cf.dty[m2]) / 2  # scan distance from rotation center (y-direction)
    if detector == 'eiger':
        dx = L * (tan1-tan2)/(tan1+tan2)    # distance from rot center along the beam (x direction) in µm (eiger detector , ns station)
    else:
        dx = L/1000 * (tan1-tan2) / (tan1+tan2)    # distance from rot center along the beam (x direction) in mm (Frelon detector, 3DXRD station)
    
    # cos(omega) + sin(omega), needed to rotate back peak coordinates to sample reference frame
    o = np.radians(cf.omega)
    o[m2] = (o[m2]-np.pi)%(2*np.pi)  # rotate peaks from c2 by 180°
    co,so = np.cos(o), np.sin(o)

    # little subtility: omega can be slightly different between two members of a friedel pair -> would lead to different xs,ys, which causes issues
    #later in the processing -> for each pair, take average value of omega and assign it to both peaks 
    co = recast((co[m1]+co[m2])/2)
    so = recast((so[m1]+so[m2])/2)
    
    # rearrange dx, dy arrays + invert sign for data from negative dty scans (needed to make xs, ys match for fp twins)
    dx = recast(dx)
    #dx[m2] = -dx[m2]
    dy = recast(dy)
    #dy[m2] = -dy[m2]
    
    r_dist = np.sqrt(dx**2 + dy**2)

    # calculate x,y coordinates in sample reference frame (xs ,ys)
    xs = co*dx + so*dy
    ys = -so*dx + co*dy
    
    
    # recast arrays and add them as new columns in cf
    cf.addcolumn(recast(tth_cor), 'tthc')
    cf.addcolumn(recast(d_cor), 'dsc')
    cf.addcolumn(xs, 'xs')
    cf.addcolumn(ys, 'ys')
    cf.addcolumn(r_dist, 'r_dist')
    
    # update gvectors
    if update_gvecs:
        cf.gx, cf.gy, cf.gz = transform.compute_g_vectors(cf.tthc, cf.eta, cf.omega,
                                                          wvln  = wl,
                                                          wedge = cf.parameters.get('wedge'),
                                                          chi   = cf.parameters.get('chi'))
    
    return cf


def split_fpairs(cf):
    """ split columnfile with friedel pairs into two twi columnfiles containing each one item of each pair"""
    
    cf.sortby('fp_id') # sort by fp label
    
    # define masks to split data into two twin columnfiles, each containing one item of each pair
    m = np.arange(cf.nrows)%2 == 0
    
    # check that splitting is ok
    assert np.all(np.equal(cf.fp_id[m],cf.fp_id[~m]))
    assert np.all(np.equal(cf.fp_dist[m],cf.fp_dist[~m]))
    
    # filter peaks
    c = cf.copy()
    c.filter(m)
    c_ = cf.copy()
    c_.filter(~m)
    
    return c, c_



def find_missing_twins(cf, selection, restrict_search=False, restrict_subset=[]):
    """ starting from a peakfile cf and a subset of it (selected with peak index), this code:
    1) find friedel pair ids (fpids) of peaks in the subset
    2) Identify "singles", ie peaks missing their friedel twin with same fpid
    3) Find the missing twins in the full columnfile or in a second subset (if restrict_search is true) and add them to the selection
    Search can be restricted to a second subset of cf '"""
    
    fpids = cf.fp_id[selection]  # fpids in selection
    
    # use np.unique to find "single" peaks: peaks missign their friedel twin
    _, ind, cnt = np.unique(fpids, return_counts=True, return_index=True)
    fp_single = fpids[ind[cnt!=2]]
    
    if len(fp_single)==0:
        return selection
    
    # list of peaks ids to select in columnfile
    if restrict_search is True:
        pks_to_add = np.concatenate([np.argwhere(cf.fp_id[restrict_subset] == i) for i in fp_single])[:,0]   # pks indices to select in subset
        newpks = restrict_subset[pks_to_add]  # pks indices to select in full subset
    else:
         newpks = np.concatenate([np.argwhere(cf.fp_id == i) for i in fp_single])[:,0]
    
    new_selection = np.unique( np.concatenate((selection, newpks)) )  # concatenate with former selection
    return new_selection
