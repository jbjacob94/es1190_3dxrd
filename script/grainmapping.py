from __future__ import print_function, division
import os, sys, timeit, tqdm, h5py, pprint, collections, concurrent.futures
import numpy as np, pylab as pl, math as m
import numba
import skimage.transform
import scipy.ndimage as ndi
import fast_histogram

import ImageD11.blobcorrector
import ImageD11.columnfile
import ImageD11.cImageD11
import ImageD11.grain
import ImageD11.indexing
import ImageD11.refinegrains
import ImageD11.sinograms.properties
import ImageD11.sparseframe
import ImageD11.sym_u
import ImageD11.transform
import ImageD11.unitcell
import xfab.tools

from id11_utils import peakfiles, friedel_pairs
from s3dxrd_es1190.utils import grain_fitter


# Group all functions necessary for indexing peaks to grains, fitting grains centre of mass, doign grainshape reconstructions
###########################################################################################################################

# ubis Indexing
############################################################################
def index(colf,
          npairs = 10,
          npk_tol = [ (200,0.02) , (100, 0.03) ],
          ds_tol = 0.01,
          max_grains = 1000,
          loglevel=3 ):
    """ set up indexer object for indexing colfile"""
    ImageD11.indexing.loglevel = loglevel
    ind = ImageD11.indexing.indexer_from_colfile(colf, ds_tol=ds_tol, max_grains=max_grains)
    for minpks, tol in npk_tol:
        ind.minpks = minpks
        ind.tol = tol
        ind.score_all_pairs(npairs)
        print(minpks,tol,len(ind.ubis))
    return ind



def indexer_from_colfile( cf, mask=None, usetthc=False, **kwds ):
    """ from ImageD11.indexing. modified to apply mask to colfile when loading gvecs"""
    
    uc = ImageD11.unitcell.unitcell_from_parameters( cf.parameters )
    w = float( cf.parameters.get("wavelength") )
    
    if usetthc:
        cf.gx, cf.gy, cf.gz = ImageD11.transform.compute_g_vectors(cf.tthc, cf.eta, cf.omega,
                                                          wvln  = w,
                                                          wedge = cf.parameters.get('wedge'),
                                                          chi   = cf.parameters.get('chi'))
    
    if mask is None:
        gv = np.array( (cf.gx,cf.gy,cf.gz), float)
    else: 
        gv = np.array( (cf.gx[mask],cf.gy[mask],cf.gz[mask]), float)
        
    kwds.update( {"unitcell": uc, "wavelength":w, "gv":gv.T } )
    return ImageD11.indexing.indexer( **kwds )



def match_ubi(cf, grains_list, group, tolangle=1., toldist=np.inf, verbose=True):
    """ find matching ubis in a grain list using symmetry group from:
    cubic|hexagonal|trigonal|rhombohedralP|tetragonal|orthorhombic|monoclinic_[a|b|c]|triclinic
    angletol : angle tolerance in degree
    disttol: distance tolerance (for grains center of mass) in mm """
    
    # Initialization: read grain list, symmetry etc.
    gl = grains_list
    print( "read grain list")

    try:
        h = getattr( ImageD11.sym_u, group )()
    except:
        print( "# No group!")
        print( "#Using cubic")
        h = ImageD11.sym_u.cubic()
        print ("Made a cubic group")
    if verbose:
        print ( "# Symmetry considered")
        for o in h.group:
            print(o)
            
    tolangle = float(tolangle)
    toldist = float(toldist)
    
    for g in gl:
        g_u = xfab.tools.ubi_to_u_b(g.ubi)[0]
        assert (abs(np.dot(g_u, g_u.T) - np.eye(3)).ravel()).sum() < 1e-6
    
    dtsum = np.zeros(3)
    ndt = 0
    toldist2 = toldist**2
    
    # find duplicates
    duplicates = []
    for i,g1 in enumerate(tqdm.tqdm(gl)):
        cmx1, cmy1 = g1.cmx, g1.cmy
        
        minangle = 180.0
        best = None
        symubis = [np.dot(o, g1.ubi) for o in h.group]
        asymusT   = np.array([xfab.tools.ubi_to_u_b(ubi)[0].T for ubi in symubis])
        trace = np.trace
        pi = np.pi
        
        for j,g2 in enumerate(gl[i:]):    
            cmx2, cmy2 = g2.cmx, g2.cmy
            dt2 = (cmx2-cmx1)**2 + (cmy2-cmy1)**2
            
            sg = None
            aumis = np.dot(asymusT, g2.u)
            arg = (aumis[:,0,0]+aumis[:,1,1]+aumis[:,2,2] - 1. )/2.
            asymangle = np.arccos(np.clip(arg, -1, 1))
            sg = np.argmin(asymangle)
            angle = asymangle[sg]*180.0/pi        

            if np.all([angle <= tolangle, dt2<=toldist2, i!=j]) :
                duplicates.append(j)
                if verbose:
                    print( i,j,"angle  %.4f"%(angle),"distance %.3f"%(np.sqrt(dt2)),"\t", end="\n")
    
    duplicates = np.unique(duplicates)
    print('found ', len(duplicates), 'matching grains with angle diff < ', '%.2f' %tolangle,' degrees and dist <','%.2f' %toldist)
    return [int(d) for d in duplicates]



def plot_hist_drlv2( ind, cf, title, maxtol=0.1):
    ind.histogram_drlv_fit(bins=np.linspace(-0.02*maxtol,maxtol,50))
    
    pl.figure(figsize=(5,5))
    for grh in ind.histogram:
        pl.plot( ind.bins[1:-2], grh[:-2], "-", lw=.5)
    pl.ylabel("number of peaks")
    pl.xlabel("error in hkl")
    pl.title(title)
    

    
def plotidx( ind, cf, title ):
    """ plot synthetic stats for indexed ubis (drlv hist etc.)"""
    ind.histogram_drlv_fit(bins=np.linspace(-1e-4,0.25,50))
    ind.fight_over_peaks()
    f,ax = pl.subplots(3,2, figsize=(10,15))
    a = ax.ravel()
    for grh in ind.histogram:
        a[0].plot( ind.bins[1:-2], grh[:-2], "-", lw=.5)
    a[0].set(ylabel = "number of peaks", xlabel = "error in hkl (e.g. hkl versus integer)",
             title = title)
    m = ind.ga == -1
    a[1].plot(cf.omega[~m], cf.dty[~m], ',')
    a[1].set(title='%d grains'%(ind.ga.max()+1), xlabel='Omega/deg', ylabel='dty/um')
    cut = cf.sum_intensity[m].max() * 1e-4
    weak = cf.sum_intensity[m] < cut
    a[2].plot(cf.omega[m][weak], cf.dty[m][weak], ',')
    a[2].plot(cf.omega[m][~weak], cf.dty[m][~weak], ',')
    a[2].set(title='todo'%(ind.ga.max()+1), xlabel='Omega/deg', ylabel='dty/um')
    a[3].plot(cf.ds[~m], cf.sum_intensity[~m], ',')
    a[3].set(xlabel='ds', ylabel='Intensity', yscale='log')
    a[4].plot(cf.ds[m][weak], cf.sum_intensity[m][weak], ',')
    a[4].plot(cf.ds[m][~weak], cf.sum_intensity[m][~weak], ',')
    a[4].set(xlabel='ds', ylabel='Intensity', yscale='log')
    npks = [(ind.ga == i).sum() for i in range(len(ind.ubis))]
    a[5].hist(npks, bins=64)
    a[5].set(xlabel='Number of peaks', ylabel='Number of grains')
    
    
    
def remove_weak_grains(ubis, fcut):
    """ remove grains with low fraction of remaining peaks / peaks initially assigned (g.npks/g.alln < fcut) : dubious grains. """
    frac_assigned = [ g.npks/g.alln for g in ubis]
    to_keep = [ i for i,f in enumerate(frac_assigned) if f >= fcut]

    ubis = [ubis[i] for i in to_keep]
    return ubis



def select_tth_rings(cf, tth_calc, tth_tol, tth_max = 20, doplot=True, usetthc = True, **kwargs):
    """ select peaks within tth_tol distance from a list of hkl rings in tth histogram. useful to select specific phases.
    If corrected tth (tthc) column is present, will try to use these instead of tth. can be avoided by setting usetthc to False"""
    
    if 'tthc' in cf.titles and usetthc:
        tth = cf.tthc
    else:
        tth = cf.tth
        
    mhkl = np.asarray([abs(tth - (tth_calc[n])) < tth_tol for n in range(len(tth_calc)) if tth_calc[n] < tth_max ])
    mhkl = np.any(mhkl, axis=0)
    
    if doplot:
        pl.figure()
        pl.plot(tth, cf.eta,',', **kwargs)
        pl.plot(tth[mhkl], cf.eta[mhkl],',', **kwargs)
        pl.vlines(x = tth_calc, ymin = cf.eta.min(), ymax = cf.eta.max(), colors='r', lw = .5, alpha=.5)
        pl.xlabel('tth')
        pl.ylabel('eta')
        
    return mhkl
    



def strongest_peaks( colf, uself = True, frac = 0.995, B=0.2, doplot=None, mask=None):
    """ select strongest peaks that make at least frac of total intensity"""
    
    if mask is None:
        mask = np.full(colf.nrows, True)
    
    cor_intensity = colf.sum_intensity[mask] * (np.exp( colf.ds[mask]*colf.ds[mask]*B ) )
    if uself:
        lf = ImageD11.refinegrains.lf(colf.tth[mask], colf.eta[mask])
        cor_intensity *= lf
    order = np.argsort( cor_intensity )[::-1] # sort the peaks by intensity
    sortedpks = cor_intensity[order]
    cums =  np.cumsum(sortedpks)
    cums /= cums[-1]
    enough = np.searchsorted(cums, frac )
    # Aim is to select the strongest peaks for indexing etc ...
    cutoff = sortedpks[enough]
    strongpks = cor_intensity > cutoff
    if doplot is not None:
        f,a = pl.subplots(1,2,figsize=(10,5))
        a[0].plot( cums/cums[-1], ',' )
        a[0].set(xlabel='npks',ylabel='fractional intensity')
        a[0].plot( [strongpks.sum(),], [frac,], "o" )
        a[1].plot( cums/cums[-1], ',' )
        a[1].set(xlabel='npks logscale',ylabel='fractional intensity', xscale='log', 
                 xlim=(np.searchsorted(cums, doplot), len(cums)))
        a[1].plot( [strongpks.sum(),], [frac,], "o" )
    return  strongpks




# pks to grain mapping
############################################################################

def assign_peaks_to_ubis(cf, ubis, hkl_tol = 0.03, nsigma=np.inf, latticesymmetry = ImageD11.refinegrains.triclinic, mask=None, doplot=True):
    """ assign peaks in cf to grains in ubi list. sees how much intensity per grain and also whether peaks are uniq to the grain or shared with others.
    returns grainlist with pks_to_grain mask for each grain and cf with grain_id label
    
    options:
    hkl_tol : max hkl tolerance for assigning a peak to a ubi
    nsigma : threshold to exclude outliers: pks with dist from center of gravity > nsigma * stdev are removed. default to np.inf (no peaks removed)
    latticesymmetry : ImageD11.refinegrain lattice symmetry, needed for ubi fitting
    mask = use a mask for selecting a subset of peaks from cf
    doplot : plot npks vs npks unique etc. to assess quality of indexed grains
    """
    
    # prepare data : g-vectors, ubis, set grain_id column in cf
    ##################################################################
    # select masked g-vectors from cf. Work in progress: try to restrain gvectors to score for a grain, using position information from friedel,pairs
    if mask is None:
        mask = np.full(cf.gx.shape, True)
        
    gv = np.transpose((cf.gx[mask], cf.gy[mask], cf.gz[mask])).copy()
    
    n = len(gv)
    drlv2 = np.full(n, 2, dtype = float)
    labels = np.full(cf.nrows, -1, 'i' )
    cf.setcolumn(labels, 'grain_id')
    
    # lorentz scaling factor for intensity (useful for centre of mass fitting)
    cor_I = cf.sum_intensity * (np.exp( 0.2*cf.ds**2 ) )
    lf = ImageD11.refinegrains.lf(cf.tth, cf.eta)
    cor_I *= lf
    
    # if only ubi orientations are provided, convert them to grains objects
    if not isinstance(ubis[0], ImageD11.grain.grain):
        ubis = [ImageD11.grain.grain(ubi.copy()) for ubi in ubis]
    
    
    # peak assignment: assign all peaks to best fitting ubi
    ##################################################################
    print('assigning peaks to ubis...')
    for i, g in enumerate( tqdm.tqdm(ubis) ):
        g.id = i
        g.alln =  ImageD11.cImageD11.score( g.ubi, gv, hkl_tol )
        npk2 = ImageD11.cImageD11.score_and_assign( g.ubi, gv, hkl_tol, drlv2, labels, i )
        label_all = np.full(n, -1, 'i' )
        npk2 = ImageD11.cImageD11.score_and_assign( g.ubi, gv, hkl_tol, np.full(n, 2, dtype=float), 
                                                   label_all, i )
        g.allpks = label_all == i
        g.isum = cf.sum_intensity[ g.allpks ].sum()
        
    # refinement: update peak labeling, find unique peaks, remove outliers, compute center of mass of grains, refine ubis
    ##################################################################   
    print('refining grains...')
    for i, g in enumerate( tqdm.tqdm(ubis) ):
        # select peaks assigned to this grain
        g.pks = labels == i
        g.npks = g.pks.sum()
        
        # find hkl indices for each peak assigned, and find unique peaks
        hklr = np.dot(g.ubi, gv[g.pks].T )
        hkli = np.round( hklr ).astype( int )
        #drlv2 = ((hklr - hkli)**2).sum(axis=0)
        g.hkl = hkli
        g.etasigns =  np.sign(cf.eta[g.pks]).astype(int)
        uniqpks = np.unique(np.vstack( (g.hkl, g.etasigns) ),axis=1)
        g.nuniq = uniqpks.shape[1]
        
        # does robust grain fitting (reassign peaks to grains and remove outliers )
        g.pks_indx = np.argwhere(g.pks)
        selected,cen,cmy,cmx = graincen( i, cf, nsigma=5, doplot = False )
        outliers = g.pks_indx[~selected]
        g.pks[outliers] = False
        g.pks_indx = np.argwhere(g.pks)
        g.npks = g.pks.sum()
        # update grain labels in colfile
        cf.grain_id[g.pks_indx] = i
    
        # centre-of-mass fitting
        if 'xs' in cf.titles:
            g.cmx = np.average(cf.xs[g.pks], weights = cor_I[g.pks])
            g.cmy = np.average(cf.ys[g.pks], weights = cor_I[g.pks])
        else:
            g.cmx = cmx
            g.cmy = cmy
        
        # ubi fitting
        gvi = np.transpose((cf.gx[g.pks], cf.gy[g.pks], cf.gz[g.pks]))
        refineubis(g, gvi, cf, latticesymmetry, tol=hkl_tol)

    npkstot = sum([g.npks for g in ubis])
    print('npks assigned = ', npkstot, '(prop =  ', '%.2f' %(npkstot/cf.nrows),')')

    # some plots
    ##################################################################   
    if doplot:
        fig = pl.figure(figsize=(10,10), constrained_layout=True)
        fig.add_subplot(221)
        pl.plot([ g.npks for g in ubis ], [ g.nuniq for g in ubis ],'.')
        pl.xlabel('number of peaks')
        pl.ylabel('number of unique peaks')
        # if you see the same peak many times - not on straight line -> grain might be dodgy

        fig.add_subplot(222)
        pl.scatter( [ g.alln for g in ubis ], [ g.npks/g.alln for g in ubis], c = [g.isum for g in ubis], norm = pl.matplotlib.colors.LogNorm() )
        pl.xlabel('Number of peaks total')
        pl.ylabel('Fraction of peaks assigned')
           
        fig.add_subplot(223)
        pl.plot( [ g.alln for g in ubis ], [ g.isum for g in ubis ],'+')
        pl.xlabel('Number of peaks total')
        pl.ylabel('Total intensity')
        pl.semilogy()
        
    return cf, ubis


def refineubis(g, gvecs, cf, latticesymmetry, tol, correct_gvecs_om = True):
    """ refine grain ubi using list of gvecs provided. modified from ImageD11.refinegrains.refine"""
    
    # update g-vectors to account for uncertainty in omega due to coarse step scanning
    if correct_gvecs_om:
        gvecs = updategv(g, cf, tol, ostep=0.8)
        gvecs = gvecs.T
    
    # first time refinement
    npks, avg_drlv2 = ImageD11.cImageD11.score_and_refine(g.ubi, gvecs, tol)
    
    # apply symmetry to mat:
    if latticesymmetry is not ImageD11.refinegrains.triclinic:
        cp = xfab.tools.ubi_to_cell( g.ubi )
        U  = xfab.tools.ubi_to_u( g.ubi )
        g.ubi = xfab.tools.u_to_ubi( U, latticesymmetry( cp ) ).copy()

    # Second time updates the score with the new mat
    npks, avg_drlv2 = npks, avg_drlv2 = ImageD11.cImageD11.score_and_refine(g.ubi, gvecs, tol)
    
    # apply symmetry to mat:
    if latticesymmetry is not ImageD11.refinegrains.triclinic:
        cp = xfab.tools.ubi_to_cell( g.ubi )
        U  = xfab.tools.ubi_to_u( g.ubi )
        g.ubi = xfab.tools.u_to_ubi( U, latticesymmetry( cp ) ).copy()
        
        
        
def updategv(g, cf, tol, ostep):
    """ update g vectors assigned to grain g to account for uncertainty in omega if acquisition was done with coarse steps in omega (shoud not be
    needed for data acquired at the nanofocus station, but may be useful otherwise. Modified from ImageD11.refinegrains.computegv
    g: grain with peak assignment (g.pks)
    pars: ImageD11 parameters"""
    try:
        sign = pars.get('omegasign')
    except:
        sign = 1.0
    
    # omega angles
    om = cf.omega[g.pks]
    
    # recompute gvecs with corrected tth to be sure
    gv = ImageD11.transform.compute_g_vectors(cf.tthc[g.pks], cf.eta[g.pks], om*sign,
                                     wvln  = cf.parameters.get('wavelength'),
                                     wedge = cf.parameters.get('wedge'),
                                     chi   = cf.parameters.get('chi') )
    
    # compute ideal g-vectors: galc from ubi  + recompute tth, eta, omega from gcalc
    mat = g.ubi.copy()
    gvT = np.ascontiguousarray(gv.T)
    hklf = np.dot( mat, gv )
    hkli = np.round( hklf )

    gcalc = np.dot( np.linalg.inv(mat) , hkli )
    tth,[eta1,eta2],[omega1,omega2] = ImageD11.transform.uncompute_g_vectors( gcalc,
                                                                             cf.parameters.get('wavelength'),
                                                                             cf.parameters.get('wedge'),
                                                                             cf.parameters.get('chi') )
    e1e = np.abs(eta1 - cf.eta[g.pks])
    e2e = np.abs(eta2 - cf.eta[g.pks])
    eta_err = np.array( [ e1e, e2e ] )
    best_fitting = np.argmin( eta_err, axis = 0 )

    # pick the right omega (confuddled by take here)
    omega_calc = best_fitting * omega2 + ( 1 - best_fitting ) * omega1
    # Take a weighted average within the omega error of the observed
    omerr = (om*sign - omega_calc)
    # Clip to 360 degree range
    omerr = omerr - ( 360 * np.round( omerr / 360.0 ) )
    # print omerr[0:5]
    omega_calc = om*sign - np.clip( omerr, -ostep/2 , ostep/2 )
    # Now recompute with improved omegas... (tth, eta do not change much)

    gv = ImageD11.transform.compute_g_vectors(cf.tthc[g.pks], cf.eta[g.pks], omega_calc,
                                                                             cf.parameters.get('wavelength'),
                                                                             cf.parameters.get('wedge'),
                                                                             cf.parameters.get('chi') )
    return gv




def calcy( cos_omega, sin_omega, sol):
    """ for fitting the grain centroid and rotation axis position """
    return sol[0] + cos_omega*sol[1] + sin_omega*sol[2]



def counter(labels):
    """ count occurrences of each label in a label list, and return it as a dict object
    e.g: labs = [1,1,2,3,4,2,1,5,1,] -> counter(labs) = defaultdict(int, {1: 4, 2: 2, 3: 1, 4: 1, 5: 1}) """
    d = collections.defaultdict(int)
    for l in labels:
        d[l]+=1
    return d



def fity( y, cos_omega, sin_omega, wt = 1 ):
    """
    Fit a sinogram to get a grain centroid
    # calc = d0 + x*co + y*so
    # dc/dpar : d0 = 1
    #         :  x = co
    #         :  y = so
    # gradients
    """
    g = [ wt*np.ones( y.shape, float ),  wt*cos_omega, wt*sin_omega ]
    nv = len(g)
    m = np.zeros((nv,nv),float)
    r = np.zeros( nv, float )
    for i in range(nv):
        r[i] = np.dot( g[i], wt * y )
        for j in range(i,nv):
            m[i,j] = np.dot( g[i], g[j] )
            m[j,i] = m[i,j]
    sol = np.dot(np.linalg.inv( m ), r)
    return sol



def fity_robust( dty, co, so, nsigma = 5, doplot=False ):
    """ Does robust fitting, so it throws out the outliers ..."""
    cen, dx, dy = fity( dty, co, so )
    calc2 = calc1 = calcy(co, so, (cen, dx, dy))
    selected = np.ones(co.shape, bool)
    for i in range(3):
        err = dty - calc2
        estd = max( err[selected].std(), 1.0 ) # 1 micron
        #print(i,estd)
        es = estd*nsigma
        selected = abs(err) < es
        cen, dx, dy = fity( dty, co, so, selected.astype(float) )
        calc2 = calcy(co, so, (cen, dx, dy))
    # bad peaks are > 5 sigma
    if doplot:
        f, a = pl.subplots(1,2)
        theta = np.arctan2( so, co )
        a[0].plot(theta, calc1, ',')
        a[0].plot(theta, calc2, ',')
        a[0].plot(theta[selected], dty[selected], "o")
        a[0].plot(theta[~selected], dty[~selected], 'x')
        a[1].plot(theta[selected], (calc2 - dty)[selected], 'o')
        a[1].plot(theta[~selected], (calc2 - dty)[~selected], 'x')
        a[1].set(ylim = (-es, es))
        pl.show()
    return selected, cen, dx, dy



def graincen( gid, colf,nsigma=5, doplot=True ):
    """ Find the centre of mass of the grain and the centre of rotation for one grain"""
    m = colf.grain_id == gid
    if sum(m) == 0:
        return m, 0, 0, 0
    romega = np.radians( colf.omega[m] )
    co = np.cos( romega )
    so = np.sin( romega )
    dty = colf.dty[m]
    selected, cen, dx, dy = fity_robust( dty, co, so, nsigma, doplot=doplot)
    return selected, cen, dx, dy



def map_grain_from_peaks( g, flt, ds):
    """
    Computes sinogram
    flt is already the peaks for this grain
    Returns angles, sino
    """   
    NY = len(ds.ybincens)
    iy = np.round( (flt.dty - ds.ybincens[0]) / (ds.ybincens[1]-ds.ybincens[0]) ).astype(int)

    # The problem is to assign each spot to a place in the sinogram
    hklmin = g.hkl.min(axis=1)
    dh = g.hkl - hklmin[:,np.newaxis]
    de = (g.etasigns.astype(int) + 1)//2
    #   4D array of h,k,l,+/-
    pkmsk = np.zeros( list(dh.max(axis=1) + 1 )+[2,], int )
    pkmsk[ dh[0], dh[1], dh[2], de ] = 1
    #   sinogram row to hit
    pkrow = np.cumsum( pkmsk.ravel() ).reshape( pkmsk.shape ) - 1
    pkhkle = np.arange( np.prod( pkmsk.shape ), dtype=int )[ pkmsk.flat == 1 ]
    npks = pkmsk.sum( )
    destRow = pkrow[ dh[0], dh[1], dh[2], de ] 
    sino = np.zeros( ( npks, NY ), 'f' )
    hits = np.zeros( ( npks, NY ), 'f' )
    angs = np.zeros( ( npks, NY ), 'f' )
    adr = destRow * NY + iy 
    # Just accumulate 
    sig = flt.sum_intensity
    ImageD11.cImageD11.put_incr64( sino, adr, sig )
    ImageD11.cImageD11.put_incr64( hits, adr, np.ones(len(de),dtype='f'))
    ImageD11.cImageD11.put_incr64( angs, adr, flt.omega)
    
    sinoangles = angs.sum( axis = 1) / hits.sum( axis = 1 )
    # Normalise sino:
    sino = (sino.T/sino.max( axis=1 )).T
    # Sort (cosmetic):
    order = np.lexsort( (np.arange(npks), sinoangles) )
    sinoangles = sinoangles[order]
    ssino = sino[order].T
    return sinoangles, ssino, hits[order].T



@numba.njit(parallel=True)
def pmax( ary ):
    """ Find the min/max of an array in parallel """
    mx = ary.flat[0]
    mn = ary.flat[0]
    for i in numba.prange(1,ary.size):
        mx = max( ary.flat[i], mx )
        mn = min( ary.flat[i], mn )
    return mn, mx

@numba.njit(parallel=True)
def palloc( shape, dtype ):
    """ Allocate and fill an array with zeros in parallel """
    ary = np.empty( shape, dtype=dtype )
    for i in numba.prange( ary.size ):
        ary.flat[i] = 0
    return ary

# counting sort by grain_id
@numba.njit
def counting_sort( ary, maxval=None, minval=None ):
    """ Radix sort for integer array. Single threaded. O(n)
    Numpy should be doing this...
    """
    if maxval is None:
        assert minval is None
        minval, maxval = pmax( ary ) # find with a first pass
    maxval = int(maxval)
    minval = int(minval)
    histogram = palloc( (maxval - minval + 1,), np.int64 )
    indices = palloc( (maxval - minval + 2,), np.int64 )
    result = palloc( ary.shape, np.int64 )
    for gid in ary:
        histogram[gid - minval] += 1
    indices[0] = 0
    for i in range(len(histogram)):
        indices[ i + 1 ] = indices[i] + histogram[i]
    i = 0
    for gid in ary:
        j = gid - minval
        result[indices[j]] = i
        indices[j] += 1
        i += 1
    return result, histogram


@numba.njit(parallel=True)
def find_grain_id( spot3d_id, grain_id, spot2d_label, grain_label, order  ):
    """
    Assignment grain labels into the peaks 2d array
    spot3d_id = the 3d spot labels that are merged and indexed
    grain_id = the grains assigned to the 3D merged peaks
    spot2d_label = the 3d label for each 2d peak
    grain_label => output, which grain is this peak
    order = the order to traverse spot2d_label sorted
    """
    assert spot3d_id.shape == grain_id.shape
    assert spot2d_label.shape == grain_label.shape
    assert spot2d_label.shape == order.shape
    T = 40
    print("Using",T,"threads")
    for tid in numba.prange( T ):
        pcf = 0 # thread local I hope?
        for i in order[tid::T]:
            grain_label[i] = -1
            pkid = spot2d_label[i]
            while spot3d_id[pcf] < pkid:
                pcf += 1
            if spot3d_id[pcf] == pkid:
                grain_label[i] = grain_id[pcf]
                



# grainshape reconstructions
############################################################################

def do_sinos(args):
    
    """ function compute sinogram in parallel for a list  of grains. grainlist : list of imageD11 grains; i: grain index in list
    cf: columnfile containing peaks data (must have a grain_id column);
    hkl_tol : tolerance for peaks selection"""
    
    g, i, cf, ds, hkltol = args
    # count peaks per grain
    gord, counts = counting_sort( cf.grain_id )
    inds = np.concatenate( ((0,), np.cumsum(counts) ))
    
    # the inds[0] refers to not indexed peaks
    g.pksinds = gord[ inds[i+1] : inds[i+2] ]  
    
    # check that selected peaks in g.pks match with grain_id in cf
    assert cf.grain_id[g.pksinds[0]] == i
    
    # fit between selected peaks and grain ubi
    hkl_real = np.dot( g.ubi, (cf.gx, cf.gy, cf.gz) )
    hkl_int = np.round(hkl_real).astype(int)
    dh = ((hkl_real - hkl_int)**2).sum(axis = 0)
    assert len(dh)  == cf.nrows # sanity check: make sure dh has been computed for every peak in colfile 
    
    # save dh_err + npks before filtering
    g.dherrall = dh.mean()
    g.npksall = cf.nrows
    
    # keep only peaks with dh<hkltol
    flt = cf.copy()
    flt.filter( dh < hkltol*hkltol )
    
    # recompute dh error for filtered peaks only
    hkl_real = np.dot( g.ubi, (flt.gx, flt.gy, flt.gz) )
    hkl_int = np.round(hkl_real).astype(int)
    dh = ((hkl_real - hkl_int)**2).sum(axis = 0)
    
    # dh_err + npks + hkl for filtered peaks
    g.dherr = dh.mean()
    g.npks = flt.nrows
    g.etasigns = np.sign( flt.eta )
    g.hkl = hkl_int
    g.sinoangles, g.ssino, g.hits = map_grain_from_peaks(g, flt, ds )
    return i,g


def do_recons(args):
    """function to run iradon + friedel pairs recon in parallel for sinos in grainmap. return normalized reconstructions"""
    # iradon recon
    #############################
    g, i, cf, ds, pad, mf_size, rcut, do_iradon = args
    
    if do_iradon:
        outsize = g.ssino.shape[0] + pad
        r = skimage.transform.iradon( g.ssino, 
                                      theta = g.sinoangles, 
                                      output_size = outsize,
                                      circle=False )
        #filter + normalize
        r = np.flipud(r)  # to match with fpr recon
        r = ndi.median_filter(r, size=mf_size)  # smooth a bit the iradon recon with a filter
        r_norm = np.where(r>=0, r/r.max(), 0)  # normalize
        msk = np.where(r_norm > rcut, True, False)
    
        #mask
        g.recon_iradon = np.where(msk, r_norm, 0)                # add to grain obect
        g.mask_iradon = clean_mask(msk)
    else:
        g.recon_iradon = None
        g.mask_iradon = None
    
    
    # friedel pair recon
    ##################################
    # check that selected peaks in g.pks match with grain_id in cf
    assert cf.grain_id[g.pks_indx[0]] == i
    
    # compute histogram 2d
    weights = np.log(np.exp(1)+cf.sum_intensity[g.pks])  # weigths in histogram
    fpr = fast_histogram.histogram2d( cf.xs[g.pks], 
                                      cf.ys[g.pks],
                                      weights = weights,   # gives more weight to peaks with high intensity
                                      range = [[ds.ybinedges[0], ds.ybinedges[-1]],
                                               [ds.ybinedges[0], ds.ybinedges[-1]]],
                                      bins = (len(ds.ybinedges), len(ds.ybinedges)) );
    # filter + normalize
    fpr = ndi.median_filter(fpr, size=mf_size)
    fpr_norm = fpr / fpr.max()
    
    # mask
    msk = np.where(fpr_norm > rcut, True, False)
    g.recon_fp = np.where(msk, fpr_norm, 0)
    g.mask_fp = clean_mask(msk)
    
    return i, g


def clean_mask(gm, nb_it=1, structure=np.ones((3,3))):
    """ clean grain mask gm: select largest connected domain (removes all islands disconnected from grain) + fill holes inside grain"""
    
    # find largest connected domain
    lm, nlabs = ndi.label(gm)   # labeled mask, nlabels
    
    if nlabs == 0:
        print('no grain in mask')
        return gm
    
    if nlabs > 1:  # more than one domain in mask     
        # pixels per labeled domain
        npix = np.asarray([len(np.argwhere(lm==l)) for l in range(1,nlabs)])
        labmax = np.argmax(npix)
    
        cm = np.where(lm==labmax+1,True,False)   # take the largest domain
    else: 
        cm = np.where(lm==1,1,0)
        
    # binary closing, to smooth a bit the contour
    cm = ndi.binary_closing(cm, structure=np.ones((3,3)), iterations=5)
    
    # remove holes in the mask
    cm = ndi.binary_fill_holes(cm)

    return cm


def update_grainshapes( grain_recons, grain_masks ):
    '''
    Update a grain masks based on their overlap and intensity.
    At each point the grain with strongest intensity is assigned
    Assumes that the grain recons have been normalized
    '''
    cnt = 0
    for i in range(grain_recons[0].shape[0]):
        for j in range(grain_recons[0].shape[1]):
            if conflict_exists(i,j,grain_masks):
                max_int = 0.0
                leader  = None
                for n,grain_recon in enumerate(grain_recons):
                    if grain_recon[i,j]>max_int:
                        leader = n
                        max_int = grain_recon[i,j]

                #The losers are...
                for grain_mask in grain_masks:
                    grain_mask[i,j]=0

                #And the winner is:
                if leader is not None:
                    grain_masks[leader][i,j]=1
                
                cnt +=1
    print("%d" %cnt, ' pixels corrected')
    return grain_masks


def conflict_exists( i,j,grain_masks):
    '''
    Help function for update_grainshapes()
    Checks if two grain shape masks overlap at index i,j
    '''
    claimers = 0
    for grain_mask in grain_masks:
        if grain_mask[i,j]==1:
            claimers+=1
    if claimers>=2:
        return True
    else:
        return False
    
    

def update_pks_mask(g, cf, ds):
    """update g.pks (pks mask for grain g) to keep only peaks which originate from within 2d grain mask. works only if Friedel pairs data are available""" 
    
    # no pixel in mask
    if np.any(g.mask_fp) == False:   
        g.pks = np.full(cf.nrows, False)
        g.pks_indx = []
        return 
    
    # mpks2d : 2d array of [xs,ys] positions corresponding to grain mask g.mask2D
    mpks2d = ds.ybincens[np.argwhere(g.mask_fp)]

    # msk : any of the peaks in pks for which (xs,ys) coord match with at least one (xs,ys) position in mpks2d
    
    msk = np.any( [
          np.all( [ abs(cf.xs - mpks2d[i][0]) <= ds.ystep/2,
                    abs(cf.ys - mpks2d[i][1]) <= ds.ystep/2],
                 axis=0 ) for i in range(len(mpks2d)) ], axis = 0 )

    # update pks mask
    g.pks = np.all([msk, g.allpks], axis=0)   # keep only peaks which had been initially assigned to grain
    g.npks = sum(g.pks)
    g.pks_indx = np.argwhere(g.pks)

    


