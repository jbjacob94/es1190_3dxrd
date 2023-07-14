import os, sys
import numpy as np, pylab as pl, math as m, h5py
import fast_histogram
import skimage.transform
import scipy.spatial
from scipy.stats import gaussian_kde

from ImageD11 import unitcell, blobcorrector, columnfile, transform, sparseframe, cImageD11, refinegrains
from ImageD11.grain import grain, write_grain_file, read_grain_file
from ImageD11.indexing import indexer
from ImageD11.blobcorrector import eiger_spatial

from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import Phase 



# General functions to do some operations on imageD11 columnfiles

def colf_to_dict(cf):
    "converts ImageD11 columnfile to python dictionnary"""
    keys = cf.titles
    cols = [cf[k] for k in keys]
    cf_dict = dict(zip(keys, cols))
    return cf_dict

def rnd( a, p, t):
    if p == 0:
        return a.astype(t)
    # hack to try to get floats to compress better. Round to nearest integer p
    a = (np.round(a*pow(2,p)).astype(int).astype(t))/pow(2,p)
    return a



def colf_to_hdf( colfile, hdffile, save_mode='minimal', name=None, compression='lzf', compression_opts=None):
    
        """
        Copy a columnfile into hdf file. Modified from ImageD11.columnfile.colfile_to_hdf
        saving options: minimal: saves only necessary information that cannot be computed with updategeometry() (default); 
        full: saves all columns 
        """  
        # LIST OF COLUMNS TO SAVE AS INTEGERS. UPDATED FROM IMAGED11
        INTS = [
        "Number_of_pixels",
        "IMax_f",
        "IMax_s",
        "Min_f",
        "Max_f",
        "Min_s",
        "Max_s",
        "spot3d_id",
        "fp_id",
        "phase_id",
        "h", "k", "l",
        "onfirst", "onlast", "labels",
        "labels",
        "Grain",
        "grainno",
        "grain_id",
        "IKEY",
        "npk2d"
        ]
        
        
        if isinstance(colfile, columnfile.columnfile):
            c = colfile
        else:
            c = columnfile.columnfile( colfile )

        h = h5py.File( hdffile , 'w') # will overwrite if exists
        opened = True
        
        if name is None:
            # Take the file name
            try:
                name = os.path.split(c.filename)[-1]
            except:
                name = 'peaks'
        if name in list(h.keys()):
            g = h[name]
        else:
            g = h.create_group( name )
        g.attrs['ImageD11_type'] = 'peaks'
        
        exclude = ['xl', 'yl', 'zl', 'tth', 'tthc', 'eta', 'gx', 'gy', 'gz', 'ds', 'dsc', 'xs', 'ys', 'r_dist']
        
        if save_mode == 'minimal':
            cols = [col for col in c.titles if col not in exclude]
        else:
            cols = c.titles
        
        for t in cols:
            if t in INTS:
                ty = np.int32
            else:
                ty = np.float32
            # print "adding",t,ty
            dat = getattr(c, t).astype( ty )
            if t in list(g.keys()):
                if g[t].shape != dat.shape:
                    g[t].resize( dat.shape )
                g[t][:] = dat
            else:
                g.create_dataset( t, data = dat,
                                  compression=compression,
                                  compression_opts=compression_opts )
        if opened:
            h.close()
            
            
                
def pkstocolf( pkd , parfile, 
            dxfile="/data/id11/nanoscope/Eiger/spatial_20210415_JW/e2dx.edf",
            dyfile="/data/id11/nanoscope/Eiger/spatial_20210415_JW/e2dy.edf",
          ):
    """ Converts a dictionary of peaks saved in pkstable into and ImageD11 columnfile
    adds on the geometric computations (tth, eta, gvector, etc) """
    spat = eiger_spatial( dxfile = dxfile, dyfile = dyfile )
    cf = columnfile.colfile_from_dict( spat( pkd ) )
    cf.parameters.loadparameters(parfile)
    cf.updateGeometry()
    return cf
            

def colfile_from_dict( pks ):
    """Convert a dictionary of numpy arrays to columnfile """
    titles = list(pks.keys())
    nrows = len(pks[titles[0]])
    for t in titles:
        assert len(pks[t]) == nrows, t
    colf = columnfile.newcolumnfile( titles=titles )
    colf.nrows = nrows
    colf.set_bigarray( [ pks[t] for t in titles ] )
    return colf


def fix_flt( c, cor, parfile ):
    """ 
    spline correction for ImageD11 columnfile
    c : columnfile
    cor : spline correction file
    parfile : imageD11 parameter file
    """
    if any(['s_raw' in c.titles, 'f_raw' in c.titles]):
        c.addcolumn( c.s_raw.copy(), 'sc' )
        c.addcolumn( c.f_raw.copy(), 'fc' )
    
    for i in range( c.nrows ):
        c.sc[i], c.fc[i] = cor.correct( c.s_raw[i], c.f_raw[i] )
    c.parameters.loadparameters(parfile)
    c.updateGeometry()
    return c


def get_uc(c, parfile):
    """ computes unitcell and hkl rings using parameters in the parameter files """ 
    c.parameters.loadparameters(parfile)
    wl = c.parameters.get('wavelength')
    spg = c.parameters.get('cell_sg')

    # compute unit cell
    uc = unitcell.unitcell_from_parameters(c.parameters)
    uc.makerings(c.ds.max())
    
    ds = uc.ringds
    hkls = uc.ringhkls
    ds = np.unique(ds)

    tth_calc = [np.arcsin( wl*d/2 )*360/np.pi for d in ds]
    
    return uc, ds, hkls, tth_calc,wl

def gethkl(cell,spg, sym, wl, dsmax=1.):
    """ return unique ds + hkl rings for a given space group + wavelength """
    u = unitcell.unitcell(cell,sym)
    hkls = u.gethkls_xfab(dsmax, spg) 
    d = [hkls[i][0] for i in range(len(hkls))]
    d = np.unique(d)
    return d, hkls

def save_phase_dict(data_dict, file_path):
    """ save phase structure parameters dictionnary to hdf5
    dict is of the form dict{phase: values}, where values contains the following list of items: [plot_color, unit cell,
    space group (spg), space group nb (spg_no), symmetry (sym), ds_rings, tth_rings] for a specific phase
    """
    with h5py.File(file_path, 'w') as f:
        
        datasets_names = ['plot_color', 'cell', 'spg', 'spg_no', 'sym', 'ds_rings', 'tth_rings']
        
        for key, value in data_dict.items():
            group = f.create_group(key)
            for i, item in enumerate(value):
                if isinstance(item, list):
                    item = np.array(item, dtype=np.float32)  # convert cell list to np array
                
                if isinstance(item, np.ndarray):
                    group.create_dataset(datasets_names[i], data=item, dtype = np.float32)
                elif isinstance(item, str):
                    group.create_dataset(datasets_names[i], data=np.string_(item))
                else:
                    group.create_dataset(datasets_names[i], data=item, dtype = np.int32)
                   
                    
def load_phase_dict(phase_dict_path):
    """ 
    load phase parameters from hdf5 and store them into a dictionnary of the form {phase: values} where each key is a phase name
    and values is a list of params: ['plot_color', 'cell', 'spg', 'spg_no', 'sym', 'ds_rings', 'tth_rings']
    """

    with h5py.File(phase_dict_path,'r') as hin:
        phases = list(hin.keys())
        dataset_names = ['plot_color', 'cell', 'spg', 'spg_no', 'sym', 'ds_rings', 'tth_rings']
        
        phase_dict = {}
        
        for p in phases:
            data = []
            for dsn in dataset_names:
                if isinstance(hin[p][dsn][()], bytes):
                    data.append(hin[p][dsn][()].decode())
                else:
                    data.append(hin[p][dsn][()])
            
            phase_dict[p] = data
        
    return phase_dict


def add_phase_to_dict(phase_dict, save_path, phase_name, cell, spg, spg_no, sym, color, wl, tthmax= 25, overwrite=False):
    """ add new phase to phase dict. By default, it will return assertion error if trying to overwrite a phase already present in phase_dict. to overwrite, set overwrite keyword to True """
    
    # values to write
    if not os.path.exists(save_path) or overwrite:
        d,_ = gethkl(cell,spg, sym, wl, dsmax=2/wl * np.sin(np.radians(tthmax)/2))
        t = 2 * np.arcsin( wl*d/2 )*180/np.pi
        values = [color, cell, spg, spg_no, sym, d, t]
        
    if os.path.exists(save_path):
        if not overwrite:
            assert phase_name not in phase_dict.keys()
    else:
        phase_dict = {}
        
    phase_dict[phase_name] = values
    save_phase_dict(phase_dict, save_path)
    
    return phase_dict


def orix_phase_from_dict(phase_dict, phase_name):
    """ create orix Phase object using crystal structure data in phase_dict"""
    a, b, c, alpha, beta, gamma = [i for i in phase_dict[phase_name][1]]
    
    orix_phase = Phase(space_group = int(phase_dict[phase_name][3]),
                       structure = Structure(title=phase_name,
                                             lattice=Lattice(a,b,c,alpha,beta,gamma)))
    return orix_phase

    


def update_colf_cell(cf, phase_dict, phase, mute=False):
    """ update cf parameters with cell params (a, b, c, alpha, beta, gamma, sg, lattice) from phase in phase_dict """
    uc = phase_dict[phase][1]
    spg =  phase_dict[phase][2]
    sym =  phase_dict[phase][4]

    pars = [uc[0], uc[1], uc[2], uc[3], uc[4], uc[5], spg, sym]
    parnames = 'cell__a', 'cell__b', 'cell__c', 'cell_alpha', 'cell_beta', 'cell_gamma', 'cell_sg', 'cell_lattice_[P,A,B,C,I,F,R]'

    for p, n in zip(pars, parnames):
        cf.parameters.parameters[n] = p
    if not mute:
        print('updated colfile parameters with cell parameters of phase ', phase)


def get_colf_size(cf):
    """ returns size in memory of a columnfile"""
    size_MB = sum([sys.getsizeof(cf[item]) for item in cf.keys()]) / (1024**2)
    print('Total size = ', round(size_MB,2), 'MB')
    return size_MB


def merge_colf(c1, c2):
    """ merge two columnfiles with same columns (same ncols + colnames)"""
    titles = list(c1.keys())
    c_merged = columnfile.newcolumnfile(titles=titles)
    c_merged.setparameters(c1.parameters)
    
    assert c1.ncols==c2.ncols
    for i in range(c1.ncols):
        item1, item2 = list(c1.keys())[i], list(c2.keys())[i]
        assert item1 == item2
    c_merged.set_bigarray( [ np.append(c1[t], c2[t]) for t in titles ] )    
    return c_merged

def dropcolumn(cf, colname):
    """ remove column from colfile """
    assert colname in cf.titles
    
    titles = [t for t in cf.titles if t != colname]
    c_out = columnfile.newcolumnfile(titles=titles)
    c_out.setparameters(cf.parameters)
    c_out.set_bigarray( [ cf[t] for t in titles ] )
    del cf
    return c_out


def iradon_recon(cf, obins, ybins, mask=None, doplot=True, weight_by_intensity=True,circle=True, norm=False, **kwargs):
    """
    compute (and plot) sinogram + reconstruction
    Input : cf = imageD11 columnfile that must contain dty and omega columns
    obins = bins array for omega  # use obinedges!
    ybins = bins array for dty,   # use ybindedges
    mask  = boolean array of len cf.nrows to filter data (default = None), plot : flag to choose whether to plot the data or just return the output 
     weights_by_intensity : weights peak peaks by intensity in the histogram. Intensity defined as log(e+sumI). default is True
     norm : normalize sinogram. Default is false
    """
    if mask is None :
        mask=np.full(cf.nrows,True)
        
    if weight_by_intensity is True:
        weights = np.log(np.exp(1)+cf.sum_intensity[mask])
    else:
        weights = np.ones(cf.sum_intensity[mask].shape)
    
    sino= fast_histogram.histogram2d( cf.dty[mask], 
                                      cf.omega[mask],
                                      weights = weights, 
                                      range = [[ybins[0], ybins[-1]],
                                               [obins[0], obins[-1]]],
                                      bins = (len(ybins)-1, len(obins)-1) );
    
    outsize = sino.shape[0]
    
    if norm is True:
        sino = (sino.T/np.nanmax(sino,axis=1)).T
        sino = np.nan_to_num(sino, nan=0, copy=False)
    
    r = skimage.transform.iradon(sino,theta=obins[:-1],output_size = outsize, circle=circle)
    
    
    if norm is True:
        r = r / np.nanmax(r)

    print(sino.shape, r.shape)
    
    if doplot is True:
        f, a = pl.subplots(1,2, figsize=(10,5), constrained_layout=True)
        a[0].pcolormesh( obins, ybins, sino, **kwargs);
        a[0].set_ylabel('dty')
        a[0].set_xlabel('omega')
        a[0].set_title('sinogram')

        a[1].imshow(r, **kwargs);
        a[1].set_xlabel('x')
        a[1].set_ylabel('y')
        a[1].set_title('iradon reconstruction')

    return sino, r


def friedel_recon(cf, xbins, ybins, doplot=True, mask=None, weight_by_intensity=True, norm=False, **kwargs):
    """
    compute (and plot) sinogram + reconstruction
    Input : cf = imageD11 columnfile that must contain dty and omega columns, obins = bins array for omega, ybins = bins array for dty,
    mask  = boolean array of len cf.nrows to filter data (default = None), plot : flag to choose whether to plot the data or just return the output 
    weights_by_intensity : weights peak peaks by intensity in the histogram. Intensity defined as log(e+sumI). default is True
     norm : normalize 2d histogram. Default is false
    """
    if mask is None :
        mask=np.full(cf.nrows,True)
        
    if weight_by_intensity is True:
        weights = np.log(np.exp(1)+cf.sum_intensity[mask])
    else:
        weights = np.ones(cf.sum_intensity[mask].shape)
        
    r = fast_histogram.histogram2d( cf.xs[mask], 
                                      cf.ys[mask],
                                      weights = weights,   # gives more weight to peaks with high intensity
                                      range = [[xbins[0], xbins[-1]],
                                               [ybins[0], ybins[-1]]],
                                      bins = (len(xbins), len(ybins)) );
    if norm is True:
        r = r / r.max()
    
    if doplot is True:
        f = pl.figure(figsize=(5,5))
        ax = f.add_subplot(111)
        ax.pcolormesh(xbins, ybins, r, **kwargs);
        ax.set_ylabel('y mm')
        ax.set_xlabel('x mm')
        ax.set_title('Friedel pairs reconstruction')

    return r


def select_subset(c, selection_type = 'rectangle',
                  xmin=0, ymin=0, xmax=1, ymax=1,
                  xcenter=0, ycenter=0, r=1):
    """
    select subset of peaks based on position on the map. Columnfile must contain xs, ys columns, obtained from Friedel pairing
    choose either rectangular selection with xmin, ymin, xmax, ymax or circle selection with xcenter, ycenter, radius
    """
    assert selection_type in ['rectangle', 'circle']
    
    if selection_type == 'rectangle':
        assert all([xmin < xmax, ymin < ymax])
        mask = np.all([c.xs <= xmax, c.xs >= xmin, c.ys <= ymax, c.ys >= ymin], axis=0)
        
    else:
        mask = (c.xs - xcenter)**2 + (c.ys - ycenter)**2 <= r**2

    return mask


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


def compute_kde(cf, ds, tthmin=0, tthmax=20, tth_step = 0.001, bw = 0.001, usetthc = True,
                uself = True,  doplot=True, save = True, fname=None):
    """ compute kernel density estimate of cf.tth weighted by intensity to get a 1d XRD profile, that can further be used for phase identification, refinment of cell parameters,etc.
    cf, ds: columnfile + dataset file
    tthmin, tthmax, tthstep: range for kde computation
    bw : kde bandwidth
    usetthc : use Friedel pair corrected tth. default is True
    uself: use Lorentz scaling factor for intensity:  L( theta,eta ) = sin( 2*theta )*|sin( eta )| (Poulsen 2004) """
    
    # select tth col + range
    if usetthc is True:
        msk = np.all([cf.tthc <= tthmax, cf.tthc >= tthmin], axis=0)
        tth = cf.tthc[msk]
    else: 
        msk = np.all([cf.tth <= tthmax, cf.tth >= tthmin], axis=0)
        tth = cf.tth[msk]
    
    #Lorentz factor for intensity correction
    if uself is True:
        cor_I = cf.sum_intensity[msk] * (np.exp( cf.ds[msk]*cf.ds[msk]*0.2 ) )
        lf = refinegrains.lf(tth, cf.eta[msk])
        cor_I *= lf
        
    # compute kde
    print('computing kde. This may take a while...')
    kde = gaussian_kde(tth, bw_method=bw, weights=cor_I)

    # resample tth values with given tth_step
    x = np.arange(tth.min(),tth.max(),tth_step)
    pdf = kde.pdf(x)

    # plot kde
    if doplot:
        fig = pl.figure(figsize=(10,5))
        fig.add_subplot(111)
        pl.hist(tth, bins=x, weights=cor_I,density=True);
        pl.plot(x, pdf, '-', linewidth=1., label='bw = ' +str(bw))
        pl.xlim(tthmin, tthmax)
        pl.xlabel('tth (deg)')
        pl.legend()
        pl.title('Integrated intensity profile - '+ds.dsname)
        pl.show()
        if save:
            fig.savefig(ds.datapath+'_kde.png', format='png')
        
    if save:
        if fname is None:
            fname = ds.datapath+'_bw'+str(bw)+'_kde.txt'
        f = open(fname,'w')
        for l in range(len(x)):
            if format(pdf[l],'.4f') == '0.0000':    # replace zeros by 0.0001 to avoid crashes with profex
                f.write(format(x[l],'.4f')+' '+'0.0001'+'\n')
            else:
                f.write(format(x[l],'.4f')+' '+format(pdf[l],'.4f')+'\n')
        f.close()
    
    return x, pdf


def split_xy_chunks(cf,ds, nx, ny, doplot=True):
    """ using relocated peak sources position (xs,ys) from friedel pairs, split data into x,y chunks
    nx, ny: number of chunks along x and y directions in the sample """
    # bins for chunking
    xbins = np.linspace(ds.ybinedges.min(), ds.ybinedges.max(), nx+1)
    ybins = np.linspace(ds.ybinedges.min(), ds.ybinedges.max(), ny+1)
    
    cf.setcolumn(-1*(np.ones_like(cf.fp_id)), 'chunk_id')
    chunk_labels = np.arange(nx*ny)
    
    # chunk splitting
    chunks = {}
    for chk in tqdm.tqdm(chunk_labels):
        row = chk // nx
        col = chk % nx
        
        # x-y coords of chunk. return them as a dict
        xmin, xmax, ymin, ymax = xbins[col], xbins[col+1], ybins[row], ybins[row+1]
        chunks[chk] = [( xmin, xmax, ymin, ymax )]

        msk = select_subset(cf, 'rectangle', xmin, ymin, xmax, ymax)
        cf.chunk_id[msk] = chk
    
    if doplot:
        fpr = friedel_recon(cf, ds.ybinedges, ds.ybinedges,doplot=False, mask=None, weight_by_intensity=True, norm=True)
        pl.figure()
        pl.pcolormesh(ds.ybinedges, ds.ybinedges,fpr, cmap='Greys_r')
        pl.hlines(y = ybins[1:-1], xmin = min(xbins), xmax = max(xbins), colors='red', alpha=.8)
        pl.vlines(x = xbins[1:-1], ymin = min(ybins), ymax = max(ybins), colors='red', alpha=.8)
        pl.xlabel('x mm')
        pl.ylabel('y mm')
    return cf, chunks





