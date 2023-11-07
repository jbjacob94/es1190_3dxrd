from concurrent.futures import ProcessPoolExecutor
import os, sys, copy, h5py, tqdm
import numpy as np, pylab as pl, math as m

import fast_histogram
from matplotlib_scalebar.scalebar import ScaleBar

import ImageD11.columnfile, ImageD11.grain, ImageD11.refinegrains, ImageD11.sym_u, ImageD11.cImageD11
import xfab

from diffpy.structure import Atom, Lattice, Structure
from orix import data, io, plot as opl, crystal_map as ocm, quaternion as oq, vector as ovec

from id11_utils import peakfiles, crystal_structure


# Collection of functions to work on 2D pixel maps. Includes Pixelmap Class: A class to store data on a 2D pixel grid. It is used to do pixel-by-pixel indexing, phase assignment and plot grain properties (orientation, indexing quality, etc.) on a 2D map
#################################################################################################

# Note on pixel-by-pixel indexing strategy:
##################################
#The classic indexing strategy with ImageD11 consists in finding g-vectors that are compatible in orientation, considering a given crystal symmetry and lattice parameters. However, this process becomes very inefficient as the number of peaks in the dataset grows very large, because it does not take into account positional information on grains, which is refined in a later stage. Using Friedel pairs, it is possible to relocate the source of diffraction in the sample reference frame (as long as the sample does not consists in a few very large grains, in this case peak position within the grain is poorly defined). Thus, it is possible to assign each peak to a pixel on a 2d xy grid and do 'pixel-by'pixel' indexing. This allows to take into account precise information on peak location. This makes the indexing process much faster, as the number of g-vectors to match for each pixel is only a small subset of the entire dataset and the indexing process can be parallelized to each pixel on the map. This pixel-by-pixel processing can also be done for phase identification (for polyphased samples) prior to indexing. This gives significantly better results than separating phases using only a simple threshold on two-theta angle, conflicts cause dby overlappinf tth rings can be solved pixel-by-pixel by selecting for each one the phase with the largest cumulated intensity.


# General functions
###########################################################################
def xyi(xi, yi):
    """ returns xyi from xi,yi coordinates. xyi = 1000*yi+xi. Allows to assign each pixel a single index, which is easier to process. It will fail if the map is larger than 999x999 pixels (in this case, change definition of xyi to allow larger map size)"""
    return int(xi+1000*yi)

def xyi_inv(xyi):
    """ return xi, yi from xyi coord"""
    xi = xyi % 1000
    yi = xyi // 1000
    return xi, yi


def pks_from_neighbour_pixels(cf, xp, yp, n_px=1):
    """ find peaks in a square of size 2.n_px+1 centered on central pixel xp,yp. 
    n_px=1 -> 3x3 pixel domain selected; n_px=2 -> 5x5 etc. """
    xyi_p = xyi(xp,yp)  # xyi coord
    xmax, ymax = cf.xi.max(), cf.yi.max()  # map boundary
    
    # x,y index of pixel to select
    x_range = np.arange(max(xp-n_px,0), min(xp+n_px+1,xmax))
    y_range = np.arange(max(yp-n_px,0), min(yp+n_px+1,ymax))
    
    pks_sel = []
    for i in x_range:
        for j in y_range:
            start, end = np.searchsorted(cf.xyi, (xyi(i,j), xyi(i,j)+1))
            pks = np.arange(start, end, dtype='int')
            
            pks_sel = np.append(pks_sel,pks).astype(int)
    
    return pks_sel


def pks_from_neighbour_pixels_fast(cf, xp, yp, xymax):
    """ faster variant of pks_from_neighbour_pixels, but can only select 3x3 domain """
    xyi_p = xyi(xp,yp)  # xyi coord    

    s1, e1, s2, e2, s3, e3 = np.searchsorted(cf.xyi, (max(xyi_p-1001,0), max(xyi_p-998,0),
                                                      max(xyi_p-1,0), min(xyi_p+2,xymax),
                                                      min(xyi_p+999,xymax), min(xyi_p+1002,xymax) ) )
    
    pks = np.concatenate((np.arange(s1, e1, dtype='int'),
                          np.arange(s2, e2, dtype='int'),
                          np.arange(s3, e3, dtype='int')))
    return pks



def get_searchsort_indx(ary):
    """ find pixel indices xyi to pass to np.searchsorted. Needed for get_grain_pks (below)"""
    searchsort_indx = [ary[0]]
    for i in range(len(ary)-1):
        if ary[i+1] == ary[i]+1:
            continue
        searchsort_indx.extend([ary[i]+1, ary[i+1]])
    searchsort_indx.append(ary[-1]+1)
    
    assert len(searchsort_indx)%2 == 0 # even nb of indices
    return searchsort_indx

def get_grain_pks(cf, g):
    """find peak indices corresponding to grain g in columnfile cf
    input:
    cf: columnfile sorted by xyi index
    g: grain, must contain a property xyi_indx  xyi indices coresponding to the grain in pixelmap
    output : 
    pks: indices of peaks corresponding to grain g in cf (sorted by xyi)"""
    
    assert 'xyi_indx' in dir(g)
    searchsort_inds = get_searchsort_indx(g.xyi_indx)  # indices to pass to np.searchsorted
    bounds = np.searchsorted(cf.xyi, searchsort_inds)    # bounding values in 
    
    pks = []
    
    for i,j in zip(bounds[::2], bounds[1::2]):
        pksi = np.arange(i,j, dtype='int')
        pks = np.concatenate((pks, pksi)).astype(np.int32)
    return pks


def map_grains_to_cf(glist, cf, overwrite=False):
    """ find peak indices for all grains in glist and map grains to cf / pks to grains 
    (add grain_id column to cf and pksindx prop to all grains in glist)
    
    glist : list of grains (should have xyi_indx property)
    cf: peakfile
    overwrite : if True, reset grain_id column in peakfile. default if False
    """
    
    cf.sortby('xyi')  # needed for searchsorted
    
    if 'grain_id' not in cf.titles or overwrite:
        cf.addcolumn(np.full(cf.nrows, -1, dtype=np.int16), 'grain_id')

    for g in tqdm.tqdm(glist):
        assert np.all(['gid' in dir(g), 'xyi_indx' in dir(g)])   # check grain has grain id  + xyi coordinates mapping
        gid = g.__getattribute__('gid')
        
        pksindx = get_grain_pks(cf,g)  # get peaks from grain g
        
        # map grain to cf and pks to grain
        cf.grain_id[pksindx] = gid
        g.pksindx = pksindx
                
    print('completed')  
    
    
def refine_grains(glist, cf, hkl_tol, sym, return_stats=True):
    """ run score_and_refine on a list of grains, taking initial ubis as a starting guess. peaks are selected using the g.pksindx
    computed using "map_grain_to_cf". Run it before if no "pksindx" property is found in grains
    glist : list of ImageD11 grains
    cf : ImageD11 columnfile sorted by xyi
    hkl_tol : tolerance passed to score_and_refine
    sym : crystal symmetry, used to compute misorientation between old and new orientation. orix.quaternion.symmetry.Symmetry object
    return_stats: returns list of rotation (angle between old and new crystal orientation) + prop of peaks retained. Default is True"""

    ubis_new, prop_indx, misOri, gids, Npks = [], [], [], [], []
    for g in tqdm.tqdm(glist):
        assert 'pksindx' in dir(g)
    
        gv = np.transpose([cf.gx[g.pksindx], cf.gy[g.pksindx], cf.gz[g.pksindx]]).copy() 
        ubi = g.ubi.copy()
    
        # refine grain ubis
        npks_ind, drlv = ImageD11.cImageD11.score_and_refine(ubi, gv, hkl_tol)
    
        # compute rotation angle between former and new ubi + prop of peaks retained
        o = oq.Orientation.from_matrix(g.U, symmetry =sym)
        o2 = oq.Orientation.from_matrix( xfab.tools.ubi_to_u(ubi), symmetry = sym)
        
        ubis_new.append(ubi)
        misOri.append( o2.angle_with(o, degrees=True)[0] )
        prop_indx.append( npks_ind/len(g.pksindx) )
        gids.append(g.gid)
        Npks.append(len(g.pksindx))
        
        g.set_ubi(ubi)
        
    if return_stats:
        return gids, Npks, prop_indx, misOri


    
# to add to pixelmap: 
###################

# index_phase_to_pixel(args=(cf_to_index, xi, yi, minpks))  from 005_label_pixelmap in apr23 folder
# find_pixel_orientations(args=(to_index, xi, yi, etc.))    from 006_index_pixelmap in apr23 folder
#
# methods: 
# - update column  Need to secure operations to avoid overwriting a full column by mistake


# Pixelmap object
###########################################################################

class Pixelmap:
    """ A class to store pixel information on a 2d grid """
    
    # Init
    ##########################
    def __init__(self, xbins, ybins, h5name=None):
        # grid + pixel index
        self.grid = self.GRID(xbins, ybins)
        self.xyi = np.asarray([i + 1000*j for j in ybins for i in xbins]).astype(np.int32)
        self.xi = np.array(self.xyi % 1000, dtype=np.int16)
        self.yi = np.array(self.xyi // 1000, dtype=np.int16)
        
        # phase / grain labeling  + crystal structure information
        self.phases = self.PHASES() 
        self.phase_id = np.full(self.xyi.shape, -1, dtype=np.int8)   # map of phase_ids
        self.grain_id = np.full(self.xyi.shape, -1, dtype=np.int16)   # map of grain_ids
        
        # grains
        self.grains = self.GRAINS_DICT()
        
        self.h5name = h5name
    
    def __str__(self):
        return f"Pixelmap:\n size:{self.grid.shape},\n phases: {self.phases.pnames},\n phase_ids: {self.phases.pids},\n titles:{self.titles()}, \n grains:{len(self.grains.glist)}"
    
    def get(self,attr):
        # alias for __getattribute__
        return self.__getattribute__(attr)
    
    
    # subclasses
    ##########################  
    class GRID:
        # grid properties
        def __init__(self, xbins, ybins):
            self.xbins = xbins
            self.ybins = ybins
            self.shape = (len(xbins),len(ybins))
            self.nx = len(xbins)
            self.ny = len(ybins)
            self.pixel_size = 1
            self.pixel_unit = 'um'
            
            
        def __str__(self):
            return f"grid: size:{self.shape}, pixel size: {str(self.pixel_size)+' '+self.pixel_unit}"
   
        def scalebar(self):
        # scalebar for plotting maps
            scalebar =  ScaleBar(dx = self.pixel_size,
                                     units = self.pixel_unit,
                                     length_fraction=0.2,
                                     location = 'lower left',
                                     box_color = 'w',
                                     box_alpha = 0.5,
                                     color = 'k',
                                     scale_loc='top')
            return scalebar
    
    class PHASES:
        # crystal structures
        def __init__(self):
            self.notIndexed = crystal_structure.CS(name='notIndexed')
            self.pnames = ['notIndexed']
            self.pids = [-1]
            self.cif_paths = ['']
            
        def __str__(self):
            return f"phases: {self.pnames}"
        
        def get(self,attr):
            return self.__getattribute__(attr)
            
        def add_phase(self, pname, cs):
            """ add phase to pixelmap.phases. pname = phase name, cs = crystal_structure.CS object """
           
            # if this phase name already exists, delete it
            if pname in self.pnames:
                print(pname, ': already have a phase with this name. Will overwrite it')
                self.delete_phase(pname)
                
            # write new phase and update pnames and pids lists    
            setattr(self, pname, cs)
            self.pnames.append(pname)
            self.pids.append(cs.phase_id)
            self.cif_paths.append(cs.cif_path)
            self.sort_phase_lists()
            
        def delete_phase(self, pname):
            cs = self.get(pname)
            pid = cs.get('phase_id')
            path = cs.get('cif_path')
            self.pnames = [p for p in self.pnames if p != pname]
            self.pids = [i for i in self.pids if i != pid]
            self.cif_paths = [p for p in self.cif_paths if p != path]
            delattr(self, pname)
            
            self.sort_phase_lists()
             
        def sort_phase_lists(self):
            # sort pnames and pids by phase id
            sorted_pids = [l1 for (l1, l2, l3) in sorted(zip(self.pids, self.pnames, self.cif_paths), key=lambda x: x[0])]
            sorted_pnames = [l2 for (l1, l2, l3) in sorted(zip(self.pids, self.pnames, self.cif_paths), key=lambda x: x[0])]
            sorted_paths = [l3 for (l1, l2, l3) in sorted(zip(self.pids, self.pnames, self.cif_paths), key=lambda x: x[0])]
            self.pids = sorted_pids
            self.pnames = sorted_pnames
            self.cif_paths = sorted_paths   
            
            
    class GRAINS_DICT:
        # dict of ImageD11 grains
        def __init__(self):
            self.dict = {}
            self.gids = list(self.dict.keys())
            self.glist = list(self.dict.values())
            
        def __str__(self):
            return f"nb grains: {len(self.glist)}"
        
        
        def get(self,prop, grain_id):
            """ return a property prop for a grain selected by grain id"""
            g = self.dict[grain_id]
            return g.__getattribute__(prop)
        
            
        def get_all(self, prop):
            """ return a grain property for all grains in grains_dict as an array """
            return np.array( [g.__getattribute__(prop) for g in list(self.dict.values())] )
        
        
        def select_by_phase(self, pname):
            """ return all grains from a given phase. pname : phase name, must be in xmap.phases"""
            gsel = [g for g in self.glist if g.phase == pname]
            return gsel
        
        
        def add_prop(self, prop, grain_id, val):
            """ add new property to a grain in grains. prop: name for new property to add; val : value of new property"""
            g = self.dict[grain_id]
            setattr(g, prop, val)
        
        
        def compute_strain(self, unit_cell_ref):
            """ compute strain relative to a reference cell for all grains is grain_dict"""
            for g in tqdm.tqdm(self.glist):
                g.e11, g.e12, g.e13, g.e22, g.e23, g.e33 = g.eps_sample(unit_cell_ref)  # strain components voigt
                
                # strain invariants
                g.I1 = g.e11+g.e22+g.e33
                #  J2 = 1∕2.trace(e**2) - 1/6.trace(e)**2 = 1/2.trace(s**2) where s is the deviatoric strain tensor
                J2 = 1/2*(g.e11**2 + g.e22**2 + g.e33**2) - 1/6*g.I1**2
                g.J2 = np.max(J2,0) # J2 is positive, so negative values are likely dubious
        
        
        def compute_stress(self, COMP_MAT):
            """ compute elastic stress tensor for all grains in self.grains. strain must have been computed before
            COMP_MAT: compliance matrix to relate strain to stress"""
            for g in tqdm.tqdm(self.glist):
                eps = np.array([g.e11,g.e22,g.e33,g.e23,g.e13,g.e12])
                s = np.matmul(COMP_MAT , np.array([1,1,1,2,2,2])*eps ) 
                g.s11, g.s22, g.s33, g.s23, g.s13, g.s12 = s[0],s[1],s[2],s[3],s[4],s[5]
                # stress invariants
                g.sI1 = g.s11+g.s22+g.s33
                sJ2 = 1/2*(g.s11**2 + g.s22**2 + g.s33**2) - 1/6*g.sI1**2
                g.sJ2 = np.max(sJ2,0) # J2 is positive
                
                                    
                
        def plot_grains_prop(self, prop, s_fact=10, trim=[5,95], out=False, **kwargs):
            """ scatter plot of grains colored by property prop, where (x,y) is grain centroid position and s is grainsize. 
            To plot all strain /stress components at once, type prop='stress' or prop='strain' 
            s_fact: factor to adjust spot size on the scatter plot
            trim: upper/ lower percentile values of prop to adjust colorbar limits """
            
            try:
                cen = self.get_all('centroid')
                gs = self.get_all('grainsize')
            except:
                print('missing grainSize or centroid position')
                return
            
            if prop not in 'strain,stress'.split(','): 
                assert np.all( [hasattr(g, prop) for g in self.glist] )
                colorsc = self.get_all(prop) # color scale defined by selected property
        
                fig = pl.figure(figsize=(6,6))
                ax = fig.add_subplot(111, aspect='equal')
                sc = ax.scatter(cen[:,0], cen[:,1], s = gs/10, c = colorsc, **kwargs)
                ax.set_title(prop)
                cbar = pl.colorbar(sc, ax=ax, orientation='vertical', pad = 0.05, shrink=0.75)
                cbar.formatter.set_powerlimits((0, 0)) 
            
            else:
                if prop == 'strain':
                    titles = 'e11,e12,e13,e22,e23,e33'.split(',')
                else:
                    titles = 's11,s12,s13,s22,s23,s33'.split(',')
                
                fig, ax = pl.subplots(2,3, figsize=(10,6), sharex=True, sharey=True)
                ax = ax.flatten()
                
                for i, (a,t) in enumerate(zip(ax, titles)):
                    a.set_aspect('equal')
                    x = self.get_all(t)
                    low, up = np.percentile(x, (trim[0],trim[1]))
    
                    # plots
                    norm=pl.matplotlib.colors.CenteredNorm(vcenter=np.median(x), halfrange=up)
                    sc = a.scatter(cen[:,0], cen[:,1], s = gs/10, c = x, norm=norm, **kwargs)
                    a.set_title(t)
            
                    # colorbar
                    cbar = pl.colorbar(sc, ax=a, orientation='vertical', pad=0.04, shrink=0.7)
                    cbar.formatter.set_powerlimits((0, 0)) 
            
            # Adjust layout
            fig.tight_layout()
            fig.suptitle('grain scatterplot - '+prop, y=1.0)
                    
            if out:
                return fig
            
            
            
        def hist_grains_prop(self, prop, trim=[5,95], out=False, **kwargs):
            """ histgram of grains property prop. 
            To plot all strain /stress components at once, type prop='stress' or prop='strain' 
            trim: upper/ lower percentile values of prop to adjust histogram limits """
            
            
            if prop not in 'strain,stress'.split(','): 
                assert np.all( [hasattr(g, prop) for g in self.glist] )
                x = self.get_all(prop) 
                low, up = np.percentile(x, (trim[0],trim[1]))
        
                fig = pl.figure(figsize=(6,6))
                ax = fig.add_subplot(111)
                h = ax.hist(x, **kwargs)
                ax.vlines(np.median(x), ymin=0, ymax=h[0].max(), colors='r', label='median')
                ax.set_xlim(low, up)
                ax.set_title(prop)
            
            else:
                
                if prop == 'strain':
                    titles = 'e11,e12,e13,e22,e23,e33'.split(',')
                else:
                    titles = 's11,s12,s13,s22,s23,s33'.split(',')
                
                fig, ax = pl.subplots(2,3, figsize=(10,6))
                ax = ax.flatten()
                for i, (a,t) in enumerate(zip(ax, titles)):
                    x = self.get_all(t)
                    low, up = np.percentile(x, (trim[0],trim[1]))
                    h = a.hist(x, **kwargs)
                    a.vlines(np.median(x), ymin=0, ymax=h[0].max(), colors='r', label='median')
                    a.set_title(t)
            
            # Adjust layout
            fig.tight_layout()
            fig.suptitle('distribution - '+prop, y=1.0)
            
            if out:
                return fig

                

    # methods
    ######################
    def add_data(self, data, dname):
        """ add data column to pixelmap.
        preferentially use numpy array or ndarray with first dimension = nx.ny, but lists may work as well"""
        assert len(data) == self.grid.nx * self.grid.ny
        setattr(self, dname, data)
        
    def rename_data(self, oldname, newname):
        """ rename data column """
        data = self.__getattribute__(oldname)
        setattr(self, newname, data)
        delattr(self, oldname)
        
    def titles(self):
        return [t for t in self.__dict__.keys() if t not in ['grid', 'phases', 'grains', 'h5name'] ]
        
    
    def copy(self):
        """ returns a (deep) copy of the pixelmap """
        pxmap_new = copy.deepcopy(self)
        return pxmap_new
    
    
    def update_pixels(self, xyi_indx, dname, newvals):
        """ update data column dname with new values for a subset of pixel selected by xyi indices, without touching other pixels
        xyi_indx: list of xyi index of pixels to update
        dname: data column to update
        newvals: new values"""
        
        assert dname in self.__dict__.keys()
        
        xyi_indx = np.array(xyi_indx)
        
        # select data column and pixels to update
        dat = self.get(dname)
        dtype = type(dat.flatten()[0])
        pxindx = np.searchsorted(self.xyi, xyi_indx)
        
        
        # update data
        assert newvals.shape[1:] == dat.shape[1:]  # check array shape compatibility
        if len(dat.shape) == 1:  # dat is simple 1d array
            dat[pxindx] = newvals.astype(dtype)
        else:    # nd array of arbitrary size
            dat[pxindx,:] = newvals.astype(dtype)
            
        setattr(self, dname, dat)
        
        
    def update_grains_pxindx(self, mask=None, update_map=False):
        """ update grains pixel indexing (pxindx / xyi_indx), according to criteria defined in mask.
        Allows to remove bad pixels (large misorientation, low npks indexed, high drlv2, etc.) from grain masks
        mask: bool array of same shape as data columns (nx*ny,) to filter bad pixels
        update_map: if True, grain_id in pixelmap will also be updated. MAKE A COPY OF PIXELMAP FIRST,
        OR INITIAL GRAIN INDEXING WILL BE LOST"""
        
        if mask is None:
            mask = np.full(self.xyi.shape, True)
    
        assert mask.shape == self.xyi.shape # make sure mask is the good size
    
        for gi,g in tqdm.tqdm(zip(self.grains.gids, self.grains.glist)):
            gm = np.all([mask, self.grain_id==gi], axis=0)  # select pixels for each grain
            g.pxindx = np.argwhere(gm)[:,0].astype(np.int32)  # reassign pxindx
            g.xyi_indx = self.xyi[g.pxindx].astype(np.int32)    # pixel labeling using XYi indices. needed to select peaks from cf
        # update grain ids
        if update_map:
            self.grain_id[~mask] = -1
        
        
    def filter_by_phase(self, pname):
        """ makes a deep copy of pixelmap and reset all pixels not corresponding to selected phase to zero. 
        also update h5name in new pixelmap to avoid overwriting former file
        input: phase name, must be in self.phases
        output: xmap_p: new xmap """
        
        # make a copy of pixelmap
        xmap_p = self.copy()
        xmap_p.h5name = self.h5name.replace('.h5','_'+pname+'.h5')
        # select phase
        phase = xmap_p.phases.get(pname)
        pid = phase.phase_id
        
        # update columns
        for dname in self.__dict__.keys():
            if dname in ['grid', 'xyi', 'xi', 'yi', 'phases', 'h5name', 'grains']:
                continue
            
            msk = self.phase_id == pid
            array = self.get(dname)
            
            if 'strain' in dname or 'stress' in dname:
                new_array = np.full(array.shape, float('inf'))
            elif dname == 'phase_id' or dname == 'grain_id':
                new_array = np.full(array.shape, -1, dtype=int)
            else:
                new_array = np.zeros_like(array)
                
            new_array[msk] = array[msk]
            xmap_p.add_data(new_array, dname)
        
        # update phases
        for p in xmap_p.phases.pnames:
            if p == 'notIndexed' or p == pname:
                continue
            xmap_p.phases.delete_phase(p)
        
        return xmap_p
    
    
    def add_grains_from_map(self, pname, overwrite=False):
        """ create grains dict directly from pixelmap. requires data columns for grain ids ('grain_id') and pixel UBI
        ('UBI') in pixelmap. It uses a mask to select pixels by grain id, compute median UBI over each grain, create 
        ImageD11.grains from it and add them to pixelmap.grains
        
        pname: name of phase to select. must be in self.phases
        overwrite: re-initialize grains dict. Default is False
        
        NB: grains ubis at this stage are 'mean' ubis obtained by averaging each component of the UBI matrix individually.
        This is quite dodgy, so you might want to refine ubis after this step. For this, you need to map peaks in the original peakfile
        used for indexing to grain in xmap, and then run score_and_refine using these peaks. These two steps are done using the method 
        "refine_pks_to_grains" """
        
        assert 'UBI' in self.__dict__.keys()
              
        # crystal structure
        cs = self.phases.get(pname)
        pid = cs.phase_id
        sym = cs.orix_phase.point_group.laue
        # phase + UBI masks. pm: also select notindexed pixels, because some are included to grain masks by smoothing (mtex)
        pm = np.any([self.phase_id == pid, self.phase_id == -1], axis=0)  
        isUBI = np.asarray( [np.trace(ubi) != 0 for ubi in self.UBI] )   # mask: True if UBI at a given pixel position is not null
      
        # list of unique grain ids for the selected phase
        gid_u = np.unique(self.grain_id[pm]).astype(np.int16)  
        
        # if overwrite, re-initialize grains dict. Otherwise, keep existing grains in grains dict and append new ones
        if overwrite:
            self.grains.__init__()
        
        # loop through unique grain ids, select pixels, compute mean ubi and create grain
        ########################################
        for i in tqdm.tqdm(gid_u):
            # skip notindexed domains
            if i == -1:
                continue  
            # compute median grain properties and create grain.
            gm = self.grain_id==i
            try:   
                ubi_g = np.nanmedian(self.UBI[pm*gm*isUBI], axis=0)
                g = ImageD11.grain.grain(ubi_g)  
            except:
                print('could not create grain for id', i)
                continue
        
            # grain to xmap indexing
            g.gid = i
            g.phase = pname
            g.pxindx = np.argwhere(gm*isUBI)[:,0].astype(np.int32)  # pixel indices in grainmap matching with this grain
            g.grainsize = len(g.pxindx)
            g.surf = g.grainsize * self.grid.pixel_size**2  # grain surface in pixel_unit square
            g.xyi_indx = self.xyi[g.pxindx]    # pixel labeling using XYi indices. needed to select peaks from cf
        
            # phase properties + misorientation
            try:
                og = oq.Orientation.from_matrix(g.U, symmetry = sym)
                opx = oq.Orientation.from_matrix(self.U[gm*isUBI], symmetry=sym)
                misOrientation = og.angle_with(opx, degrees=True)
                g.GOS = np.median(misOrientation)  # grain orientation spread defined as median misorientation over the grain
            except:
                f
                
            # grain centroid
            cx = np.average(self.xi[g.pxindx], weights = self.nindx[g.pxindx])
            cy = np.average(self.yi[g.pxindx], weights = self.nindx[g.pxindx])
            g.centroid = np.array([cx,cy])
                
            # add grain to grains dict
            self.grains.glist.append(g)
            self.grains.gids.append(g.gid)
        
        # update grains dict
        self.grains.dict = dict(zip(self.grains.gids, self.grains.glist))
        
        
    def refine_pks_to_grains(self, pname, cf, hkl_tol, sym, overwrite=False):
        """ run map_grains_to_cf and refine_grains functions on grains from selected phase in xmap.grains.glist.
        -> peaks to grain assignment: updates cf.grain_id column in peakfile and g.pksindx prop in grains
        -> grain ubi refinment : refines grain ubis and returns statistics about % of peaks retained and rotation between old and new orientation
        make sure gdict is also updated
        
        pname : phase name to select
        cf : peakfile
        hkl_tol : tolerance to pass to score_and_refine
        sym : crystal symmetry, used to compute angular shift between new and old orientation
        overwrite : if True, reset 'grain_id' column in cf. default if False
        """
        
        glist = self.grains.select_by_phase(pname)
        
        print('peaks to grains mapping...')
        map_grains_to_cf(glist, cf, overwrite=overwrite)
        print('refining ubis...')
        gids, Npks, prop_indx, misOri = refine_grains(glist, cf, hkl_tol, sym, return_stats=True)
        
        self.grains.dict = dict(zip(self.grains.gids, self.grains.glist))
         
        return gids, Npks, prop_indx, misOri
    

    def map_grain_prop(self, prop, pname):
        """ map a grain property (U, UBI, unitcell, grainsize, etc.) taken from grains in grains.dict to the 2D grid.
        For a grain property p, this function creates a new data column 'p_g' in pixelmap to map this property for each grain on
        the 2D grid. For now, it only works for a single phase (pname): filter pixelmap before, to keep only one phase in the map
        
        To quickly map all six strain / stress components, simply type 'stress" or 'strain' as a prop and the function will
        look for all tensor components and return a single output as a ndarray.
        
        If grain orientation (U matrix) is selected and pixel orientations are available in pixelmap, 
        it will also compute misorientation (angle between grain and pixel orientation in degree)
        """
        
        # Initialize new array
        #####################################
        array_shape = [ self.grid.nx * self.grid.ny ]  # size of pixelmap
        compute_misO = False  # flag to compute misOrientation
        
        if any([s in prop for s in ['stress','strain']]):   # special case for stress / strain
            prop_name = prop+'_g'
            array_shape.append(6)
            newarray = np.full(array_shape, float('inf'))  # default value = inf to avoid confonding zero strain / stress with no data
        else:
            prop_shape = list( self.grains.get_all(prop).shape[1:] )
            prop_name = str(prop)+'_g'  # add g suffix to make it clear it is derived from a grain property
            array_shape.extend(prop_shape)
        
            # special values to initialize grain/phase id: -1. Default: 0
            if any([s in prop for s in 'gid,grain_id,phase_id'.split(',')]):
                init_val = -1
            elif any([s in prop for s in 'I1,J2'.split(',')]):
                init_val = float('inf')
            else:
                init_val = 0
            
            # dtype: float (default) or int 
            if 'int' in str( type(self.grains.get_all(prop)[0]) ):
                dtype = 'int'
            else:
                dtype = 'float'
            newarray = np.full(array_shape, init_val, dtype = dtype)
        
        #if grain orientation U is selected, try to compute misorientation as well
        if prop == 'U':
            try:
                U_px = self.get('U')
                misO = np.full(self.xyi.shape, 180, dtype=float)   # default misorientation to 180°
                # crystal structure
                cs = self.phases.get(pname)
                sym = cs.orix_phase.point_group.laue
                compute_misO = True
            except:
                print('No orientation data in pixelmap, or name not recognized (must be U). Will not compute misorientation.')
                
        # update with values from grains in graindict
        #####################################
        gid_map = self.get('grain_id')
        
        for gi,g in tqdm.tqdm(zip(self.grains.gids, self.grains.glist)):
            gm = np.argwhere(gid_map == gi).T[0]  # grain mask
            
            # fill newarray. Different cases depending of prop shape  + strain / stress cases
            if prop == 'strain':
                eps = np.array([g.e11, g.e12, g.e13, g.e22, g.e23, g.e33])
                newarray[gm,:] = eps
            elif prop == 'stress':
                sigma = np.array([g.s11, g.s12, g.s13, g.s22, g.s23, g.s33])
                newarray[gm,:] = sigma
            elif len(prop_shape) == 0:
                newarray[gm] = self.grains.get(prop,gi)
            else:
                newarray[gm,:] = self.grains.get(prop,gi)
                
            # misorientation
            if prop=='U' and compute_misO:
                og = oq.Orientation.from_matrix(newarray[gm], symmetry=sym)
                opx =  oq.Orientation.from_matrix(self.get('U')[gm], symmetry=sym)
                misO[gm] = og.angle_with(opx, degrees=True)
        
        # add newarray to pixelmap
        self.add_data(newarray, prop_name)
        if compute_misO:
            self.add_data(misO, 'misOrientation')

                                                 

# TO DO: alterative method to define grains: load grain list with grain ids and grain masks and fill pixelmap using grain masks
#    def add_grains_from_glist(self, glist):


        
    def plot(self, dname, save=False, hide_cbar=False, out=False, **kwargs):
        """ plot data from column dname. data in self.dname must be a single array"""
        nx, ny = self.grid.nx, self.grid.ny
        xb, yb = self.grid.xbins, self.grid.ybins
        dat = self.get(dname).reshape(nx, ny)
        
        fig = pl.figure(figsize=(6,6))
        ax = fig.add_subplot(111, aspect ='equal')
        ax.set_axis_off()
        im = ax.pcolormesh(xb, yb, dat, **kwargs)
        ax.set_title(dname)
        ax.add_artist(self.grid.scalebar())
        
        if not hide_cbar:
            fig.suptitle(self.h5name.split('/')[-1].split('.h')[0], y=.9)
            if 'phase_id' in dname:
                cbar = pl.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7, ticks = self.phases.pids)
                cbar.ax.set_yticklabels(self.phases.pnames)
            else:
                cbar = pl.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7, label=dname)
                cbar.formatter.set_powerlimits((-1, 1)) 
        
        if save:
            fname = self.h5name.replace('.h5', '_'+dname+'.png', dpi=150)
            fig.savefig(fname, format='png') 
        if out:
            return fig
            
            
            
    def plot_voigt_tensor(self, dname, trim = [2,98], save=False, show_axis=True, out=False, **kwargs):
        """ plot all components of strain / stress tensor (voigt notation)
        dname: column name to plot. data must be a Nx8 array with strain / stress components in the following order:
        e11, e12, e13, e22, e23, e33, I1, J2
        trim : trim strain / stress distribution for each components to selected percentile values to exclude outliers""" 
        
        nx, ny = self.grid.nx, self.grid.ny 
        xb, yb = self.grid.xbins, self.grid.ybins
        voigt_tensor = self.get(dname)
        
        # figures layout
        fig, ax = pl.subplots(2,3, figsize=(10,6), sharex=True, sharey=True)
        ax = ax.flatten()
        
        if 'strain' in dname:
            titles = 'e11,e12,e13,e22,e23,e33'.split(',')
        else:
            titles = 's11,s12,s13,s22,s23,s33'.split(',')
        
        # loop through strain / stress components and plot them in map + histogram
        for i, (a,t) in enumerate(zip(ax, titles)):
            a.set_aspect('equal')
            a.set_axis_off()
            x = voigt_tensor[:,i].reshape(nx,ny)
            x_u = np.unique(voigt_tensor[voigt_tensor != float('inf')])  # select unique values to get distribution across grains  
            low, up = np.percentile(x_u, (trim[0],trim[1]))
    
            # plots
            norm=pl.matplotlib.colors.CenteredNorm(vcenter=np.median(x_u), halfrange=up)
            im = a.pcolormesh(xb, yb, x, norm=norm, **kwargs)
            a.set_title(t)
            
            # colorbar
            cbar = pl.colorbar(im, ax=a, orientation='vertical', pad=0.04, shrink=0.7)
            cbar.formatter.set_powerlimits((0, 0)) 
            
        # Adjust layout
        fig.tight_layout()
        dsname = self.h5name.split('/')[-1].split('.h')[0]
        fig.suptitle('grainmap '+dname+' - '+dsname, y=1.0)

        if save:
            fname = self.h5name.replace('.h5', '_'+dname+'.png')
            fig.savefig(fname, format='png', dpi=300)
            
        if out:
            return fig
            
        
        
    def hist_voigt_tensor(self, dname, trim=[2,98], nbins=100, save=False, out=False, **kwargs):
        """ histogram of all six strain / stress voigt tensor components """
        voigt_tensor = self.get(dname)
        # figures layout
        fig, ax = pl.subplots(2,3, figsize=(10,6))
        ax = ax.flatten()
        
        if 'strain' in dname:
            titles = 'e11,e12,e13,e22,e23,e33'.split(',')
            lab = dname.replace('strain', 'eps')
        else:
            titles = 's11,s12,s13,s22,s23,s33'.split(',')
            lab = dname.replace('stress', 'sigma')
          
        for i, (a,t) in enumerate(zip(ax, titles)):
            x = voigt_tensor[:,i]
            x_c = x[x != float('inf')]
            low, up = np.percentile(x_c, (trim[0],trim[1]))
            bins = np.linspace(low, up, nbins)
            h = a.hist(x_c, label=lab, bins=bins, **kwargs)
            a.vlines(x=np.median(x_c), ymin=0, ymax=h[0].max(), colors='navy', label='median')
            a.set_title(t)
            a.set_xlim(np.percentile(x_c, (1,99)))
            a.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                
        fig.tight_layout()
        dsname = self.h5name.split('/')[-1].split('.h')[0]
        fig.suptitle('hist '+dname+' - '+dsname, y=1.0)
            
        if save:
            fname = self.h5name.replace('.h5', '_'+dname+'_hist.png')
            fig.savefig(fname, format='pdf', dpi=300)
        if out:
            return fig
            
    
    
    def plot_ipf_map(self, dname, phase, ipfdir = [0,0,1], save=False, out=False, **kwargs):
        """ plot orientation color map (using orix)
        dname: data column, must be a Nx3x3 ndarray of orientation matrices
        phase : name of the phase to plot. must be in self.phases
        ipfdir: direction for the ipf colorkey. must be a 3x1 vctor [x,y,z]. Default: z-vector
        out: return figure"""
        
        # select phase properties
        assert phase in self.phases.pnames
        cs = self.phases.get(phase)
        cs.str_diffpy.title = cs.name
        sym = cs.orix_phase.point_group.laue
        ipf_key = opl.IPFColorKeyTSL(sym, direction=ovec.Vector3d(ipfdir))
        
        #convert matrix orientation to quaternions
        U = self.get(dname)
        ori = oq.Orientation.from_matrix(U, symmetry=sym)
        
        # select phase id
        phase_id = np.where(self.phase_id==cs.phase_id, cs.phase_id,-1)
        
        # orix crystal map
        orix_map = ocm.CrystalMap(rotations = ori,
                      phase_id = phase_id,
                      x = self.xi,
                      y = self.yi,
                      phase_list = ocm.PhaseList(space_groups=[cs.spg_no],
                                                 structures=[cs.str_diffpy]),
                      scan_unit = self.grid.pixel_unit)
        
        # select orientations to plot
        o = orix_map[phase].orientations
        rgb = ipf_key.orientation2color(o)
        
        # plot ipf map
        pl.matplotlib.rcParams.update({'font.size': 10})
        fig = pl.figure(figsize=(8,8))

        ax0=fig.add_subplot(111, aspect='equal', projection='plot_map')
        ax0.set_axis_off()
        
        ax0.plot_map(orix_map[phase], rgb, scalebar=False, **kwargs)
        ax0.title.set_text(phase+' - ipf map '+str(ipfdir))

        # plot color key
        pl.matplotlib.rcParams.update({'font.size': 4})
        fig.subplots_adjust(right=0.75)
        ax1 = fig.add_axes([0.8, 0.25, 0.15, 0.15], projection='ipf',  symmetry=sym)
        ax1.plot_ipf_color_key(show_title=False)

        dsname = self.h5name.split('/')[-1].split('.h')[0]
        
        if save:
            ipfd_str = ''.join(map(str, ipfdir))
            fname = self.h5name.replace('.h5', '_'+phase+'_ipf_'+ipfd_str+'.pdf')
            fig.savefig(fname, format='pdf', dpi=300)

        pl.matplotlib.rcParams.update({'font.size': 10})
        fig.suptitle(dsname, y=0.9)
        
        if out:
            return fig
        

            
    # TO DO : Make it compress better. Can use less space-consuming dtypes on some arrays.  Need also to update save grains function
    def save_to_hdf5(self, h5name=None, debug=0):
        """ save pixelmap to hdf5 format"""
        # save path
        if h5name is None:
            try:
                h5name = self.h5name
                h5name[0]
            except:
                print("please enter a path for the h5 file")
        
        with h5py.File(h5name, 'w') as f:
            
            f.attrs['h5path'] = h5name
            
            # Save grid information
            grid_group = f.create_group('grid')
            
            attr = 'pixel_size', 'pixel_unit'
            for item in self.grid.__dict__.keys():
                if item in attr:
                    grid_group.attrs[item] = self.grid.__getattribute__(item)
                else:
                    data = self.grid.__getattribute__(item)
                    grid_group.create_dataset(item, data = data, dtype = int) 
            
            # Save phases information 
            phases_group = f.create_group('phases')
            for pname, pid, path in zip(self.phases.pnames, self.phases.pids, self.phases.cif_paths):
                phase = phases_group.create_group(pname)
                phase.attrs['pid'] = pid
                phase.attrs['cif_path'] = path
                
            # save grains
            save_grains_dict(self.grains.dict, h5name)

            
            # Save other data
            skip = ['grid', 'xi', 'yi', 'phases', 'pksind', 'h5name', 'grains']
            
            for item in self.__dict__.keys():
                if item in skip or '_g' in item:  # do not save map columns in skip list or map columns exported from grains dict
                    continue
                data = self.__getattribute__(item)
                if debug:
                    print(item) 
                f.create_dataset(item, data = data, dtype = type(data.flatten()[0]))

        print("Pixelmap saved to:", h5name)
        


##########################    
        
    
def load_from_hdf5(h5name):
    """ load pixelmap for hdf5 file"""
    with h5py.File(h5name, 'r') as f:
        # Load grid information
        xbins  = f['grid/xbins'][()]
        ybins  = f['grid/xbins'][()]
        pxsize = f['grid'].attrs['pixel_size']
        pxunit = f['grid'].attrs['pixel_unit']

        # Load phases information
        pnames, pids, paths = [], [], []
        for pname, phase in f['phases'].items():
            pid = phase.attrs['pid']
            cif_path = phase.attrs['cif_path']
            pnames.append(pname)
            pids.append(pid)
            paths.append(cif_path)
            
        # Load grains
        if 'grains' in list(f.keys()):
            grainsdict = load_grains_dict(h5name)
        else:
            grainsdict = {}
    
        
        # Load other data
        skip = ['grid', 'phases', 'grains']
        data = {}
        for item in f.keys():
            if item in skip:
                continue
            data[item] = f[item][()]
    
    # Create a new Pixelmap object 
    pixelmap = Pixelmap(xbins, ybins, h5name=h5name)
    # update grid
    pixelmap.grid.pixel_size = pxsize
    pixelmap.grid.pixel_unit = pxunit
    # Add phases to Pixelmap
    for pname, pid, path in zip(pnames, pids, paths):
        if pname == "notIndexed":
            continue
        cs = crystal_structure.CS(pname,pid,path)
        pixelmap.phases.add_phase(pname, cs)
    # Add data
    for d in data.keys():
        pixelmap.add_data(data[d], d)
    # Add grainsdict
    pixelmap.grains.dict = grainsdict
    pixelmap.grains.gids = list(grainsdict.keys())
    pixelmap.grains.glist = list(grainsdict.values())
        
    return pixelmap




def save_grains_dict(grainsdict, h5name, debug=0):
    """ save grain dictionnary to hdf5. Append data to existing h5 file"""
    
    with h5py.File( h5name, 'a') as hout:
        # Delete the existing 'grains' group if it already exists
        if 'grains' in hout:
            del hout['grains']
            
        grains_grp = hout.create_group('grains')

        for i,g in grainsdict.items():
            gr = grains_grp.create_group(str(i))    
            gprops = [p for p in list(g.__dict__.keys()) if not p.startswith('_')]  # list grain properties, skip _U, _UB etc. (dependent)
            
            if debug:
                print(gprops)
            
            for p in gprops:
                attr = g.__getattribute__(p)
                if attr is None:   # skip empty attributes
                    continue
                # find data type + shape
                if np.issubdtype(type(attr), np.integer):
                    dtype = 'int'
                    shape = None
                elif np.issubdtype(type(attr), np.floating):
                    dtype = 'float'
                    shape = None
                elif isinstance(attr, str):
                    dtype = str
                    shape = None
                else:
                    attr = np.array(attr)
                    shape = attr.shape
                    try:
                        dtype = type(attr.flatten()[0])
                    except:    # occurs if attr is empty
                        dtype = float
                
                if debug:
                    print(p,dtype)
                # save arrays as datasets
                if shape is not None: 
                    gr.create_dataset(p, data = attr, dtype = dtype) 
                else:
                    gr.attrs.update({ p : attr})


def load_grains_dict(h5name):
    grainsdict = {}
    with h5py.File(h5name,'r') as f:
        if 'grains' in list(f.keys()):
            grains = f['grains']
        else:
            grains = f
            
        gids = list(grains.keys())
        gids.sort(key = lambda i: int(i))
        
        # loop through grain ids and load data
        for gi in gids:
            gr = grains[gi]
            # create grain from ubi
            g = ImageD11.grain.grain(gr['ubi'])
            # load other properties
            for prop, vals in gr.items():
                if prop == 'ubi':
                    continue
                ary = vals[()]
                setattr(g, prop, ary)
            # add grain attributes
            for attr, val_attr in gr.attrs.items():
                setattr(g, attr, val_attr)
                        
            # add grain to grainsdict
            grainsdict[int(gi)] = g
            
    return grainsdict
        
