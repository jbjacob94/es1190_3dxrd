import os, sys, h5py
import numpy as np, pylab as pl, math as m

import fast_histogram
import skimage.transform
import scipy.spatial
from scipy.stats import gaussian_kde

from ImageD11 import unitcell, columnfile, transform, sparseframe, cImageD11, refinegrains
from ImageD11.grain import grain, write_grain_file, read_grain_file

from diffpy.structure import Atom, Lattice, Structure
from orix import data, io, plot, crystal_map as ocm, quaternion as oq, vector as ovec

from id11_utils import peakfiles, crystal_structure


# Collection of functions to work on 2D pixel maps. Includes Pixelmap Class: A class to store data on a 2D pixel grid. It is used to do pixel-by-pixel indexing and phase assignment and plot grain properties (orientation, indexing quality, etc.) on a 2D map
#################################################################################################

# Note on pixel-by-pixel indexing strategy:
##################################
#The classic indexing strategy with ImageD11 consists in finding g-vectors that are compatible in orientation, considering a given crystal symmetry and lattice parameters. However, this process becomes very inefficient as the number of peaks in the dataset grows very large, because it does not take into account positional information on grains, which is refined in a later stage. Using Friedel pairs, it is possible to relocate the source of diffraction in the sample reference frame (as long as the sample does not consists in a few very large grains, in this case peak position within the grain is poorly defined). Thus, it is possible to assign each peak a pixel on a 2d xy grid and do 'pixel-by'pixel' indexing. This allows to take into account precise information on peak location and makes the indexing process much faster, as the number of g-vectors to match for each pixel is only a small subset of the entire dataset. This pixel-by-pixel processing can also be done for phase identification (for polyphased samples) prior to indexing. This gives significantly better results than separating phases using only a simple threshold on two-theta angle


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



# to add to pixelmap: 
###################

# index_phase_to_pixel(args=(cf_to_index, xi, yi, minpks))  from 005_label_pixelmap
# find_pixel_orientations(args=(to_index, xi, yi, etc.))    from 006_index_pixelmap
#
# methods: 
# - copy
# - update column  Need to secure operations to avoid overwriting a full column by mistake
# list dataformat for each column: int16, int32, float
# - convert to orix crystalmap
# - plot data 

# Pixelmap object
###########################################################################

class Pixelmap:
    """ A class to store pixel information on a 2d grid """
    def __init__(self, xbins, ybins, h5name=None):
        # grid
        self.grid = self.GRID(xbins, ybins)
        self.xyi = np.asarray([i + 1000*j for i in xbins for j in ybins]).astype(np.int32)
        self.xi = self.xyi % 1000
        self.yi = self.xyi // 1000
        # phase labeling  + crystal structure information
        self.phases = self.PHASES()   # class storing crystal structure information on phases in pixelmap
        self.phase_id = np.full(self.xyi.shape, -1, dtype=np.int8)   # map of phase_ids
    
        self.h5name = h5name
    
    def __str__(self):
        return f"Pixelmap: size:{self.grid.shape}, phases: {self.phases.pnames}, phase_ids: {self.phases.pids}"
    
    def add_data(self, data, dname):
        """ add data column to pixelmap.
        preferentially use numpy array or ndarray with first dimension = nx.ny, but lists may work as well"""
        assert len(data) == self.grid.nx * self.grid.ny
        setattr(self, dname, data)
    
    def copy(self):
        """ returns a (deep) copy of the pixelmap """
        pxnew = Pixelmap(self.grid.xbins, self.grid.ybins)
        for pname, pid, path in zip(self.phases.pnames, self.phases.pids, self.phases.cif_paths):
            if pname == "notIndexed":
                continue
            cs = crystal_structure.CS(pname,pid,path)
            pxnew.phases.add_phase(pname, cs)
            
        skip = ['grid', 'xyi', 'xi', 'yi', 'h5name', 'phases']
        for k,v in self.__dict__.items():
            if k in skip:
                continue
            pxnew.add_data(v, k)
        return pxnew
    
 #   def update_pixels(xyi_indx, dname):
 #       """ update data column dname for a subset of pixel selected by xyi indices"""
 #       pxindx = 
    
    
    
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
            
            
            
    class PHASES:
        # crystal structures
        def __init__(self):
            self.notIndexed = crystal_structure.CS(name='notIndexed')
            self.pnames = ['notIndexed']
            self.pids = [-1]
            self.cif_paths = ['']
            
        def __str__(self):
            return f"phases: {self.pnames}"
            
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
            cs = self.__getattribute__(pname)
            pid = cs.__getattribute__('phase_id')
            path = cs.__getattribute__('cif_path')
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
            
            # Save other data
            skip = ['grid', 'xi', 'yi', 'phases', 'pksind', 'h5name']
            
            for item in self.__dict__.keys():
                if item in skip:
                    continue
                data = self.__getattribute__(item)
                if debug:
                    print(item) 
                f.create_dataset(item, data = data, dtype = type(data.flatten()[0]))

        print("Pixelmap saved to:", h5name)

        
def load_from_hdf5(h5name):
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
        
        # Load other data
        skip = ['grid', 'phases']
        data = {}
        for item in f.keys():
            if item in skip:
                continue
            data[item] = f[item][()]
    
    # Create a new Pixelmap object 
    pixelmap = Pixelmap(xbins, ybins, h5name=h5name)
    
    # Add phases to Pixelmap
    for pname, pid, path in zip(pnames, pids, paths):
        if pname == "notIndexed":
            continue
        cs = crystal_structure.CS(pname,pid,path)
        pixelmap.phases.add_phase(pname, cs)
    # Add data
    for d in data.keys():
        pixelmap.add_data(data[d], d)
        
    return pixelmap


            
        
