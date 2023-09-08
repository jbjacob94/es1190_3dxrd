import os, sys, h5py
import numpy as np, pylab as pl, math as m
import scipy.spatial, scipy.signal

import diffpy.structure
import Dans_Diffraction as dif

from orix import plot as opl, crystal_map as ocm, vector as ovec
from ImageD11 import unitcell, columnfile

from id11_utils import peakfiles


# class to store crystal structure information, imported from a cif file
class CS:
    def __init__(self, name, pid=-1, cif_path=''):
        self.name = name
        self.cif_path = cif_path
        self.phase_id = pid
        self.color = 'red'
        self.spg = None
        self.spg_no = None
        self.lattice_type = None
        self.cell = []
        self.str_dans = None
        self.str_diffpy = None
        self.orix_phase = None
        
        if self.phase_id !=-1:
            self.add_data_from_cif()
        
        
    def __str__(self):
        return f"CS: {self.name}, phase_id: {self.phase_id}, spg:{self.spg}, spg_no:{self.spg_no}, lattice:{self.cell}"
    
    def get(self,prop):
        return self.__getattribute__(prop)
        
        
    def add_data_from_cif(self):
        """import structure information from cif file and compute different properties.
        Requires diffpy.structure and Dans_diffraction modules
        There is significant overlap between str_dans and str_diffpy, but these two objects do not contain exactly the same information and one or the other is needed depending on what you want
        to do. To create an orix Phase object (needed to get ipfkey, plot crystalmap etc.), only str_diffpy works. str_dans has lot of useful features, including computation of the theoretical
        powder pattern of a given crystal structure."""
        
        if not os.path.exists(self.cif_path):
            print('incorrect path for crystal structure file')
        
        # load with Dans_diffraction and diffpy.structure 
        try:
            self.str_dans = dif.Crystal(self.cif_path)
            self.str_diffpy = diffpy.structure.loadStructure(self.cif_path, fmt='cif')
        except:
            print(self.name, ': No cif file found, or maybe it is corrupted')
    
        # symmetry info (space group nb + name) + unit cell
        self.spg = self.str_dans.Symmetry.spacegroup_name().split(' ')[0]
        try:
            self.spg_no = int(self.str_dans.Symmetry.spacegroup_number)
        except:
            print('No space group number in cif file')
    
        self.lattice_type = self.str_dans.Symmetry.spacegroup_name()[0]
        cell = self.str_dans.Cell
        self.cell = [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma]
        self.orix_phase = ocm.Phase(name=self.name, space_group=self.spg_no, structure=self.str_diffpy, color=self.color)
        
        
    def get_ipfkey(self, direction = ovec.Vector3d.zvector()):
        self.ipfkey = opl.IPFColorKeyTSL(self.orix_phase.point_group, direction = direction)
        
        
    def compute_powder_spec(self, wl, min_tth=0, max_tth=25, doplot=False):
        """simulate powder diffraction spectrum from cif data (using Dans_dif package)"""
    
        E_kev = peakfiles.get_Xray_energy(wl)
        self.str_dans.Scatter.setup_scatter(scattering_type='x-ray', energy_kev=E_kev, min_twotheta=min_tth, max_twotheta=max_tth)
    
        # simulate powder pattern
        tth, ints, _ = self.str_dans.Scatter.powder(units='twotheta')  # tth, Intensity coordinates of powder pattern
        ints = ints/ints.max()  # normalize intensity
        
        self.powder_spec = [tth, ints]
        
        if doplot:
            pl.figure()
            pl.plot(tth, ints,'-')
            pl.xlabel('tth deg')
            pl.ylabel('normalized Intensity')
            pl.title('xray powder spectrum - ' + self.name)
    

    def find_strongest_peaks(self, Imin=0.1, Nmax=30, doplot=False):
        """  do peaksearch on powder spectrum and return the N-strongest peaks sorted by decreasing intensity
        Imin: minimum intensity threshold for a peak
        Nmax: N strongest peaks to select"""
        
        try:
            tth, I = self.powder_spec[0], self.powder_spec[1]
        except:
            print('No spectrum data. Compute powder spectrum first')
        
        pksindx, pksI = scipy.signal.find_peaks(I, height=Imin)
        pks = tth[pksindx].tolist()
        pksI = list(pksI.values())[0].tolist()  # pksI returned in a dict. convert it to an array
    
        # sort peaks by intensity
        pks_sorted = [l1 for (l2, l1) in sorted(zip(pksI,pks), key=lambda x: x[0], reverse=True)]
        pksI_sorted = [l2 for (l2, l1) in sorted(zip(pksI,pks), key=lambda x: x[0], reverse=True)]
    
        # take only the most intense peaks
        if len(pks_sorted) > Nmax:
            pks_sorted = pks_sorted[:Nmax]
            pksI_sorted = pks_sorted[:Nmax]
    
        self.strong_peaks = [pks_sorted, pksI_sorted]
        
        if doplot:
            pl.figure()
            pl.plot(tth, I,'-')
            pl.vlines(x=pks_sorted, ymin=0, ymax=1, colors='red', lw=.5)
            pl.xlabel('tth deg')
            pl.ylabel('normalized Intensity')
            pl.title('strongest diffraction peaks - ' + self.name)
            
            
def load_CS_from_cif(cif_path, name='', pid = -1):
    """ create a CS object directly from cif file"""
    cs = CS(name, pid, cif_path)
    
    if name == '':
        cs.name = cs.str_dans.name
        try:
            cs.name[0]
        except:
            print('no phase name found in cif file. Please enter a name for the structure')
    return cs
    
