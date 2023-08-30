# es1190_3dxrd
This repository contains several python functions used to process 3D X-Ray diffraction (3DXRD) data for the ERC BREAK project (https://www.mn.uio.no/njord/english/research/projects/break/ - proposal es1190 at ESRF). 3DXRD data analysis is built upon the python package ImageD11 (https://github.com/FABLE-3DXRD/ImageD11).

Functions are split into diferent modules:

- crystal_structure.py: A class to store crystal structure information, imported from cif files. It is built upon diffpy and Dans_diffraction packages.
  
- friedel_pairs.py: Friedel Pairs are symmetrical (hkl) ; (-h,-k,-l) diffraction vectors observed 180 degree apart when rotating the sample along the z-axis. Using these paired diffraction peaks has several advantages, including a better assignment of diffraction peaks to hkl rings for large samples, an increase in the accuracy of diffraction vectors coordinates, plus it allows the use of improved algorithms for grain indexing and reconstruction (e.g [Ludwig et al., 2009](https://pubs.aip.org/aip/rsi/article/80/3/033905/351823)). The module contains functions to identify Friedel pairs in a dataset and compute diffraction vectors using Friedel pair positions. For now, only works for a scanning 3DXRD setting, in which scanning is done with a thin pencil beam with translations along the y-axis to get diffraction data from all grains in the sample.
  
- grainmapping.py: Functions used for peak indexing, grains centre of mass fittin and grainshape reconstruction

- peakfiles.py: General functions to do manipulations on ImageD11 columnfiles. Read description in the file to see what the different functions do.

- pixelmap.py: Module and work on 2D pixel maps. It defines a class "Pixelmap", which stores data on a 2D grid. This module is useful to do pixel-by-pixel indexing and phase assignment and plot grain maps 

- rawimage.py: Functions to be used on raw hdf5 output from the detector. It includes functions to read data from different scans, plot sinogram and do filtered back projection +
background estimation function
