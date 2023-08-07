# es1190_3dxrd
This repository contains several python functions used to process 3D X-Ray diffraction (3DXRD) data for the ERC BREAK project (https://www.mn.uio.no/njord/english/research/projects/break/ - proposal es1190 at ESRF). 3DXRD data analysis is built upon the python package ImageD11 (https://github.com/FABLE-3DXRD/ImageD11).

Functions are split into diferent modules:
- rawimage.py: Functions to be used on raw hdf5 output from the detector. It includes functions to read data from different scans, plot sinogram and do filtered back projection +
background estimation function
- peakfiles.py: general functions aimed to do manipulations on ImageD11 columnfiles. A bit messy, should be cleaned up and split into different modules. Read description in the file to see what the different functions do
- friedel_pairs.py: our data processing relies a lot on the use of "Friedel Pairs", ie paired (hkl) - (-h,-k,-l) reflection observed 180 degree apart in rotation. This has several advantages, including a better assignment of diffraction peaks to hkl rings for large samples, an increase in the accuracy of diffraction vectors determination, and it allows the use of improved algorithms for grain indexing and reconstruction (Ludwig et al., 2009). The module contains functions to identify Friedel pairs in a dataset and compute diffraction vectors using Friedel pair positions.
- grainmapping.py: Functions used for peak indexing, grains centre of mass fittin and grainshape reconstruction

