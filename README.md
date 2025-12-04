# Project Name
Sequential RF-AC H-k Phase-weighted Stacking

# Descriptions
Sequential RF-AC H-k Phase-weighted Stacking utilize the seismic converted phases from Receiver Functions and the seismic reflected phases from coda Autocorrelations of teleseismic P-waves to constrain the thicknesses and Vp/Vs ratios of the sediments and the underlying crust. It is a powerfull tool to analyze the complex converted phases in RFs in presence of the low-velocity sediments. 
(1) HkVp_RFAC_pybind11/src
package codes
The core code  stacking the H-k images is writed by C++ programming language and then is packaged by Python codes using pybind11.
(2) HkVp_RFAC_pybind11/jupyter_notebooks
Some scripts to do Sequential RF-AC H-k Phase-weighted Stacking for synthetic data and observation data. This directory also include some plotting scripts calling Matplotlib and pygmt packages.


# Prerequisites
Numpy, Matplotlib, Obspy

# Installation Steps

```bash
cd $ProjectRootDir
python setup.py install
```



