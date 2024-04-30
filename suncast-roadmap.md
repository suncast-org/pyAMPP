***PyAMPP Roadmap***

--------------
**Overarching Goal**

`PyAMPP`is an automatic model production pipeline (AMPP). The workflow is initiated by a minimal set of user-defined parameters such as heliocentric or Carrington coordinates, time, and desired spatial resolution and field of view (FOV). Then, AMPP downloads the required input data from the Solar Dynamic Observatory (SDO, Scherrer et al. 2012) repository and produces GX Simulator-compatible 3D magnetic field models of solar active regions. The package features a very fast NLFFF extrapolation engine (Stupishin 2020), and a fast non-uniform grid geometrical rendering module, allowing for efficient handling of reasonably large models.

`PyAMPP` has the objectives to be an

* Open-Source python package for everybody without cumbersome licence restriction and commercial interests
* Easy-to-Use modular package system
* Flexible framework for exploratory data analysis in cloud and precipitation research
* Robust interface to existing modern Python tools (like xarray, scipy, scikit-image, astropy, sunpy)
--------------
**Structure of `PyAMPP`**

The element of `PyAMPP` are laid out as the following: 

1. Selection of time, position and spatial resolution of the model
2. Automatic download of SDO HMI/AIA data closest to the time requested
3. WCS coordinate transformation of HMI data to create the base of the subsequent extrapolations
4. Creation of an empty-box structure containing a WCS-compatible index, LOS Bz, Ic, and the requested AIA UV/EUV reference maps
5. Initial potential field extrapolation
6. NLFFF optimization
7. Computation of the length and averaged magnetic field for all voxels crossed by closed magnetic field lines, to be used by GX Simulator to assign parametrized differential emission measure and density and temperature distribution models of the corona
   -. Nabil is working with Alexey to make a wrapper
8. Adding non-LTE density and temperature distribution models of the chromosphere