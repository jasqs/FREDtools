FRED tools repository
================================

FRED tools is a collection of python functions for image manipulation and analysis. The basic methods have been developed for analysis of the images produced by the Monte Carlo FRED (www.fred-mc.org) in MetaImage format (*.mha, *.mhd), but they can be applied for images in other formats, e.g. dicom.

Basic Concept
----------------------------
The image in the FRED tools is understood to mean scalar or vector image of any dimension from 2D to 5D. All the images read or processed by the functions are [SimpleITK](www.simpleitk.readthedocs.io) images, and any SinmpleITK routines can be applied. Check [SimpleITK filters](www.simpleitk.readthedocs.io/en/master/filters.html) for available methods of filtering, registration, etc. Because the smallest image dimension supported by SimpleITK is 2D, line profiles, i.e. 1D images are saved as SimpleITK images with the size with only one of the dimension different than one.

Most of the functions have a bool argument ``displayInfo`` (default ``displayInfo=False``) which can be used for displaying summary of the function results.

Documentation and tutorial
----------------------------
The documentation of the functions implemented in FRED tools is available at [fredtools.ifj.edu.pl](https://www.fredtools.ifj.edu.pl) or on the [GitHub repository](https://github.com/jasqs/FREDtools).

A simple [tutorial](https://github.com/jasqs/FREDtools/blob/main/examples/FREDtools%20Tutorial.ipynb) has been prepared to help with starting using FRED tools.


Installation
----------------------------
The stable version of FRED tools is available via pip.

For new installation:

  $ pip install fredtools

To update existing installation:

$ pip install --upgrade fredtools

The development version is available on GitHub.

   $ git clone jasqs/FREDtools


Development
----------------------------
Everyone is invited to support the development of the FRED tools and to add functionalities needed for a specific application.
All the new functions or classes should be documented according to numpydoc style. Check [numpydoc website](www.numpydoc.readthedocs.io) for the style guide and examples.

