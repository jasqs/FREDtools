FRED tools documentation
================================

FRED tools is a collection of python functions for image manipulation and analysis. The basic methods have been developed for analysis of the images produced by the `Monte Carlo FRED <http://www.fred-mc.org>`_ in MetaImage format (*.mha, *.mhd), but they can be applied for images in other formats, e.g. dicom.

Basic Concept
----------------------------
The image in the FRED tools is understood to mean scalar or vector image of any dimension from 2D to 5D. All the images read or processed by the functions are `SimpleITK <https://simpleitk.readthedocs.io>`_ images, and any SinmpleITK routines can be applied. Check `SimpleITK filters <https://simpleitk.readthedocs.io/en/master/filters.html>`_ for available methods of filtering, registration, etc. Because the smallest image dimension supported by SimpleITK is 2D, line profiles, i.e. 1D images are saved as SimpleITK images with the size of only one of the dimension different than one.

Most of the functions have a bool argument ``displayInfo`` (default ``displayInfo=False``) which can be used for displaying a summary of the function results.

Installation
----------------------------
The stable version of FRED tools is available via pip.

.. code:: bash

   $ pip install fredtools

The development version is available on GitHub.

.. code:: bash

   $ git clone jasqs/FREDtools


Development
----------------------------
Everyone is invited to support the development of the FRED tools and to add functionalities needed for a specific application.
All the new functions or classes should be documented according to numpydoc style. Check `numpydoc website <https://numpydoc.readthedocs.io>`_ for the style guide and examples.

The FRED tools documentation is written in reStructuredText format and build with sphinx in version 4.0.2 using sphinx_rtd_theme template in version 0.5.2.

.. toctree::
   :maxdepth: 1

   Introduction/Collaboration
   Introduction/Citation
   Introduction/Changelog

.. toctree::
   :maxdepth: 1
   :caption: Documentation:

   Documentation/ReadWrite
   Documentation/ImageAnalyse
   Documentation/dvhAnalyse
   Documentation/GettingSubimage
   Documentation/ImageManipulate
   Documentation/simTools
