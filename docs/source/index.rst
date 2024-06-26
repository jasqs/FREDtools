FRED tools v. |fredtoolsVersion| documentation
================================================================

FRED tools is a collection of python functions for image manipulation and analysis. The basic methods have been developed for analysis of the images produced by the `Monte Carlo FRED <http://www.fred-mc.org>`_ in MetaImage format (\*.mha, \*.mhd), but they can be applied for images in other formats, e.g. dicom.

Basic Concept
----------------------------
The image in the FRED tools is understood to mean scalar or vector image of any dimension from 2D to 5D. All the images read or processed by the functions are `SimpleITK <https://simpleitk.readthedocs.io>`_ images, and any SimpleITK routines can be applied. Check `SimpleITK filters <https://simpleitk.readthedocs.io/en/master/filters.html>`_ for available methods of filtering, registration, etc. Because the smallest image dimension supported by SimpleITK is 2D, line profiles, i.e. 1D images are saved as SimpleITK images with the size of only one of the dimension different than one.

Most of the functions have a bool argument ``displayInfo`` (default ``displayInfo=False``) which can be used for displaying a summary of the function results.

Documentation and tutorial
----------------------------
The documentation of the functions implemented in FRED tools is available at `fredtools.ifj.edu.pl <http://www.fredtools.ifj.edu.pl>`_ or on the `GitHub repository <https://github.com/jasqs/FREDtools>`_.

A simple `tutorial <https://github.com/jasqs/FREDtools/blob/main/examples/FREDtools%20Tutorial.ipynb>`_ has been prepared to help with starting using FRED tools.

Installation
----------------------------
The stable version of FRED tools is available via pip.

.. code:: bash

   $ pip install fredtools

To update existing installation:

.. code:: bash

   $ pip install --upgrade fredtools

The development version is available on GitHub.

.. code:: bash

   $ git clone jasqs/FREDtools

.. .. caution:: There is an installation issue for python 3.10.4 (natively installed in Ubuntu 22.04 LTS). FREDtools requires ITK in 5.2.1 version which cannot be built for python 3.10. But the prerelease ITK 5.3rc4 can be built. It is recommended to install this prerelease prior to FREDtools installation by:

..    .. code:: bash

..       $ pip install scikit-build

..       $ pip install --pre itk

..    FREDtools should be installed normally when the prerelease ITK becomes normal release.


Development
----------------------------
Everyone is invited to support the development of the FRED tools and to add functionalities needed for a specific application.
All the new functions or classes should be documented according to numpydoc style. Check `numpydoc <https://numpydoc.readthedocs.io>`_ and `napoleon <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`_ websites for the style guide and examples.

The FRED tools documentation is written in reStructuredText format and build with sphinx in version |sphinxVersion| using sphinx_rtd_theme template in version |RTDthemeVersion|.

.. toctree::
   :maxdepth: 1

   Introduction/Collaboration
   Introduction/Citation

.. toctree::
   :maxdepth: 1
   :caption: Documentation:

   Documentation/ReadWrite
   Documentation/ImageAnalyse
   Documentation/spotAnalyse
   Documentation/dvhAnalyse
   Documentation/gammaIndexAnalyse
   Documentation/braggPeakAnalyse
   Documentation/InmAnalyse
   Documentation/GettingSubimage
   Documentation/ImageManipulate
   Documentation/displayImage
   Documentation/simTools
   Documentation/optimisationTools
   Documentation/misc

Support
----------------------------
If You like the project and want to support our work you can buy as a coffee by clicking the image.

.. image:: buyCaffe.png
   :scale: 80%
   :alt: By a Caffee
   :align: center
   :target: https://buycoffee.to/fredtools
