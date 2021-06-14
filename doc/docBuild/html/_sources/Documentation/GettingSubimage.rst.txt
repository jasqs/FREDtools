Getting Subimage
=================================

A collection of useful functions for getting subimages of lower dimension, e.g. a slice or a profile from a 3D image.
The image must be an instance of a SimpleITK image and the same image, with the same dimension is returned. The subimages are
calculated user defined interpolation. The interpolation of 'nearest', 'linear' and 'spline' with order from 0 to 5 are available.

.. autofunction:: fredtools.getSlice

.. autofunction:: fredtools.getProfile

.. autofunction:: fredtools.getPoint

.. autofunction:: fredtools.getInteg
