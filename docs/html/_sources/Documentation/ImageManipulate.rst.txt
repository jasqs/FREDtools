Image Manipulate
=================================

A collection of useful functions to manipulate and change images. The image must be an instance of a SimpleITK image and the functions are mostly wrappers for SimpleITK image filters. Check `SimpleITK filters <https://simpleitk.readthedocs.io/en/master/filters.html>`_ for available methods of filtering, registration, etc.

.. autofunction:: fredtools.mapStructToImg

.. autofunction:: fredtools.cropImgToMask

.. autofunction:: fredtools.setValueMask

.. autofunction:: fredtools.resampleImg

.. autofunction:: fredtools.sumImg

.. autofunction:: fredtools.createCylindricalMask

.. autofunction:: fredtools.getImgBEV
