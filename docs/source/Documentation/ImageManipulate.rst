Image Manipulate
=================================

A collection of useful functions to manipulate and change images. The image must be an instance of a SimpleITK image and the functions are mostly wrappers for SimpleITK image filters. Check `SimpleITK filters <https://simpleitk.readthedocs.io/en/master/filters.html>`_ for available methods of filtering, registration, etc.

.. autofunction:: fredtools.mapStructToImg

.. autofunction:: fredtools.floatingToBinaryMask

.. autofunction:: fredtools.cropImgToMask

.. autofunction:: fredtools.setValueMask

.. autofunction:: fredtools.resampleImg

.. autofunction:: fredtools.sumImg

.. autofunction:: fredtools.imgDivide

.. autofunction:: fredtools.sumVectorImg

.. autofunction:: fredtools.createEllipseMask

.. autofunction:: fredtools.createConeMask

.. autofunction:: fredtools.createCylinderMask

.. autofunction:: fredtools.getImgBEV

.. autofunction:: fredtools.setIdentityDirection

.. autofunction:: fredtools.overwriteCTPhysicalProperties

.. autofunction:: fredtools.addMarginToMask
