Image Analysis
=================================

A collection of useful functions for image analysis, implemented in the ``fredtools.ImgAnalyse`` subpackage. The image must be an instance of a SimpleITK image.

Image analysis
------------------------------------------------

.. autofunction:: fredtools.getExtent

.. autofunction:: fredtools.getSize

.. autofunction:: fredtools.getImageCenter

.. autofunction:: fredtools.getMassCenter

.. autofunction:: fredtools.getMaxPosition

.. autofunction:: fredtools.getMinPosition

.. autofunction:: fredtools.getVoxelCentres

.. autofunction:: fredtools.getVoxelEdges

.. autofunction:: fredtools.getVoxelPhysicalPoints

.. autofunction:: fredtools.getExtMpl

.. autofunction:: fredtools.isPointInside

.. autofunction:: fredtools.getStatistics

.. autofunction:: fredtools.getIntegral

.. autofunction:: fredtools.compareImg

.. autofunction:: fredtools.compareImgFoR

.. autofunction:: fredtools.pos

.. autofunction:: fredtools.arr

.. autofunction:: fredtools.vec

Image information
------------------------------------------------

.. autofunction:: fredtools.displayImageInfo

Coordinate transformations
------------------------------------------------

.. autofunction:: fredtools.transformIndexToPhysicalPoint

.. autofunction:: fredtools.transformContinuousIndexToPhysicalPoint

.. autofunction:: fredtools.transformPhysicalPointToIndex

.. autofunction:: fredtools.transformPhysicalPointToContinuousIndex

Image display
------------------------------------------------

Functions and classes for image displaying. The functionalities have been designed to quickly display and analyse 3D images of CT and/or dose (or any other 3D quantity distribution) in jupyter, including the interactive mode.

.. autofunction:: fredtools.showSlice

.. autoclass:: fredtools.showSlices
    :members:

Spot analysis
------------------------------------------------

.. autofunction:: fredtools.fitSpotProfile

.. autofunction:: fredtools.fitSpotImg

.. autofunction:: fredtools.findSpots

.. autofunction:: fredtools.fitSigmaSquaredModel

DVH analysis
------------------------------------------------

Functions for Dose-Volume Histogram (DVH) analysis. The DVH data is returned as an instance of the ``DVH`` class described below.

.. autofunction:: fredtools.getDVHMask

.. autofunction:: fredtools.getDVHStruct

.. autoclass:: fredtools.ImgAnalyse.dvhAnalyse.DVH
    :members:
