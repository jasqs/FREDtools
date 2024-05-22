Influence Matrix Analyse
=================================

A collection of useful functions for reading and manipulating with influence matrices produced by a Monte Carlo. The influence matrix is usually a 3D image describing the influence (dose, LET or other) for each pencil beam, therefore it can be treated as a 4D image with geometrical X, Y, Z and pencil beam dimensions. Such matrices can occupy a lot of memory, therefore most of the functions implemented here are equipped with memory occupancy checking.

.. note::
    The binary influence matrix file format had changed from the FRED 3.70.99 version. The function has been aligned with this format but will not work with the previous format. Use FREDtools v. 0.7.6 to read the old binary influence matrix file format or contact the FREDtools developers.

.. autofunction:: fredtools.getInmFREDSumImage

.. autofunction:: fredtools.getInmFREDPoint

.. autofunction:: fredtools.getInmFREDPointSparse

.. autofunction:: fredtools.getInmFREDInfo

.. autofunction:: fredtools.getInmFREDBaseImg



