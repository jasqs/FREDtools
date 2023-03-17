Dij influence matrix Analyse
=================================

A collection of useful functions for reading and manipulating with Dij influence matrices produced by a Monte Carlo. The Dij influence matrix is usually a 3D image describing the influence (dose, LET or other) for each pencil beam, therefore it can be treated as a 4D image with geometrical X, Y, Z and pencil beam dimensions. Such matrices can occupy a lot of memory, therefore most of the functions implemented here are equipped with memory occupancy checking.

.. autofunction:: fredtools.getDijFREDVectorImage

.. autofunction:: fredtools.getDijFREDSumImage

.. autofunction:: fredtools.getDijFREDPoint

.. autofunction:: fredtools.getDijFREDInfo


