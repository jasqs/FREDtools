Monte Carlo Simulation Tools
=================================

A collection of useful functions for Monte Carlo simulations, implemented in the ``fredtools.MonteCarlo`` subpackage.

FRED Monte Carlo
------------------------------------------------

.. autofunction:: fredtools.setFieldsFolderStruct

.. autofunction:: fredtools.readFREDStat

.. autofunction:: fredtools.getFREDVersions

.. autofunction:: fredtools.checkFREDVersion

.. autofunction:: fredtools.getFREDVersion

.. autofunction:: fredtools.runFRED

Beam model
------------------------------------------------

.. autofunction:: fredtools.readBeamModel

.. autofunction:: fredtools.writeBeamModel

.. autofunction:: fredtools.interpolateBeamModel

.. autofunction:: fredtools.calcRaysVectors

.. autoclass:: fredtools.beamModel
    :members:

GATE Monte Carlo
------------------------------------------------

.. autofunction:: fredtools.readGATE_HITSActor

.. autofunction:: fredtools.readGATE_PSActor

.. autofunction:: fredtools.readGATEStat
