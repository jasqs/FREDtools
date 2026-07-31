Proton Optimisation Tools
=================================

A collection of functions useful for proton treatment plan optimisation, implemented in the ``fredtools.ProtonOptimisation`` subpackage. The functions are not exported to the top-level ``fredtools`` namespace and must be called with the full module path, e.g. ``fredtools.ProtonOptimisation.preoptimizer.convertCTtoWER``.

Pre-optimisation
------------------------------------------------

.. autofunction:: fredtools.ProtonOptimisation.preoptimizer.convertCTtoWER

.. autofunction:: fredtools.ProtonOptimisation.preoptimizer.calcWETfromWER

.. autofunction:: fredtools.ProtonOptimisation.preoptimizer.generateIsoLayers

.. autofunction:: fredtools.ProtonOptimisation.preoptimizer.calcContours

.. autofunction:: fredtools.ProtonOptimisation.preoptimizer.convertRayTargetToIsoPlane

Beam position optimisation
------------------------------------------------

.. autofunction:: fredtools.ProtonOptimisation.optimiseBeamPositions.optimiseBeamPositions

.. autofunction:: fredtools.ProtonOptimisation.optimiseBeamPositions.optimiseBeamPositionsRegular

.. autofunction:: fredtools.ProtonOptimisation.optimiseBeamPositions.optimiseBeamPositionsHexagonal

.. autofunction:: fredtools.ProtonOptimisation.optimiseBeamPositions.optimiseBeamPositionsConcentric

.. autofunction:: fredtools.ProtonOptimisation.optimiseBeamPositions.optimiseBeamPositionsDelaunay
