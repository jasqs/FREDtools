Gamma Index Analysis
=================================

A collection of functions for the Gamma Index (GI) analysis, implemented in the ``fredtools.GammaIndex`` subpackage.

.. note::
    The GI calculation is performed by an external C++ library compiled as a shared library, which is distributed with FREDtools. Currently, only the Linux version of the shared library is available.

The gamma index engine has been validated against the independent implementations in PyMedPhys and plastimatch. The comparison, together with a description of the algorithms and the reasons why the results of the tools are never exactly the same, is presented in the :ref:`GammaIndexValidation`.

.. toctree::
   :maxdepth: 1

   GammaIndexValidation

.. autofunction:: fredtools.calcGammaIndex

.. autofunction:: fredtools.getGIstat

.. autofunction:: fredtools.getGIcmap
