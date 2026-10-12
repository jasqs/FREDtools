Gamma Index Analysis
=================================

A collection of functions for the Gamma Index (GI) analysis, implemented in the ``fredtools.GammaIndex`` subpackage.

.. note::
    The GI calculation is performed by libFredGI, an external C++ library (version 4.5), which is distributed with FREDtools in compiled form, in ``fredtools/GammaIndex``:

    - ``libFredGI.so``: Linux on x86_64, with glibc 2.28 or newer (e.g. RHEL/Rocky 8, Ubuntu 20.04, Debian 10 and newer),
    - ``libFredGI.dll``: Windows 10 or newer on x86_64,
    - ``libFredGI.dylib``: macOS 11 or newer, on Apple Silicon and Intel (a universal library).

    A 64-bit Python is needed, and the library of the running system is chosen automatically. The results are the same on all three systems (bit-identical in the tests of the library). The library is not covered by the GPL licence of FREDtools; see its notice `LICENSE-libFredGI.md <https://github.com/jasqs/FREDtools/blob/main/fredtools/GammaIndex/LICENSE-libFredGI.md>`_.

    On macOS, a library downloaded with a web browser (e.g. FREDtools downloaded as a ZIP archive) can be blocked by Gatekeeper. The quarantine attribute is then removed with ``xattr -d com.apple.quarantine fredtools/GammaIndex/libFredGI.dylib``; a clone made with git is not affected.

The gamma index engine has been validated against the independent implementations in PyMedPhys and plastimatch. The comparison, together with a description of the algorithms and the reasons why the results of the tools are never exactly the same, is presented in the :ref:`GammaIndexValidation`.

.. toctree::
   :maxdepth: 1

   GammaIndexValidation

.. autofunction:: fredtools.calcGammaIndex

.. autofunction:: fredtools.getGIstat

.. autofunction:: fredtools.getGIcmap
