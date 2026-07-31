Miscellaneous
=================================

A collection of useful miscellaneous functions, implemented in the ``fredtools.Miscellaneous`` subpackage.

General purpose
------------------------------------------------

.. autofunction:: fredtools.mergePDF

.. autofunction:: fredtools.getLineFromFile

.. autofunction:: fredtools.getHistogram

.. autofunction:: fredtools.sigma2fwhm

.. autofunction:: fredtools.fwhm2sigma

.. autofunction:: fredtools.wrapAngle

.. autofunction:: fredtools.getCPUNo

Landau, Vavilov and Gauss distributions
------------------------------------------------

Functions for calculating and fitting the Landau and Vavilov probability density functions, as well as their convolutions with a Gaussian, useful for instance for fitting energy deposition spectra.

.. autofunction:: fredtools.pdfLandau

.. autofunction:: fredtools.pdfLandauGauss

.. autofunction:: fredtools.pdfVavilov

.. autofunction:: fredtools.fitLandau

.. autofunction:: fredtools.fitLandauGauss

.. autofunction:: fredtools.fitVavilov

Logging
------------------------------------------------

FREDtools uses the standard Python logging framework. All the output produced by the functions, including the summaries requested with the ``displayInfo`` argument, is emitted through loggers. The logging verbosity and format can be controlled with the functions below.

.. autofunction:: fredtools.configureLogging

.. autofunction:: fredtools.getLogger
