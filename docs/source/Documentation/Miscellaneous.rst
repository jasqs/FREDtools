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

.. autofunction:: fredtools.roundToMultiple

.. autofunction:: fredtools.getCPUNo

DICOM UIDs
------------------------------------------------

Functions for reading the UIDs identifying the dicoms and the references between them, and for checking that the dicoms describing a single patient treatment plan are consistent. When many dicoms are to be matched to each other, it is recommended to read them once with ``getDicomsInfo`` and to answer all the consecutive queries from the returned information, instead of calling the ``checkUID_*`` functions for every pair of dicoms.

.. autofunction:: fredtools.getDicomsInfo

.. autofunction:: fredtools.sortDicomsFromInfo

.. autofunction:: fredtools.matchDicomsByUID

.. autofunction:: fredtools.getSOPInstanceUID

.. autofunction:: fredtools.getFrameOfReferenceUID

.. autofunction:: fredtools.getRNReferencedStructureSetUID

.. autofunction:: fredtools.getRSReferencedImageUIDs

.. autofunction:: fredtools.getRDReferencedPlanUID

.. autofunction:: fredtools.checkUID_RNtoRS

.. autofunction:: fredtools.checkUID_RStoCT

.. autofunction:: fredtools.checkUID_RNtoRD

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
