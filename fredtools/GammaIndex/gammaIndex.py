from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def _libFredGIFileName() -> str:
    """Get the file name of the gamma index library (libFredGI) for the running platform.

    libFredGI is shipped as libFredGI.so (Linux, x86_64), libFredGI.dll (Windows, x86_64)
    and libFredGI.dylib (macOS, universal library for Apple Silicon and Intel).

    Raises
    ------
    OSError
        If libFredGI is not available for the platform or Python is not 64-bit.
    """
    import sys
    import platform

    machine = platform.machine().lower()
    if sys.maxsize > 2**32:  # 64-bit Python (the libraries are 64-bit)
        if sys.platform.startswith("linux") and machine in ("x86_64", "amd64"):
            return "libFredGI.so"
        if sys.platform == "win32" and machine in ("amd64", "x86_64"):
            return "libFredGI.dll"
        if sys.platform == "darwin" and machine in ("arm64", "x86_64"):
            return "libFredGI.dylib"
    error = OSError(f"The gamma index library is available for 64-bit Python on Linux (x86_64), Windows (x86_64) and macOS, but not on this platform ({sys.platform}, {platform.machine()}, {64 if sys.maxsize > 2**32 else 32}-bit Python).")
    _logger.error(error)
    raise error


def _loadLibFredGI() -> Any:
    """Load the gamma index library of the running platform and set the prototypes of its functions."""
    import os
    import sys
    import ctypes
    from numpy.ctypeslib import ndpointer

    libPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), _libFredGIFileName())
    try:
        libFredGI = ctypes.CDLL(libPath)
    except OSError as loadError:
        error = OSError(f"The gamma index library {libPath} could not be loaded on this platform ({sys.platform}): {loadError}")
        _logger.error(error)
        raise error from loadError

    # argument and return types of the library functions (all return an int: 0 or an error code, or true/false)
    mapArgtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # number of voxels
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # spacing
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # lower corner of the image
                   ndpointer(ctypes.c_float, flags="F_CONTIGUOUS"),  # dose values
                   ]
    prototypes = {"fredGI_init": [],
                  "fredGI_reset": [],
                  "fredGI_version": [ctypes.c_char_p],
                  "fredGI_setInterpolation": [ctypes.c_bool],
                  "fredGI_setDTA": [ctypes.c_float],
                  "fredGI_setDD": [ctypes.c_float],
                  "fredGI_setDDCriterium": [ctypes.c_int],
                  "fredGI_setDCO": [ctypes.c_float],
                  "fredGI_setGlobalNormalization": [ctypes.c_float],
                  "fredGI_setStepSize": [ctypes.c_float],
                  "fredGI_setVerbose": [ctypes.c_int],
                  "fredGI_setComputationMode": [ctypes.c_int],
                  "fredGI_setNumThreads": [ctypes.c_int],
                  "fredGI_setRef": mapArgtypes,
                  "fredGI_setEval": mapArgtypes,
                  "fredGI_startComputation": [],
                  "fredGI_isComputing": [],
                  "fredGI_waitForCompletion": [],
                  "fredGI_hasFinished": [],
                  "fredGI_abortComputation": [],
                  "fredGI_getGammaIndexPassRate": [ctypes.POINTER(ctypes.c_float)],
                  "fredGI_getGammaIndex3DMap": [ndpointer(ctypes.c_float, flags="F_CONTIGUOUS")],
                  }
    for functionName, argtypes in prototypes.items():
        function = getattr(libFredGI, functionName)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
    return libFredGI


def _unloadLibFredGI(libFredGI: Any) -> None:
    """Unload the gamma index library (its computation thread must have been released before)."""
    import sys
    import ctypes

    if sys.platform == "win32":
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]  # a 64-bit handle
        kernel32.FreeLibrary.restype = wintypes.BOOL
        kernel32.FreeLibrary(libFredGI._handle)
    else:
        libc = ctypes.CDLL(None)
        libc.dlclose.argtypes = [ctypes.c_void_p]
        libc.dlclose.restype = ctypes.c_int
        libc.dlclose(libFredGI._handle)


def calcGammaIndex(imgRef: SITKImage, imgEval: SITKImage, DD: Annotated[Numeric, Field(strict=True, ge=0, le=100)], DTA: Numeric, DCO: Annotated[Numeric, Field(strict=True, ge=0, le=1)], DDType: Literal["local", "global"] = "local", globalNorm: Numeric | None = None, stepSize: Numeric = 10, fractionalStepSize: bool = True, mode: Literal["gamma", "pass-rate"] = "gamma", displayInfo: bool = False) -> SITKImage:
    """Calculate gamma index map.

    The function calculates the gamma index map using the `imgRef` and `imgEval`,
    defined as SimpleITK 2D or 3D image objects of the same dimension, as the reference
    and evaluation images, respectively.
    The gamma index test is performed with a defined dose difference (DD) given in [%],
    distance to agreement (DTA) given in the same length unit as the reference image
    (in [mm] by default) and is calculated for the reference voxels with a dose strictly
    greater than the dose cut-off, which is a fraction of the global normalisation dose
    (the maximum dose in the reference image by default) given by the DCO parameter.
    The voxels with a dose equal to the cut-off are not analysed (-1).
    The gamma index can be calculated for `local` or `global` dose difference in two modes:

        -  *gamma*: each voxel represents the gamma index value and the voxels excluded from the GI analysis have values -1.
        -  *pass-rate*: each voxel represents passing (1) or failing (0) of the gamma index test and the voxels excluded from the GI analysis have values -1.

    The gamma index calculation is performed by an external C++ library (libFredGI), compiled for
    Linux (x86_64), Windows (x86_64) and macOS (Apple Silicon and Intel), and shipped with FREDtools.
    The gamma index engine was developed by Angelo Schiavi and validated against the PyMedPhys [PyMedPhys]_
    python library and the plastimatch [Plastimatch]_ gamma index tool (see the :ref:`GammaIndexValidation`).

    Parameters
    ----------
    imgRef : SimpleITK Image
        An object of a SimpleITK 2D or 3D image describing the reference. The image must have
        a scalar pixel type. An axis of size 1 is a single slice whose dose is constant
        across its voxel thickness.
    imgEval : SimpleITK Image
        An object of a SimpleITK 2D or 3D image describing the evaluation, of the same
        dimension and with the same requirements as for the `imgRef`. It can have a different
        size, spacing and origin than the `imgRef` but must have the same direction matrix.
    DD : float
        Dose difference in [%].
    DTA : float
        Distance-to-agreement, usually in [mm].
    DCO : float
        Dose cut-off given as the fraction of the global normalisation dose (the maximum
        reference value by default, see `globalNorm`). The gamma index is calculated only for
        the reference voxels with a dose strictly greater than the cut-off. The voxels with
        a dose equal to or lower than the cut-off are not analysed (-1).
    DDType : {'local', 'global'}, optional
        Method of calculating the absolute dose difference criterion. (def. 'local'):

            - 'local' : the absolute dose difference calculated as DD percent of the local reference value.
            - 'global' : the absolute dose difference calculated as DD percent of the maximum reference value.

    globalNorm : float, optional
        Global normalisation dose in the units of the input images, used to calculate the dose
        cutoff (`DCO`) and the 'global' dose difference criterion (`DD`). If not given or None
        then the maximum value of the reference image is used. (def. None)
    stepSize : float, optional
        Step size to search for the minimum gamma index value. It can be given
        as an absolute step in the reference length unit (for instance in [mm])
        if `fractionalStepSize` is False, or as the number of subdivisions of
        the distance-to-agreement if `fractionalStepSize` is True, so that
        the search step is calculated as DTA/stepSize. (def. 10)
    fractionalStepSize : bool, optional
        Determine if the `stepSize` should be treated as the absolute step value (False)
        or as the number of subdivisions of the distance-to-agreement (True),
        meaning that the search step is DTA/stepSize. (def. True)
    mode : {'gamma', 'pass-rate'}, optional
        Mode of calculation. (def. 'gamma'):

            -  'gamma': each voxel represents the gamma index value and the voxels excluded from the GI analysis have values -1.
            -  'pass-rate': each voxel represents passing (1) or failing (0) of the gamma index test and the voxels excluded from the GI analysis have values -1.

    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image describing the gamma index. It is of float or integer
        type when the calculation mode is 'gamma' or 'pass-rate', respectively.

    Raises
    ------
    TypeError
        If `imgRef` or `imgEval` is not an instance of a SimpleITK image object
        describing a scalar 2D or 3D image (e.g. a vector or a 4D image).
    ValueError
        If `imgRef` and `imgEval` have different dimensions,
        the value of `DD`, `DTA`, `DCO`, `globalNorm` or `stepSize` is out of range,
        `DDType` or `mode` cannot be recognised, the direction matrices of `imgRef` and
        `imgEval` do not match, the images do not overlap (detached frames of reference,
        see :func:`fredtools.compareImgFoROverlap`), any of the images contains NaN or Inf values, the maximum
        of the reference image is not positive, or no voxel of the reference image is above
        the dose cutoff.
    OSError
        If the gamma index library is not available for the platform (see above) or
        cannot be loaded.
    RuntimeError
        Run time error is raised when the execution of the gamma calculation failed,
        did not finish, or the library returned an invalid result (no voxel analysed).
        The following error codes can be raised:


        - -1 : generic error
        - -20 : value range error
        - -21 : value is nan error
        - -22 : value is inf error
        - -29 : null pointer error
        - -50 : ill-defined vector
        - -51 : ill-defined dimensions
        - -52 : ill-defined spacing
        - -53 : ill-defined offset
        - -100 : computation ongoing
        - -101 : setup not complete
        - -102 : criteria not defined
        - -103 : ref map not defined
        - -104 : eval map not defined
        - -105 : computation not done

    See Also
    --------
    getGIstat : calculate the gamma index statistics including the gamma index pass rate.

    Notes
    -----
    The gamma index pass rates calculated with this function agree within 1 percentage point with
    PyMedPhys and plastimatch. Identical gamma index maps are not expected, because the tools
    discretise the search space and interpolate the evaluation dose differently.
    Refer to the :ref:`GammaIndexValidation` to read more about the validation results.

    The images are required to be 2D or 3D, of the same dimension, and to have the same direction
    matrix. 2D images are calculated as 3D images with a single slice, so the gamma index search
    is in the image plane, and the gamma index map is returned as a 2D image. The calculation is
    performed in the common index frame of the images, which preserves the distances for any
    common direction matrix, and the resulting gamma index map is returned in the frame of
    reference of the `imgRef`. The dose cutoff in dose units is `DCO` times the global
    normalisation dose, computed by the gamma index library in single precision (float32),
    and the 'global' dose difference criterion is `DD` percent of the global
    normalisation dose, which is the maximum of the reference image unless `globalNorm` is given.

    The calculation runs in a background thread of the gamma index library and can be interrupted
    with Ctrl+C (or by interrupting the Jupyter kernel), which stops the calculation and raises
    `KeyboardInterrupt`.

    Examples
    --------
    See example jupyter notebook at [GITutorial]_

    References
    ----------
    .. [PyMedPhys] https://docs.pymedphys.com/
    .. [Plastimatch] https://plastimatch.org/
    .. [GITutorial] `Jupyter notebook of Gamma Index Analysis Tutorial <https://github.com/jasqs/FREDtools/blob/main/examples/Gamma%20Index%20analysis%20Tutorial.ipynb>`_
    """
    import time
    import fredtools as ft
    import numpy as np
    import ctypes
    import SimpleITK as sitk

    # validate imgRef and imgEval (scalar 2D or 3D images of the same dimension)
    for img, imgName in ((imgRef, "reference"), (imgEval, "evaluation")):
        ft._imgTypeChecker.isSITK(img, raiseError=True)
        if ft._imgTypeChecker.isSITK_vector(img, raiseError=False):
            error = TypeError(f"The {imgName} image must be a scalar SimpleITK image but a vector image of type '{img.GetPixelIDTypeAsString()}' was given.")
            _logger.error(error)
            raise error
        if img.GetDimension() not in [2, 3]:
            error = TypeError(f"The {imgName} image must be an instance of a SimpleITK image object describing a 2D or 3D image but a {img.GetDimension()}D image was given.")
            _logger.error(error)
            raise error
    if imgRef.GetDimension() != imgEval.GetDimension():
        error = ValueError(f"The reference and evaluation images must have the same dimension, but the reference image is {imgRef.GetDimension()}D and the evaluation image is {imgEval.GetDimension()}D.")
        _logger.error(error)
        raise error

    # calculate 2D images as 3D images with a single slice (the gamma index search is then in the image plane)
    imgRefInput = imgRef
    if imgRef.GetDimension() == 2:
        imgRef = ft.expandDimsImg(imgRef, origin=0, spacing=1)
        imgEval = ft.expandDimsImg(imgEval, origin=0, spacing=1)

    # validate DTA, DD, DDType, DCO and globalNorm
    if not isinstance(DTA, Numeric) or DTA <= 0:
        error = ValueError(f"The value of DTA {DTA} is not correct. It must be a positive scalar.")
        _logger.error(error)
        raise error
    if not isinstance(DD, Numeric) or DD <= 0 or DD >= 100:
        error = ValueError(f"The value of DD {DD} is not correct. It must be a positive scalar between 0 and 100.")
        _logger.error(error)
        raise error
    if not isinstance(DDType, str) or DDType.lower() not in ["local", "global", "l", "g"]:
        error = ValueError(f"Dose difference type must be a string and only 'local' or 'global' are supported.")
        _logger.error(error)
        raise error
    if not isinstance(DCO, Numeric) or DCO <= 0 or DCO >= 1:
        error = ValueError(f"The value of DCO {DCO} is not correct. It must be a positive scalar between 0 and 1.")
        _logger.error(error)
        raise error
    if not ((isinstance(globalNorm, Numeric) and globalNorm > 0) or globalNorm is None):
        error = ValueError(f"The value of globalNorm {globalNorm} is not correct. It must be a positive scalar or None.")
        _logger.error(error)
        raise error

    # validate stepSize
    if not isinstance(stepSize, Numeric) or stepSize <= 0:
        error = ValueError(f"The value {stepSize} is not correct. It must be a positive scalar.")
        _logger.error(error)
        raise error

    # validate calculation mode
    if not isinstance(mode, str) or mode.lower() not in ["gamma", "pass-rate", "g", "pr", "p"]:
        error = ValueError(f"Calculation mode must be a string and only 'gamma' or 'pass-rate' are supported.")
        _logger.error(error)
        raise error

    # validate that the direction matrices of the images match (the calculation is done in the common index frame of the images)
    if not np.allclose(imgRef.GetDirection(), imgEval.GetDirection(), atol=1E-6):
        error = ValueError(f"The direction matrices of the reference image {imgRef.GetDirection()} and the evaluation image {imgEval.GetDirection()} do not match. Resample one of the images to the frame of reference of the other one.")
        _logger.error(error)
        raise error

    # validate that the images overlap (the gamma index of detached images is meaningless and its search can take very long)
    if not ft.compareImgFoROverlap(imgRef, imgEval):
        error = ValueError("The reference and evaluation images do not overlap (their frames of reference are detached). Check the origins of the images.")
        _logger.error(error)
        raise error

    # validate the image values (the library returns an undefined result for non-finite values or when no voxel is above the dose cut-off)
    arrRef = sitk.GetArrayFromImage(imgRef).astype(np.float32)
    arrEval = sitk.GetArrayFromImage(imgEval).astype(np.float32)
    for arr, imgName in ((arrRef, "reference"), (arrEval, "evaluation")):
        if not np.isfinite(arr).all():
            error = ValueError(f"The {imgName} image contains NaN or Inf values, which are not supported by the gamma index calculation.")
            _logger.error(error)
            raise error
    refMax = float(arrRef.max())
    if refMax <= 0:
        error = ValueError(f"The reference image must contain positive values but its maximum is {refMax:g}.")
        _logger.error(error)
        raise error
    # the dose cut-off as computed by the gamma index library in single precision (float32): the DCO is passed in [%]
    # as float32, divided by 100 and multiplied by the global normalisation dose (globalNorm passed as float32, or the
    # maximum of the float32 reference image), and only the voxels with a dose strictly greater than it are analysed
    globalNormValue = np.float32(globalNorm) if globalNorm is not None else np.float32(refMax)
    doseCutoff = np.float32(np.float32(DCO * 100) / np.float32(100)) * globalNormValue
    if refMax <= doseCutoff:
        error = ValueError(f"No voxel of the reference image is above the dose cut-off {doseCutoff:g} (DCO {DCO:g} of the global normalisation {globalNormValue:g}), because the maximum of the reference image is {refMax:g}.")
        _logger.error(error)
        raise error

    # validate CPUNo or get it automatically if requested
    CPUNo = ft.getCPUNo(ft.CPUNO)

    # load the gamma index library of the running platform (libFredGI.so, .dll or .dylib)
    libFredGI = _loadLibFredGI()

    # communicate with the library (the argument types are set by _loadLibFredGI)
    try:
        libFredGI.fredGI_init()
        libFredGI.fredGI_reset()

        # get libFredGI version
        libFredGIVersion = ctypes.create_string_buffer(256)
        libFredGI.fredGI_version(libFredGIVersion)
        libFredGIVersion = libFredGIVersion.value.decode("utf-8").strip()

        # set interpolation dose values using neighbouring voxels
        libFredGI.fredGI_setInterpolation(True)

        # set DTA, DD, DDType, DCO and globalNorm
        libFredGI.fredGI_setDTA(float(DTA))
        libFredGI.fredGI_setDD(float(DD))
        if DDType.lower() in ["local", "l"]:
            DDType = "local"
            libFredGI.fredGI_setDDCriterium(2)  # 1 = GLOBAL, 2 = LOCAL
        elif DDType.lower() in ["global", "g"]:
            DDType = "global"
            libFredGI.fredGI_setDDCriterium(1)  # 1 = GLOBAL, 2 = LOCAL
        libFredGI.fredGI_setDCO(DCO * 100)
        if globalNorm:
            libFredGI.fredGI_setGlobalNormalization(float(globalNorm))

        # set stepSize
        if fractionalStepSize:
            stepSize = DTA / stepSize
        libFredGI.fredGI_setStepSize(float(stepSize))

        # set verbosity
        if displayInfo:
            libFredGI.fredGI_setVerbose(5)
        else:
            libFredGI.fredGI_setVerbose(0)

        # set calculation mode
        if mode.lower() in ["gamma", "g"]:
            mode = "gamma"
            libFredGI.fredGI_setComputationMode(1)  # 1 = gamma mode ; 2 = pass-rate mode
        elif mode.lower() in ["pass-rate", "pr", "p"]:
            mode = "pass-rate"
            libFredGI.fredGI_setComputationMode(2)  # 1 = gamma mode ; 2 = pass-rate mode

        # set CPUNo
        libFredGI.fredGI_setNumThreads(CPUNo)

        # set imgRef
        # the lower corner of the image is given in the index frame of the image (direction^T * origin - spacing/2), which is common to both images
        nn = np.array(imgRef.GetSize()).astype(np.int32)
        hs = np.array(imgRef.GetSpacing()).astype(np.float32)
        x0 = (ft.ImgAnalyse.imgAnalyse._getDirectionArray(imgRef).T @ np.array(imgRef.GetOrigin()) - np.array(imgRef.GetSpacing()) / 2).astype(np.float32)
        arr = np.moveaxis(arrRef, range(arrRef.ndim), range(arrRef.ndim)[::-1])
        libFredGI.fredGI_setRef(nn, hs, x0, arr)

        # set imgEval
        nn = np.array(imgEval.GetSize()).astype(np.int32)
        hs = np.array(imgEval.GetSpacing()).astype(np.float32)
        x0 = (ft.ImgAnalyse.imgAnalyse._getDirectionArray(imgEval).T @ np.array(imgEval.GetOrigin()) - np.array(imgEval.GetSpacing()) / 2).astype(np.float32)
        arr = np.moveaxis(arrEval, range(arrEval.ndim), range(arrEval.ndim)[::-1])
        libFredGI.fredGI_setEval(nn, hs, x0, arr)

        # start the computation in the background thread of the library
        computationStatus = libFredGI.fredGI_startComputation()
        if not computationStatus == 0:
            error = RuntimeError(f"Gamma Index computation failed with the error code {computationStatus}. Refer to www.fredtools.ifj.edu.pl for more details.")
            _logger.error(error)
            raise error

        # wait for the end of the computation, checking for a user interrupt (Ctrl+C) every 10 ms
        try:
            while libFredGI.fredGI_isComputing():
                time.sleep(0.01)
        except KeyboardInterrupt:
            _logger.warning("The gamma index computation was interrupted by the user.")
            raise
        libFredGI.fredGI_waitForCompletion()  # release the finished thread
        if not libFredGI.fredGI_hasFinished():
            error = RuntimeError("The gamma index computation did not finish.")
            _logger.error(error)
            raise error

        # get GI pass rate (negative when no voxel was analysed and the result is undefined)
        GIpassRate = ctypes.c_float()
        libFredGI.fredGI_getGammaIndexPassRate(ctypes.byref(GIpassRate))
        GIpassRate = GIpassRate.value
        if GIpassRate < 0:
            error = RuntimeError(f"The gamma index library returned an invalid pass rate {GIpassRate}, meaning that no voxel was analysed. Check the dose cut-off and the input images.")
            _logger.error(error)
            raise error

        # read results and convert to SimpleITK image
        arrGI = np.zeros(imgRef.GetSize(), dtype=np.float32, order="F")
        libFredGI.fredGI_getGammaIndex3DMap(arrGI)
        arrGI = np.moveaxis(arrGI, range(arrGI.ndim), range(arrGI.ndim)[::-1])
        if imgRefInput.GetDimension() == 2:
            arrGI = arrGI[0]  # the single slice of the expanded 2D images
        imgGI = sitk.GetImageFromArray(arrGI)
        imgGI.CopyInformation(imgRefInput)  # the frame of reference of the input imgRef (2D or 3D)
        if mode == "pass-rate":
            imgGI = sitk.Cast(imgGI, sitk.sitkInt8)

    finally:
        # stop and release the computation thread of the library (if any) before unloading the library
        libFredGI.fredGI_abortComputation()

        # unload the library
        _unloadLibFredGI(libFredGI)
        del libFredGI

    # set additional metadata
    imgGI.SetMetaData("GIVersion", libFredGIVersion)
    imgGI.SetMetaData("DD", str(DD))
    imgGI.SetMetaData("DTA", str(DTA))
    imgGI.SetMetaData("DDType", DDType)
    imgGI.SetMetaData("DCO", str(DCO))
    imgGI.SetMetaData("stepSize", str(stepSize))
    imgGI.SetMetaData("mode", mode)
    imgGI.SetMetaData("GIPR", str(GIpassRate))

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgGI))

    return imgGI


def getGIstat(imgGI: SITKImage, displayInfo: bool = False) -> DottedDict:
    """Get statistics of Gamma Index.

    The function calculates Gamma Index statistics from an image defined
    as a SimpleITK image object. Two modes of the gamma index calculation
    are recognised automatically based on the image type:

        -  *gamma* (float): each voxel represents the gamma index value and the voxels excluded from the GI analysis have values -1 or numpy.nan.
        -  *pass-rate* (integer): each voxel represents passing (1) or failing (0) of the gamma index test and the voxels excluded from the GI analysis have values -1.

    Parameters
    ----------
    imgGI : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    DottedDict
        A dictionary-like object (DottedDict) with the gamma index statistics
        with the keys: 'passRate' (gamma index pass rate in [%]), 'mean', 'std',
        'min' and 'max' of the gamma index values. In the 'pass-rate' mode
        the 'mean', 'std', 'min' and 'max' values are numpy.nan.

    Raises
    ------
    TypeError
        If `imgGI` is not an instance of a SimpleITK image object,
        or the image is neither of integer nor float type.
    ValueError
        If the image is of integer type ('pass-rate' mode) but contains
        unique values other than -1, 0 and 1.

    See Also
    --------
    calcGammaIndex : calculate the Gamma Index map for two images.
    """
    import fredtools as ft
    import numpy as np
    import warnings
    ft._imgTypeChecker.isSITK(imgGI, raiseError=True)

    arrGI = ft.arr(imgGI)
    GIstat = {}
    if np.issubdtype(arrGI.dtype, np.integer):
        if not set(np.unique(arrGI)).issubset(set([-1, 0, 1])):
            error = ValueError(f"The calculation mode was recognised as 'pass-rate' because the input image is of integer type but it should contain only [-1, 0, 1] unique values and the uniques are {np.unique(arrGI)}")
            _logger.error(error)
            raise error
        mode = "pass-rate"
        with np.errstate(divide="ignore", invalid="ignore"):
            GIstat["passRate"] = (arrGI == 1).sum() / (arrGI >= 0).sum() * 100
        GIstat["mean"] = np.nan
        GIstat["std"] = np.nan
        GIstat["min"] = np.nan
        GIstat["max"] = np.nan

    elif np.issubdtype(arrGI.dtype, np.floating):
        mode = "gamma"
        arrGI[arrGI < 0] = np.nan
        # the statistics of an image without any analysed voxel are NaN (no warnings)
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            GIstat["passRate"] = (arrGI <= 1).sum() / (arrGI >= 0).sum() * 100
            GIstat["mean"] = np.nanmean(arrGI)
            GIstat["std"] = np.nanstd(arrGI)
            GIstat["min"] = np.nanmin(arrGI)
            GIstat["max"] = np.nanmax(arrGI)
    else:
        error = TypeError(f"The input image is of type {arrGI.dtype} but it should be either integer or float type.")
        _logger.error(error)
        raise error

    if displayInfo:
        strLog = [f"Gamma Index statistics for the image calculated in '{mode}' mode:",
                  f"GIPR: {GIstat['passRate']:.2f}"]
        if mode == "gamma":
            strLog.append("mean/std: {:.2f} / {:.2f}".format(GIstat["mean"], GIstat["std"]))
            strLog.append("min/max: {:.2f} / {:.2f}".format(GIstat["min"], GIstat["max"]))
        _logger.info("\n\t".join(strLog))

    return DottedDict(**GIstat)


def getGIcmap(maxGI: Numeric, N: NonNegativeInt = 256) -> LinearSegmentedColormap:
    """Get colormap for Gamma Index images.

    The function creates a colormap for Gamma Index (GI) images,
    that can be used by matplotlib.pyplot.imshow function for
    displaying 2D images. The colormap is created from 0 to
    the `maxGI` value, whereas from 0 to 1 (GI test passed) the color
    changes from dark blue to white, and from 1 to `maxGI` it changes
    from light red to red.

    Parameters
    ----------
    maxGI : scalar
        The maximum value of the colormap. Values less than 1
        are clamped to 1 (with a warning).
    N : int, optional
        Number of segments of the colormap, at least 2. (def. 256)

    Returns
    -------
    colormap
        An instance of matplotlib.colors.LinearSegmentedColormap object.

    Raises
    ------
    ValueError
        If `N` is not an integer greater than 1.

    See Also
    --------
    calcGammaIndex : calculate the Gamma Index for two images.

    Examples
    --------
    It is assumed that the img is an image describing a slice
    of Gamma Index (GI) values calculated up to maximum value 3.
    To plot the GI map with the GI colormap:

    >>> plt.imshow(ft.arr(img), cmap=getGIcmap(maxGI=3))
    """
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    if not isinstance(N, (int, np.integer)) or N < 2:
        error = ValueError(f"The number of colormap segments N must be an integer greater than 1 but {N} was given.")
        _logger.error(error)
        raise error

    if maxGI < 1:
        _logger.warning(f"The value of the parameter 'maxGI' cannot be less than 1 and a value {maxGI} was given. It was set to 1.")
        maxGI = 1

    colorLowStart = np.array([1, 0, 128]) / 255
    colorLowEnd = np.array([253, 253, 253]) / 255
    colorHighStart = np.array([254, 193, 192]) / 255
    colorHighEnd = np.array([255, 67, 66]) / 255

    cdict = {
        "red":   ((0.0, 0.0, float(colorLowStart[0])),
                  (1 / maxGI, float(colorLowEnd[0]), float(colorHighStart[0])),
                  (1.0, float(colorHighEnd[0]), 0.0)),
        "green": ((0.0, 0.0, float(colorLowStart[1])),
                  (float(1 / maxGI), float(colorLowEnd[1]), float(colorHighStart[1])),
                  (1.0, float(colorHighEnd[1]), 0.0)),
        "blue":  ((0.0, 1.0, float(colorLowStart[2])),
                  (1 / maxGI, float(colorLowEnd[2]), float(colorHighStart[2])),
                  (1.0, float(colorHighEnd[2]), 0.0)),
    }

    cmapGI = LinearSegmentedColormap(name="GIcmap", segmentdata=cdict, N=N)  # type: ignore

    return cmapGI
