def calcGammaIndex(imgRef, imgEval, DD, DTA, DCO, DDType="local", globalNorm=None, stepSize=10, fractionalStepSize=True, mode="gamma", CPUNo="auto", displayInfo=False):
    """Calculate gamma index map.

    The function calculates the gamma index map using the `imgRef` and `imgEval`,
    defined SimpleITK image objects, as the reference and evaluation images, respectively.
    The gamma index test is performed with defined dose distance (DD) given in [%],
    distance to agreement (DTA) given in the same length unit as the reference image
    (in [mm] by default) and is calculated for the dose values greater or equal than
    a fraction of the maximum dose in the reference image, given by the DCO parameter.
    The gamma index can be calculated for `local` or `global` dose difference in tho modes:

        -  *gamma*: each voxel represents the gamma index value and the voxels excluded from the GI analysis have values -1.
        -  *pass-rate*: each voxel represents passing (1) or falling (0) of the gamma index test and the voxels excluded from the GI analysis have values -1.

    The gamma index calculation is performed by an external C++ library complied as a linux shared library.
    The gamma index engine was developed by Angelo Schiavi and validated against PyMedPhys [1]_ python library.

    Parameters
    ----------
    imgRef : SimpleITK Image
        Object of a SimpleITK 2D or 3D image describing the reference.
    imgEval : SimpleITK Image
        Object of a SimpleITK 2D or 3D image describing the evaluation.
    DD : float
        Dose distance in [%].
    DTA : float
        Distance-to-agreement, usually in [mm].
    DCO : float
        Lower dose cutoff below which gamma will not be calculated given as the fraction of the maximum reference value.
    DDType : {'local', 'global'}, optional
        Method of calculating the absolute dose difference criterion. (def. 'local'):

            - 'local' : the absolute dose difference calculated as DD percent of the local reference value.
            - 'global' : the absolute dose difference calculated as DD percent of the maximum reference value.

    globalNorm : float, optional
        Global normalisation of the input images. If not given or None then the maximum value of
        the reference image is used. (def. None)
    stepSize : float, optional
        Step size to search for minimum gamma index value. Can be given
        as an absolute value in the reference length unit (for instance in [mm])
        or as the fraction of the distance-to-agreement if fractionalStepSize=True. (def. 10)
    fractionalStepSize : bool, optional
        Determine if the `stepSize` should be treated as the absolute value (false)
        or the fraction of the distance-to-agreement (true). (def. True).
    mode : {'gamma', 'pass=-rate'}, optional
        Mode of calculation. (def. 'gamma'):

            -  'gamma': each voxel represents the gamma index value and the voxels excluded from the GI analysis have values -1.
            -  'pass-rate': each voxel represents passing (1) or falling (0) of the gamma index test and the voxels excluded from the GI analysis have values -1.

    CPUNo : {'auto', 'none'}, scalar or None, optional
        Define if the multiprocessing should be used and how many cores should
        be exploited. Can be None, then no multiprocessing will be used,
        a string 'auto', then the number of cores will be determined by os.cpu_count(),
        or a scalar defining the number of CPU cored to be used. (def. 'auto')
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image describing the gamma index. It is of float or integer
        type when the calculation mode is 'gamma' or 'pass-rate', respectively.

    Raises
    ------
    RuntimeError
        Run time error is raised when the execution of the gamma calculation failed.
        The following error codes can be raised:


        - -1 : generic error
        - -20 : value range error
        - -21 : value is nan error
        - -22 : value is inf error
        - -29 : null pointer error
        - -50 : illdefined vector
        - -51 : illdefined dimensions
        - -52 : illdefined spacing
        - -53 : illdefined offset
        - -100 : computation ongoing
        - -101 : setup not complete
        - -102 : criteria not defined
        - -103 : ref map not defined
        - -104 : eval map not defined
        - -105 : computation not done

    See Also
    --------
        getGIstat: calculate the gamma index statistics including the gamma index pass rate.

    Examples
    --------
    See example jupyter notebook at [2]_

    References
    ----------
    .. [1] https://docs.pymedphys.com/
    .. [2] `Jupyter notebook of Gamma Index Analysis Tutorial <https://github.com/jasqs/FREDtools/blob/main/examples/Gamma%20Index%20analysis%20Tutorial.ipynb>`_
    """
    import sys, os
    import fredtools as ft
    import numpy as np
    import ctypes
    from numpy.ctypeslib import ndpointer
    import SimpleITK as sitk

    # validate imgRef
    ft._isSITK(imgRef, raiseError=True)
    if not ft._isSITK_slice(imgRef, raiseError=False) and not ft._isSITK_volume(imgRef, raiseError=False):
        raise TypeError(f"The reference image must be an instance of a SimpleITK image object describing a 3D volume or 2D slice.")

    # validate imgEval
    ft._isSITK(imgEval, raiseError=True)
    if not ft._isSITK_slice(imgEval, raiseError=False) and not ft._isSITK_volume(imgEval, raiseError=False):
        raise TypeError(f"The reference image must be an instance of a SimpleITK image object describing a 3D volume or 2D slice.")

    # validate DTA, DD, DDType, DCO and globalNorm
    if not np.isscalar(DTA) or DTA <= 0:
        raise ValueError(f"The value od DTA {DTA} is not correct. It must be a positive scalar.")
    if not np.isscalar(DD) or DD <= 0 or DD >= 100:
        raise ValueError(f"The value of DD {DD} is not correct. It must be a positive scalar between 0 and 100.")
    if not isinstance(DDType, str) or DDType.lower() not in ["local", "global", "l", "g"]:
        raise ValueError(f"Dose distance type must be a string and only 'local' or 'global' are supported.")
    if not np.isscalar(DCO) or DCO <= 0 or DCO >= 1:
        raise ValueError(f"The value of DCO {DCO} is not correct. It must be a positive scalar between 0 and 1.")
    if not ((np.isscalar(globalNorm) and globalNorm > 0) or globalNorm is None):
        raise ValueError(f"The value of globalNorm {globalNorm} is not correct. It must be a positive scalar or None.")

    # validate stepSize
    if not np.isscalar(stepSize) or stepSize <= 0:
        raise ValueError(f"The value {stepSize} is not correct. It must be a positive scalar.")

    # validate calculation mode
    if not isinstance(mode, str) or mode.lower() not in ["gamma", "pass-rate", "g", "pr", "p"]:
        raise ValueError(f"Calculation mode must be a string and only 'gamma' or 'pass-rate' are supported.")

    # validate CPUNo or get it automatically if requested
    CPUNo = ft._getCPUNo(CPUNo)

    # load, init and reset shared library
    if sys.platform == "linux" or sys.platform == "linux2":
        dlclose = ctypes.cdll.LoadLibrary("").dlclose
        dlclose.argtypes = [ctypes.c_void_p]
        libFredGI = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "./libFredGI.so"))
        libFredGI_h = libFredGI._handle
    elif sys.platform == "win32":
        raise OSError("Windows version not implemented yet. Only the linux shared library is working now.")
        libFredGI = ctypes.WinDLL("./libFredGI.dll")
        libFredGI_h = libFredGI._handle
    libFredGI.fredGI_init()
    libFredGI.fredGI_reset()

    # communicate with the library
    try:
        # get libFredGI version
        libFredGIVersion = b"\0" * 256
        libFredGI.fredGI_version(ctypes.c_char_p(libFredGIVersion))
        libFredGIVersion = libFredGIVersion.decode("utf-8")

        # set interpolation dose values using neighboring voxels
        libFredGI.fredGI_setInterpolation(ctypes.c_int(1))

        # set DTA, DD, DDType, DCO and globalNorm
        libFredGI.fredGI_setDTA(ctypes.c_float(DTA))
        libFredGI.fredGI_setDD(ctypes.c_float(DD))
        if DDType.lower() in ["local", "l"]:
            DDType = "local"
            libFredGI.fredGI_setDDCriterium(ctypes.c_int(2))  # 1 = GLOBAL, 2 = LOCAL
        elif DDType.lower() in ["global", "g"]:
            DDType = "global"
            libFredGI.fredGI_setDDCriterium(ctypes.c_int(1))  # 1 = GLOBAL, 2 = LOCAL
        libFredGI.fredGI_setDCO(ctypes.c_float(DCO * 100))
        if globalNorm:
            libFredGI.fredGI_setGlobalNormalization(ctypes.c_float(globalNorm))

        # set stepSize
        if fractionalStepSize:
            stepSize = DTA / stepSize
        libFredGI.fredGI_setStepSize(ctypes.c_float(stepSize))

        # set verbosity
        if displayInfo:
            libFredGI.fredGI_setVerbose(ctypes.c_int(5))
        else:
            libFredGI.fredGI_setVerbose(ctypes.c_int(0))

        # set calculation mode
        if mode.lower() in ["gamma", "g"]:
            mode = "gamma"
            libFredGI.fredGI_setComputationMode(ctypes.c_int(1))  # 1 = gamma mode ; 2 = pass-rate mode
        elif mode.lower() in ["pass-rate", "pr", "p"]:
            mode = "pass-rate"
            libFredGI.fredGI_setComputationMode(ctypes.c_int(2))  # 1 = gamma mode ; 2 = pass-rate mode

        # set CPUNo
        libFredGI.fredGI_setNumThreads(ctypes.c_int(CPUNo))

        # set imgRef
        libFredGI.fredGI_setRef.argtypes = [
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="F_CONTIGUOUS"),
        ]
        nn = np.array(imgRef.GetSize()).astype(np.int32)
        hs = np.array(imgRef.GetSpacing()).astype(np.float32)
        x0 = np.array(ft.getExtent(imgRef))[:, 0].astype(np.float32)
        arr = sitk.GetArrayFromImage(imgRef).astype(np.float32)
        arr = np.moveaxis(arr, range(arr.ndim), range(arr.ndim)[::-1])
        libFredGI.fredGI_setRef(nn, hs, x0, arr)

        # set imgEval
        libFredGI.fredGI_setEval.argtypes = [
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="F_CONTIGUOUS"),
        ]
        nn = np.array(imgEval.GetSize()).astype(np.int32)
        hs = np.array(imgEval.GetSpacing()).astype(np.float32)
        x0 = np.array(ft.getExtent(imgEval))[:, 0].astype(np.float32)
        arr = sitk.GetArrayFromImage(imgEval).astype(np.float32)
        arr = np.moveaxis(arr, range(arr.ndim), range(arr.ndim)[::-1])
        libFredGI.fredGI_setEval(nn, hs, x0, arr)

        # start computation
        computationStatus = libFredGI.fredGI_startComputation()
        if not computationStatus == 0:
            raise RuntimeError(f"Gamma Index computation failed with the error code {computationStatus}. Refer to www.fredtools.ifj.edu.pl for more details.")

        # read results and convert to SimpleITK image
        arrGI = np.zeros(imgRef.GetSize(), dtype=np.float32, order="F")
        libFredGI.fredGI_getGammaIndex3DMap.argtypes = [ndpointer(ctypes.c_float, flags="F_CONTIGUOUS")]
        libFredGI.fredGI_getGammaIndex3DMap(arrGI)
        arrGI = np.moveaxis(arrGI, range(arrGI.ndim), range(arrGI.ndim)[::-1])
        imgGI = sitk.GetImageFromArray(arrGI)
        imgGI.CopyInformation(imgRef)
        if mode == "pass-rate":
            imgGI = sitk.Cast(imgGI, sitk.sitkInt8)

        # get GI pass rate
        GIpassRate = ctypes.c_float()
        libFredGI.fredGI_getGammaIndexPassRate(ctypes.byref(GIpassRate))
        GIpassRate = GIpassRate.value
        # set additional metadata
        imgGI.SetMetaData("GIVersion", libFredGIVersion)
        imgGI.SetMetaData("DD", str(DD))
        imgGI.SetMetaData("DTA", str(DTA))
        imgGI.SetMetaData("DDType", DDType)
        imgGI.SetMetaData("DCO", str(DCO))
        imgGI.SetMetaData("stepSize", str(stepSize))
        imgGI.SetMetaData("mode", mode)
        imgGI.SetMetaData("GIPR", str(GIpassRate))

    finally:
        # unload the library
        if sys.platform == "linux" or sys.platform == "linux2":
            del libFredGI
            dlclose(libFredGI_h)
        elif sys.platform == "win32":
            raise OSError("Windows version not implemented yet. Only the linux shared library is working now.")
            del libFredGI
            ctypes.windll.kernel32.FreeLibrary(libFredGI_h)

    if displayInfo:
        print()
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(imgGI)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgGI


def getGIstat(imgGI, displayInfo=False):
    """Get statistics of Gamma Index.

    The function calculates Gamma Index statistics from an image defined
    as a SimpleITK image object. Tho modes of the gamma index calculation
    are recognized automatically based on the image type:

        -  *gamma* (float): each voxel represents the gamma index value and the voxels excluded from the GI analysis have values -1 or numpy.nan.
        -  *pass-rate* (integer): each voxel represents passing (1) or falling (0) of the gamma index test and the voxels excluded from the GI analysis have values -1.

    Parameters
    ----------
    imgGI : SimpleITK Image
        Object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dict
        Dictionary with the gamma index statistics.

    See Also
    --------
    calcGammaIndex : calculate Gamma Index map for two images.
    """
    import fredtools as ft
    import numpy as np

    arrGI = ft.arr(imgGI)
    GIstat = {}
    if np.issubdtype(arrGI.dtype, np.integer):
        if not set(np.unique(arrGI)).issubset(set([-1, 0, 1])):
            raise ValueError(
                f"The calculation mode was recognised as 'pass-rate' because the input image is of integer type but it should contain only [-1, 0 , 1] unique values and the uniques are {np.unique(arrGI)}"
            )
        mode = "pass-rate"
        GIstat["passRate"] = (arrGI == 1).sum() / (arrGI >= 0).sum() * 100
        GIstat["mean"] = np.nan
        GIstat["std"] = np.nan
        GIstat["min"] = np.nan
        GIstat["max"] = np.nan

    elif np.issubdtype(arrGI.dtype, np.floating):
        mode = "gamma"
        arrGI[arrGI < 0] = np.NaN
        GIstat["passRate"] = (arrGI <= 1).sum() / (arrGI >= 0).sum() * 100
        GIstat["mean"] = np.nanmean(arrGI)
        GIstat["std"] = np.nanstd(arrGI)
        GIstat["min"] = np.nanmin(arrGI)
        GIstat["max"] = np.nanmax(arrGI)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# GIPR: {:.2f}".format(GIstat["passRate"]))
        if mode == "gamma":
            print("# mean/std: {:.2f} / {:.2f}".format(GIstat["mean"], GIstat["std"]))
            print("# min/max: {:.2f} / {:.2f}".format(GIstat["min"], GIstat["max"]))
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return GIstat
