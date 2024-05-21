def _generateSpatialCentresString(pixelCentres):
    """Generate voxels' centres string

    The function generates a formatted string with voxels' centres in one direction.
    This routine is useful for displaying image information.

    Parameters
    ----------
    pixelCentres : array_like
        A one-dimensional, array-like object of centres to be converted to a string.

    Returns
    -------
    string
        Formatted string
    """
    if len(pixelCentres) > 4:
        spatialCentresString = "[ {:12f}, {:12f}, ..., {:12f}, {:12f} ]".format(pixelCentres[0], pixelCentres[1], pixelCentres[-2], pixelCentres[-1])
    elif len(pixelCentres) == 4:
        spatialCentresString = "[ {:12f}, {:12f}, {:12f}, {:12f} ]".format(pixelCentres[0], pixelCentres[1], pixelCentres[2], pixelCentres[3])
    elif len(pixelCentres) == 3:
        spatialCentresString = "[ {:12f}, {:12f}, {:12f} ]".format(pixelCentres[0], pixelCentres[1], pixelCentres[2])
    elif len(pixelCentres) == 2:
        spatialCentresString = "[ {:12f}, {:12f} ]".format(pixelCentres[0], pixelCentres[1])
    elif len(pixelCentres) == 1:
        spatialCentresString = "[ {:12f} ]".format(pixelCentres[0])
    return spatialCentresString


def _generateExtentString(axisExtent):
    """Generate image extent string

    The function generates a formatted string with the extent in one direction.
    This routine is useful for displaying image information.

    Parameters
    ----------
    axisExtent : array_like
        A One-dimensional, two-element, array-like object to be converted to a string.

    Returns
    -------
    string
        Formatted string
    """
    import numpy as np

    if not len(axisExtent) == 2:
        raise ValueError(f"Extent of a single axis must be of length 2 and is {len(axisExtent)}.")
    extentString = "[ {:12f} , {:12f} ] => {:12f}".format(axisExtent[0], axisExtent[1], np.abs(np.diff(axisExtent)[0]))
    return extentString


def _displayImageInfo(img):
    """Display some information about the image without the name of the function.

    The function displays information about an image given as a SimpleITK image object.
    The information is displayed without the function name.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    """
    import SimpleITK as sitk
    import numpy as np
    import fredtools as ft

    import inspect
    print(__name__)
    print(inspect.getmodule(inspect.stack()[1][0]).__name__)
    logger = ft._getLogger(inspect.getmodule(inspect.stack()[1][0]).__name__)

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    extent_mm = ft.getExtent(img)
    size_mm = ft.getSize(img)
    axesNames = ["x", "y", "z", "t"]
    arr = sitk.GetArrayFromImage(img)
    voxelCentres = ft.getVoxelCentres(img)
    isVector = ft._imgTypeChecker.isSITK_vector(img)
    isMask = ft._imgTypeChecker.isSITK_mask(img)
    maskType = f"({ft._imgTypeChecker.getMaskType(img)} mask)" if isMask else ""
    isIdentity = ft.ImgAnalyse.imgAnalyse._checkIdentity(img)

    if ft._imgTypeChecker.isSITK_point(img):
        logger.info("{:d}D{:s} image describing a point (0D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))
    elif ft._imgTypeChecker.isSITK_profile(img):
        logger.info("{:d}D{:s} image describing a profile (1D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))
    elif ft._imgTypeChecker.isSITK_slice(img):
        logger.info("{:d}D{:s} image describing a slice (2D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))
    elif ft._imgTypeChecker.isSITK_volume(img):
        logger.info("{:d}D{:s} image describing a volume (3D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))
    elif ft._imgTypeChecker.isSITK_timevolume(img):
        logger.info("{:d}D{:s} image describing a time volume (4D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))

    if not isIdentity:
        logger.info("# The image direction is not identity")

    logger.info(f"dims ({''.join(axesNames[: img.GetDimension()])})      = {np.array(img.GetSize())}")
    logger.info(f"voxel size [mm] = {np.array(img.GetSpacing())}")
    logger.info(f"origin [mm]     = {np.array(img.GetOrigin())}")
    for vox, axisName in zip(voxelCentres, axesNames):
        logger.info(f"{axisName}-spatial voxel centre [mm] = {_generateSpatialCentresString(vox)}")
    for ext, axisName in zip(extent_mm, axesNames):
        logger.info(f"{axisName}-spatial extent [mm] = {_generateExtentString(ext)}")

    realSize = [size for idx, size in enumerate(ft.getSize(img)) if img.GetSize()[idx] > 1]
    if ft._imgTypeChecker.isSITK_profile(img):
        logger.info("length = {:.2f} mm  =>  {:.2f} cm".format(np.prod(realSize), np.prod(realSize) / 1e1))
    elif ft._imgTypeChecker.isSITK_slice(img):
        logger.info("area = {:.2f} mm²  =>  {:.2f} cm²".format(np.prod(realSize), np.prod(realSize) / 1e2))
    elif ft._imgTypeChecker.isSITK_volume(img):
        logger.info("volume = {:.2f} mm³  =>  {:.2f} l".format(np.prod(realSize), np.prod(realSize) / 1e6))
    elif ft._imgTypeChecker.isSITK_timevolume(img):
        logger.info("time volume = {:.2f} mm³*s  =>  {:.2f} l*s".format(np.prod(realSize), np.prod(realSize) / 1e6))

    realVoxelSize = [size for idx, size in enumerate(img.GetSpacing()) if img.GetSize()[idx] > 1]
    if ft._imgTypeChecker.isSITK_profile(img):
        logger.info("step size = {:.2f} mm  =>  {:.2f} cm".format(np.prod(realVoxelSize), np.prod(realVoxelSize) / 1e1))
    elif ft._imgTypeChecker.isSITK_slice(img):
        logger.info("pixel area = {:.2f} mm²  =>  {:.2f} cm²".format(np.prod(realVoxelSize), np.prod(realVoxelSize) / 1e2))
    elif ft._imgTypeChecker.isSITK_volume(img):
        logger.info("voxel volume = {:.2f} mm³  =>  {:.2f} ul".format(np.prod(realVoxelSize), np.prod(realVoxelSize)))
    elif ft._imgTypeChecker.isSITK_timevolume(img):
        logger.info("voxel time volume = {:.2f} mm³*s  =>  {:.2f} ul*s".format(np.prod(realVoxelSize), np.prod(realVoxelSize)))

    logger.info(f"data type: {img.GetPixelIDTypeAsString()}" + ("with NaN values" if np.any(np.isnan(arr)) else ""))
    logger.info(f"range: from {np.nanmin(arr)} to {np.nanmax(arr)}" + (" (sum of vectors)" if isVector else ""))
    logger.info(f"sum = {np.nansum(arr)}, mean = {np.nanmean(arr)} ({np.nanstd(arr)})" + ("(sum of vectors)" if isVector else ""))

    nonZeroVoxels = (arr[~np.isnan(arr)] != 0).sum()
    nonAirVoxels = (arr[~np.isnan(arr)] > -1000).sum()

    if ft._imgTypeChecker.isSITK_profile(img):
        logger.info("non-zero (dose=0)  values  = {:d} ({:.2%}) => {:.2f} cm".format(nonZeroVoxels, nonZeroVoxels / arr.size, np.prod(realVoxelSize) * nonZeroVoxels / 1e1) + ("(sum of vectors)" if isVector else ""))
        logger.info("non-air (HU>-1000) values  = {:d} ({:.2%}) => {:.2f} cm".format(nonAirVoxels, nonAirVoxels / arr.size, np.prod(realVoxelSize) * nonAirVoxels / 1e1) + ("(sum of vectors)" if isVector else ""))
    elif ft._imgTypeChecker.isSITK_slice(img):
        logger.info("non-zero (dose=0)  pixels  = {:d} ({:.2%}) => {:.2f} cm²".format(nonZeroVoxels, nonZeroVoxels / arr.size, np.prod(realVoxelSize) * nonZeroVoxels / 1e2) + ("(sum of vectors)" if isVector else ""))
        logger.info("non-air (HU>-1000) pixels  = {:d} ({:.2%}) => {:.2f} cm²".format(nonAirVoxels, nonAirVoxels / arr.size, np.prod(realVoxelSize) * nonAirVoxels / 1e2) + ("(sum of vectors)" if isVector else ""))
    elif ft._imgTypeChecker.isSITK_volume(img):
        logger.info("non-zero (dose=0)  voxels  = {:d} ({:.2%}) => {:.2f} l".format(nonZeroVoxels, nonZeroVoxels / arr.size, np.prod(realVoxelSize) * nonZeroVoxels / 1e6) + ("(sum of vectors)" if isVector else ""))
        logger.info("non-air (HU>-1000) voxels  = {:d} ({:.2%}) => {:.2f} l".format(nonAirVoxels, nonAirVoxels / arr.size, np.prod(realVoxelSize) * nonAirVoxels / 1e6) + ("(sum of vectors)" if isVector else ""))
    elif ft._imgTypeChecker.isSITK_timevolume(img):
        logger.info("non-zero (dose=0)  time voxels  = {:d} ({:.2%}) => {:.2f} l*s".format(nonZeroVoxels, nonZeroVoxels / arr.size, np.prod(realVoxelSize) * nonZeroVoxels / 1e6) + ("(sum of vectors)" if isVector else ""))
        logger.info("non-air (HU>-1000) time voxels  = {:d} ({:.2%}) => {:.2f} l*s".format(nonAirVoxels, nonAirVoxels / arr.size, np.prod(realVoxelSize) * nonAirVoxels / 1e6) + ("(sum of vectors)" if isVector else ""))

    if img.GetMetaDataKeys():
        logger.info("# Additional metadata:")
        keyLen = []
        for key in img.GetMetaDataKeys():
            keyLen.append(len(key))
        keyLen = np.max(keyLen)
        for key in img.GetMetaDataKeys():
            if "UNKNOWN_PRINT_CHARACTERISTICS" in img.GetMetaData(key):
                continue
            logger.info(f"{key.ljust(keyLen)} : {img.GetMetaData(key)}")


def displayImageInfo(img):
    """Display some image information.

    The function displays information about an image given as a SimpleITK image object.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.

    Examples
    --------
    Reading image from file and displaying the image information.

    >>> imgCT=fredtools.readMHD('CT.mhd')
    >>> fredtools.displayImageInfo(imgCT)
    ### displayImageInfo ###
    # 3D image describing volume (3D)
    # dims (xyz) =  [511 415 218]
    # voxel size [mm] =  [0.68359375 0.68359375 1.2       ]
    # origin [mm]     =  [-174.65820312 -354.28710938 -785.6       ]
    # x-spatial voxel centre [mm] =  [  -174.658203,  -173.974609, ...,   173.291016,   173.974609 ]
    # y-spatial voxel centre [mm] =  [  -354.287109,  -353.603516, ...,   -71.962891,   -71.279297 ]
    # z-spatial voxel centre [mm] =  [  -785.600000,  -784.400000, ...,  -526.400000,  -525.200000 ]
    # x-spatial extent [mm] =  [  -175.000000 ,   174.316406 ] =>   349.316406
    # y-spatial extent [mm] =  [  -354.628906 ,   -70.937500 ] =>   283.691406
    # z-spatial extent [mm] =  [  -786.200000 ,  -524.600000 ] =>   261.600000
    # volume = 25924053.15 mm³  =>  25.92 l
    # voxel volume = 0.56 mm³  =>  0.56 ul
    # data type:  16-bit signed integer
    # range: from  -1024  to  3071
    # sum = -33870013138 , mean = -732.6387321958799 ( 468.4351806663016 )
    # non-zero (dose=0)  voxels  = 46188861 (99.91%) => 25.90 l
    # non-air (HU>-1000) voxels  = 15065800 (32.59%) => 8.45 l
    # Additional metadata:
    ########################

    The same can be displayed when reading the image.

    >>> imgCT=fredtools.readMHD('CT.mhd', displayInfo=True)
    ### readMHD ###
    # 3D image describing volume (3D)
    # dims (xyz) =  [511 415 218]
    # voxel size [mm] =  [0.68359375 0.68359375 1.2       ]
    # origin [mm]     =  [-174.65820312 -354.28710938 -785.6       ]
    # x-spatial voxel centre [mm] =  [  -174.658203,  -173.974609, ...,   173.291016,   173.974609 ]
    # y-spatial voxel centre [mm] =  [  -354.287109,  -353.603516, ...,   -71.962891,   -71.279297 ]
    # z-spatial voxel centre [mm] =  [  -785.600000,  -784.400000, ...,  -526.400000,  -525.200000 ]
    # x-spatial extent [mm] =  [  -175.000000 ,   174.316406 ] =>   349.316406
    # y-spatial extent [mm] =  [  -354.628906 ,   -70.937500 ] =>   283.691406
    # z-spatial extent [mm] =  [  -786.200000 ,  -524.600000 ] =>   261.600000
    # volume = 25924053.15 mm³  =>  25.92 l
    # voxel volume = 0.56 mm³  =>  0.56 ul
    # data type:  16-bit signed integer
    # range: from  -1024  to  3071
    # sum = -33870013138 , mean = -732.6387321958799 ( 468.4351806663016 )
    # non-zero (dose=0)  voxels  = 46188861 (99.91%) => 25.90 l
    # non-air (HU>-1000) voxels  = 15065800 (32.59%) => 8.45 l
    # Additional metadata:
    ###############
    """
    import fredtools as ft
    logger = ft._getLogger(__name__)

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    logger.info(f"{ft._helper.currentFuncName()}")
    _displayImageInfo(img)
    logger.info("#" * len(f"### {ft._helper.currentFuncName()} ###"))
