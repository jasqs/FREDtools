def getExtent(img, displayInfo=False):
    """Get the extent of an image.

    The function gets the extent of an image defined as a SimpleITK image object.
    Extent means the coordinates of most side voxels' borders
    in each direction. It is assumed that the coordinate system is in [mm].

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    tuple
        A tuple of extent values in form ((xmin,xmax),(ymin,ymax),...)

    See Also
    --------
    getExtMpl : get the extent of a SimpleITK image object describing a slice in matplotlib format.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    cornerLow = img.TransformContinuousIndexToPhysicalPoint(np.zeros(img.GetDimension(), dtype="float64") - 0.5)
    cornerHigh = img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize(), dtype="float64") - 1 + 0.5)

    cornerLow = np.dot(np.abs(ft.ft_imgAnalyse._getDirectionArray(img)).T, cornerLow)
    cornerHigh = np.dot(np.abs(ft.ft_imgAnalyse._getDirectionArray(img)).T, cornerHigh)

    extent = tuple(zip(cornerLow, cornerHigh))

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        axesNames = ["x", "y", "z", "t"]
        for ext, axisName in zip(extent, axesNames):
            print("# {:s}-spatial extent [mm] = ".format(axisName), _generateExtentString(ext))
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return extent


def getSize(img, displayInfo=False):
    """Get the size of an image.

    The function gets the size of an image defined as a SimpleITK image object
    in each direction. The size is defined as the absolute difference of
    the image extent. It is assumed that the coordinate system is in [mm].

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    tuple
        A Tuple of sizes in each direction in form (xSize,ySize,...)

    See Also
    --------
    getExtent : get the extent of an image.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    size = tuple(np.abs(np.diff(getExtent(img))).squeeze())

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        axesNames = ["x", "y", "z", "t"]
        for siz, axisName in zip(size, axesNames):
            print("# {:s}-spatial size [mm] = ".format(axisName), siz)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return size


def getImageCenter(img, displayInfo=False):
    """Get the centre of an image.

    The function calculates the centre of an image defined as a SimpleITK image object
    in each direction. It is assumed that the coordinate system is in [mm].

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    tuple
        A Tuple of image centre coordinates in form (xCenter,yCenter,...)

    See Also
    --------
    getMassCenter : get the centre of mass of an image.
    getMaxPosition : get the position of an image maximum.
    getMinPosition : get the position of an image minimum.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    imageCentre = tuple(np.mean(np.array(getExtent(img)), 1))

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        axesNames = ["x", "y", "z", "t"]
        for cent, axisName in zip(imageCentre, axesNames):
            print("# {:s}-spatial image centre [mm] = ".format(axisName), cent)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return imageCentre


def getMassCenter(img, displayInfo=False):
    """Get the centre of mass of an image.

    The function calculates the centre of mass of an image defined as
    a SimpleITK image object in each direction. It is assumed that
    the coordinate system is in [mm].

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    tuple
        A tuple of image centre of mass coordinates in form (xMassCenter,yMassCenter,...)

    See Also
    --------
    getImageCenter : get the centre of an image.
    getMaxPosition : get the position of an image maximum.
    getMinPosition : get the position of an image minimum.
    """
    import numpy as np
    import itk
    import SimpleITK as sitk
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    # check if are nan values in the image and replace them with zeros
    arr = sitk.GetArrayFromImage(img)
    if np.any(np.isnan(arr)):
        imgOrg = img
        arr[np.isnan(arr)] = 0
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(imgOrg)

    imgITK = ft.SITK2ITK(img)
    moments = itk.ImageMomentsCalculator.New(imgITK)
    moments.Compute()
    massCentre = np.dot(np.abs(ft.ft_imgAnalyse._getDirectionArray(img)).T, moments.GetCenterOfGravity())
    massCentre = tuple(massCentre)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        axesNames = ["x", "y", "z", "t"]
        for cent, axisName in zip(massCentre, axesNames):
            print("# {:s}-spatial mass centre [mm] = ".format(axisName), cent)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return massCentre


def getMaxPosition(img, displayInfo=False):
    """Get the maximum position of an image.

    The function calculates the position of the maximum voxel of
    an image defined as a SimpleITK image object.
    It is assumed that the coordinate system is in [mm].

    Parameters
    ----------
    img : SimpleITK Image
        The object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    tuple
        A Tuple of image maximum voxel coordinates in form (xPosition,yPosition,...).

    See Also
    --------
    getImageCenter : get the centre of an image.
    getMassCenter : get the centre of mass of an image.
    getMinPosition : get the position of an image minimum.
    """
    import numpy as np
    import fredtools as ft
    import warnings

    ft._isSITK(img, raiseError=True)

    # convert image to array
    arr = ft.arr(img)

    # check if only one maximum value exists and raise warning
    if (arr == np.nanmax(arr)).sum() > 1:
        warnings.warn("Warning: more than one maximum value was found. The first one was returned.")

    # get maximum position
    maxPosition = np.unravel_index(np.nanargmax(arr), arr.shape[::-1], order="F")
    maxPosition = img.TransformIndexToPhysicalPoint([int(pos) for pos in maxPosition])

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        axesNames = ["x", "y", "z", "t"]
        for cent, axisName in zip(maxPosition, axesNames):
            print("# {:s}-spatial max position [mm] = ".format(axisName), cent)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return maxPosition


def getMinPosition(img, displayInfo=False):
    """Get the minimum position of an image.

    The function calculates the position of the minimum voxel of
    an image defined as a SimpleITK image object.
    It is assumed that the coordinate system is in [mm].

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    tuple
        A Tuple of image minimum voxel coordinates in form (xPosition,yPosition,...).

    See Also
    --------
    getImageCenter : get the centre of an image.
    getMassCenter : get the centre of mass of an image.
    getMaxPosition : get the position of an image maximum.
    """
    import numpy as np
    import itk
    import SimpleITK as sitk
    import fredtools as ft
    import warnings

    ft._isSITK(img, raiseError=True)

    # convert image to array
    arr = ft.arr(img)

    # check if only one minimum value exists and raise warning
    if (arr == np.nanmin(arr)).sum() > 1:
        warnings.warn("Warning: more than one minimum value was found. The first one was returned.")

    # get minimum position
    minPosition = np.unravel_index(np.nanargmin(arr), arr.shape[::-1], order="F")
    minPosition = img.TransformIndexToPhysicalPoint([int(pos) for pos in minPosition])

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        axesNames = ["x", "y", "z", "t"]
        for cent, axisName in zip(minPosition, axesNames):
            print("# {:s}-spatial min position [mm] = ".format(axisName), cent)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return minPosition


def getVoxelCentres(img, displayInfo=False):
    """Get voxel centres.

    The function gets voxels' centres in each direction of an image
    defined as a SimpleITK image object. It is assumed that the coordinate
    system is in [mm].

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    tuple
        A tuple of tuples with voxels' centres in each
        direction in form ([x0,x1,...],[y0,y1,...],...)

    See Also
    --------
    getVoxelEdges : get voxel edges.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    voxelCentres = []
    for axis, axisSize in enumerate(img.GetSize()):
        voxelIndices = np.zeros([axisSize, 3], dtype=np.int32)
        voxelIndices[:, axis] = np.arange(axisSize, dtype=np.int32)
        voxelCentresAxis = np.array(ft.transformIndexToPhysicalPoint(img, voxelIndices))
        voxelCentres.append(tuple(voxelCentresAxis[:, axis]))

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        axesNames = ["x", "y", "z", "t"]
        for vox, axisName in zip(voxelCentres, axesNames):
            print("# {:s}-spatial voxel centre [mm] = ".format(axisName), _generateSpatialCentresString(vox))
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return tuple(voxelCentres)


def getVoxelEdges(img, displayInfo=False):
    """Get voxel edges.

    The function gets voxels' edges in each direction of an image
    defined as a SimpleITK image object. It is assumed that the coordinate
    system is in [mm].

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    tuple
        A tuple of tuples with voxels' edges in each
        direction in form ([x0,x1,...],[y0,y1,...],...)

    See Also
    --------
    getVoxelCentres : get voxel centres.
    """
    import fredtools as ft
    import numpy as np

    ft._isSITK(img, raiseError=True)

    voxelEdges = []
    for axis, axisSize in enumerate(img.GetSize()):
        voxelIndices = np.zeros([axisSize + 1, 3], dtype=np.float64)
        voxelIndices[:, axis] = np.arange(axisSize + 1, dtype=np.float64) - 0.5
        voxelEdgesAxis = np.array(ft.transformContinuousIndexToPhysicalPoint(img, voxelIndices))
        voxelEdges.append(tuple(voxelEdgesAxis[:, axis]))

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        axesNames = ["x", "y", "z", "t"]
        for vox, axisName in zip(voxelEdges, axesNames):
            print("# {:s}-spatial voxel edge [mm] = ".format(axisName), _generateSpatialCentresString(vox))
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return tuple(voxelEdges)


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

    ft._isSITK(img, raiseError=True)

    extent_mm = ft.getExtent(img)
    size_mm = ft.getSize(img)
    axesNames = ["x", "y", "z", "t"]
    arr = sitk.GetArrayFromImage(img)
    voxelCentres = ft.getVoxelCentres(img)
    isVector = ft._isSITK_vector(img)
    isMask = ft._isSITK_mask(img)
    maskType = f"({ft._getMaskType(img)} mask)" if isMask else ""
    isIdentity = ft.ft_imgAnalyse._checkIdentity(img)

    if ft._isSITK_point(img):
        print("# {:d}D{:s} image describing a point (0D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))
    elif ft._isSITK_profile(img):
        print("# {:d}D{:s} image describing a profile (1D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))
    elif ft._isSITK_slice(img):
        print("# {:d}D{:s} image describing a slice (2D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))
    elif ft._isSITK_volume(img):
        print("# {:d}D{:s} image describing a volume (3D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))
    elif ft._isSITK_timevolume(img):
        print("# {:d}D{:s} image describing a time volume (4D) {:s}".format(img.GetDimension(), " vector" if isVector else "", maskType))

    if not isIdentity:
        print("# The image direction is not identity")

    print("# dims ({:s}) = ".format("".join(axesNames[: img.GetDimension()])), np.array(img.GetSize()))
    print("# voxel size [mm] = ", np.array(img.GetSpacing()))
    print("# origin [mm]     = ", np.array(img.GetOrigin()))
    for vox, axisName in zip(voxelCentres, axesNames):
        print("# {:s}-spatial voxel centre [mm] = ".format(axisName), ft.ft_imgAnalyse._generateSpatialCentresString(vox))
    for ext, axisName in zip(extent_mm, axesNames):
        print("# {:s}-spatial extent [mm] = ".format(axisName), ft.ft_imgAnalyse._generateExtentString(ext))

    realSize = [size for idx, size in enumerate(ft.getSize(img)) if img.GetSize()[idx] > 1]
    if ft._isSITK_profile(img):
        print("# length = {:.2f} mm  =>  {:.2f} cm".format(np.prod(realSize), np.prod(realSize) / 1e1))
    elif ft._isSITK_slice(img):
        print("# area = {:.2f} mm²  =>  {:.2f} cm²".format(np.prod(realSize), np.prod(realSize) / 1e2))
    elif ft._isSITK_volume(img):
        print("# volume = {:.2f} mm³  =>  {:.2f} l".format(np.prod(realSize), np.prod(realSize) / 1e6))
    elif ft._isSITK_timevolume(img):
        print("# time volume = {:.2f} mm³*s  =>  {:.2f} l*s".format(np.prod(realSize), np.prod(realSize) / 1e6))

    realVoxelSize = [size for idx, size in enumerate(img.GetSpacing()) if img.GetSize()[idx] > 1]
    if ft._isSITK_profile(img):
        print("# step size = {:.2f} mm  =>  {:.2f} cm".format(np.prod(realVoxelSize), np.prod(realVoxelSize) / 1e1))
    elif ft._isSITK_slice(img):
        print("# pixel area = {:.2f} mm²  =>  {:.2f} cm²".format(np.prod(realVoxelSize), np.prod(realVoxelSize) / 1e2))
    elif ft._isSITK_volume(img):
        print("# voxel volume = {:.2f} mm³  =>  {:.2f} ul".format(np.prod(realVoxelSize), np.prod(realVoxelSize)))
    elif ft._isSITK_timevolume(img):
        print("# voxel time volume = {:.2f} mm³*s  =>  {:.2f} ul*s".format(np.prod(realVoxelSize), np.prod(realVoxelSize)))

    print("# data type: ", img.GetPixelIDTypeAsString(), "with NaN values" if np.any(np.isnan(arr)) else "")
    print("# range: from ", np.nanmin(arr), " to ", np.nanmax(arr), "(sum of vectors)" if isVector else "")
    print("# sum =", np.nansum(arr), ", mean =", np.nanmean(arr), "(",    np.nanstd(arr), ")", "(sum of vectors)" if isVector else "")

    nonZeroVoxels = (arr[~np.isnan(arr)] != 0).sum()
    nonAirVoxels = (arr[~np.isnan(arr)] > -1000).sum()

    if ft._isSITK_profile(img):
        print("# non-zero (dose=0)  values  = {:d} ({:.2%}) => {:.2f} cm".format(nonZeroVoxels, nonZeroVoxels / arr.size, np.prod(realVoxelSize) * nonZeroVoxels / 1e1), "(sum of vectors)" if isVector else "")
        print("# non-air (HU>-1000) values  = {:d} ({:.2%}) => {:.2f} cm".format(nonAirVoxels, nonAirVoxels / arr.size, np.prod(realVoxelSize) * nonAirVoxels / 1e1), "(sum of vectors)" if isVector else "")
    elif ft._isSITK_slice(img):
        print("# non-zero (dose=0)  pixels  = {:d} ({:.2%}) => {:.2f} cm²".format(nonZeroVoxels, nonZeroVoxels / arr.size, np.prod(realVoxelSize) * nonZeroVoxels / 1e2), "(sum of vectors)" if isVector else "")
        print("# non-air (HU>-1000) pixels  = {:d} ({:.2%}) => {:.2f} cm²".format(nonAirVoxels, nonAirVoxels / arr.size, np.prod(realVoxelSize) * nonAirVoxels / 1e2), "(sum of vectors)" if isVector else "")
    elif ft._isSITK_volume(img):
        print("# non-zero (dose=0)  voxels  = {:d} ({:.2%}) => {:.2f} l".format(nonZeroVoxels, nonZeroVoxels / arr.size, np.prod(realVoxelSize) * nonZeroVoxels / 1e6), "(sum of vectors)" if isVector else "")
        print("# non-air (HU>-1000) voxels  = {:d} ({:.2%}) => {:.2f} l".format(nonAirVoxels, nonAirVoxels / arr.size, np.prod(realVoxelSize) * nonAirVoxels / 1e6), "(sum of vectors)" if isVector else "")
    elif ft._isSITK_timevolume(img):
        print("# non-zero (dose=0)  time voxels  = {:d} ({:.2%}) => {:.2f} l*s".format(nonZeroVoxels, nonZeroVoxels / arr.size, np.prod(realVoxelSize) * nonZeroVoxels / 1e6), "(sum of vectors)" if isVector else "")
        print("# non-air (HU>-1000) time voxels  = {:d} ({:.2%}) => {:.2f} l*s".format(nonAirVoxels, nonAirVoxels / arr.size, np.prod(realVoxelSize) * nonAirVoxels / 1e6), "(sum of vectors)" if isVector else "")

    if img.GetMetaDataKeys():
        print("# Additional metadata:")
        keyLen = []
        for key in img.GetMetaDataKeys():
            keyLen.append(len(key))
        keyLen = np.max(keyLen)
        for key in img.GetMetaDataKeys():
            if "UNKNOWN_PRINT_CHARACTERISTICS" in img.GetMetaData(key):
                continue
            print(f"# {key.ljust(keyLen)} : {img.GetMetaData(key)}")


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

    ft._isSITK(img, raiseError=True)

    print(f"### {ft._currentFuncName()} ###")
    _displayImageInfo(img)
    print("#" * len(f"### {ft._currentFuncName()} ###"))


def _getAxesVectorNotUnity(img):
    """Get a boolean vector of axes size unitary.

    The function calculates a boolean vector of size equal to the image dimension,
    in which the True values represent those axes whose size is more than one.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.

    Returns
    -------
    tuple
        Tuple of the length of the `img` dimension with 1/0 (True/False) values.

    Examples
    --------
    Assuming that the `img` shape is [200,1,100,400] (4D image).

    >>> fredtools.ft_imgAnalyse._getAxesVectorNotUnity(img)
    (1,0,1,1)
    """
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    axesVectorNotUnity = [int(axis != 1) for axis in img.GetSize()]

    return tuple(axesVectorNotUnity)


def _getAxesNumberNotUnity(img):
    """Get axis indexes for the axes of size different than one.

    The function calculates the indexes of the axes for which the size is more than one.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.

    Returns
    -------
    tuple
        Tuple axis indexes.

    Examples
    --------
    Assuming that the `img` shape is [200,1,100,400] (4D image).

    >>> fredtools.ft_imgAnalyse._getAxesNumberNotUnity(img)
    (0, 2, 3)
    """
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    axesNumberNotUnity = [axis_idx for axis_idx, axis in enumerate(img.GetSize()) if axis != 1]

    return tuple(axesNumberNotUnity)


def _getDirectionArray(img):
    """Get direction in the form of a 2D array.

    The function converts direction from an image defined as a SimpleITK image
    object to a 2D numpy array. It reshapes img.GetDirection() results
    in a tuple to a 2D numpy array.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.

    Returns
    -------
    numpy
        A 2D numpy array with direction.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    return np.array(img.GetDirection()).reshape(img.GetDimension(), img.GetDimension())


def _checkIdentity(img):
    """Check image identity.

    The function checks if the image direction represents the identity matrix
    (1 in diagonal).

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.

    Returns
    -------
    bool
        Is it an identity or not.
    """
    import fredtools as ft
    import numpy as np

    ft._isSITK(img, raiseError=True)

    return np.all(np.identity(img.GetDimension(), dtype="int").flatten() == img.GetDirection())


def getExtMpl(img):
    """Get the extent of a slice in a format consistent with imshow of matplotlib module.

    The function gets the extent of an image defined as a SimpleITK image object
    describing a slice. Extent means the coordinates of the most side pixels' borders
    in each direction and are returned in format (left, right, bottom, top), which is
    consistent with the ``extent`` optional parameter of ``imshow`` of
    ``matplotlib.pyplot`` module.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image describing a slice.

    Returns
    -------
    tuple
        A tuple of extent values in the form (left, right, bottom, top).

    See Also
    --------
        matplotlib.pyplot.imshow : displaying 2D images.
        getExtent : get the extent of the image in each direction.
        getSize : get the size of the image in each direction.
        arr : get squeezed array from image.

    Examples
    --------
    Assuming that the `img` is describing a slice (e.g. the shape is [100,1,300,1]), the line:

    >>> matplotlib.pyplot.imshow(fredtools.arr(img), extent=fredtools.getExtMpl(img))

    will display a 2D image of the shape 100x300 px in real coordinates.
    """
    import fredtools as ft

    ft._isSITK_slice(img, raiseError=True)

    extent = [extent for axis_idx, extent in enumerate(ft.getExtent(img)) if axis_idx in _getAxesNumberNotUnity(img)]
    extent[1] = extent[1][::-1]
    extent = [inner for outer in extent for inner in outer]

    return tuple(extent)


def pos(img):
    """Get voxels' centres for axes of the size different than one.

    The function calculates the voxels' centres of an image defined
    as a SimpleITK image object in each direction, only for those axes for
    which the size is more than one. The function is useful for plotting profiles.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.

    Returns
    -------
    tuple or list of tuples
        A tuple with voxels' centres for the image describing a profile
        (the size in only one direction is greater than one) or list of tuples
        for all directions for which the size is greater than one.

    See Also
    --------
        getVoxelCentres : get voxels' centres of the image in each direction.
        vec : get a vector with values for the image describing a profile.

    Examples
    --------
    Let's assume that the `img` is a 3D image describing a profile, so is of size [1,200,1].
    It is possible to get the voxels' centres only in the Y direction:

    >>> fredtools.pos(img)
    (-174.658203125,
     -173.974609375,
     ...
    )

    This can be used for plotting profiles. The line:

    >>> matplotlib.pyplot.plot(fredtools.pos(img), fredtools.vec(img))

    will plot the profile in the Y direction of the image.
    """
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    pos = ft.getVoxelCentres(img)
    pos = [pos[i] for i in range(img.GetDimension()) if not img.GetSize()[i] == 1]

    if len(pos) == 1:
        pos = pos[0]

    return pos


def arr(img):
    """Get squeezed array from image.

    The function gets a squeezed array (with no unitary dimensions) from
    an image defined as a SimpleITK image object. This can be used for plotting

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.

    Returns
    -------
    numpy array
        A numpy array with no unitary dimensions.

    See Also
    --------
        pos : get voxels' centres for axes of size different than one.
        getExtMpl : Get the extent of a slice in a format consistent with imshow of matplotlib module.
        vec : get a vector with values for the image describing a profile.

    Examples
    --------
    Assuming that the `img` is describing a slice (e.g. the shape is [100,1,300,1]), the line:

    >>> matplotlib.pyplot.imshow(fredtools.arr(img), extent=fredtools.getExtMpl(img))

    will display a 2D image of the shape 100x300 px in real coordinates.
    """
    import SimpleITK as sitk
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    arr = sitk.GetArrayFromImage(img).squeeze()

    return arr


def vec(img):
    """Get 1D array from image describing a profile.

    The function gets a squeezed (with no unitary dimensions), 1D array from
    an image defined as a SimpleITK image object describing a profile.
    This can be used for plotting

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image describing a profile.

    Returns
    -------
    numpy array
        A 1D numpy array.

    See Also
    --------
        pos : get voxels' centres for axes of the size different than one.
        arr : Get squeezed array from image.

    Examples
    --------
    Assuming that the `img` is describing a profile (e.g. the shape is [1,1,300,1]), the line:

    >>> matplotlib.pyplot.plot(fredtools.pos(img), fredtools.vec(img))

    will plot the profile in the Z direction of the image.
    """
    import SimpleITK as sitk
    import fredtools as ft

    ft._isSITK_profile(img, raiseError=True)
    arr = sitk.GetArrayFromImage(img).squeeze()

    return arr


def isPointInside(img, point, displayInfo=False):
    """Check if a point is inside the image extent.

    The function checks if a point or list of points are inside
    the extent of an image defined as a SimpleITK image object
    The dimension of points must match the dimension of the img.
    The points at the border of the image (equal to the image extent)
    are considered to be inside the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    point : NxD array_like
        An iterable (numpy array, list of lists, etc) of N points.
        Every point must be of the image dimension size.        
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    single or tuple
        A single or a tuple of boolean values.

    See Also
    --------
        getExtent : get the extent of the image in each direction.

    Examples
    --------
    Let's assume that the `img` is a 3D image with extent:

    >>> fredtools.getExtent(img)
    ((-175.0, 174.3),
     (-354.6, -70.9),
     (-786.2, -524.6))

    It means that the `img` expands from -175.0 to 174.3, from -354.6 to-70.9
    and from -786.2 to -524.6 in X,Y and Z directions, respectively. Let's check
    if the point [0,0,0] and a list of points [[0,0,0],[0,-100,-600],[-175,-354.6,-786.2]]
    are inside the image extent:

    >>> fredtools.isPointInside(img, [0,0,0])
    False
    >>> fredtools.isPointInside(img, [[0,0,0],[0,-100,-600],[-175,-354.6,-786.2]])
    (False, True, True)
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    # convert point to numpy array
    point = np.array(point, ndmin=1)

    # check if it is 1D array and expand to 2D if needed
    if point.ndim == 1:
        point = np.expand_dims(point, 0)

    # get extent of an sitk image
    extents = ft.getExtent(img)
    # convert tuple of tuples to list of lists and sort each sublist
    extents = [list(sorted(x)) for x in extents]

    isIns = []
    for axis in range(point.shape[1]):
        isIns.append((point[:, axis] >= extents[axis][0]) & (point[:, axis] <= extents[axis][1]))
    isIns = list(np.array(isIns).all(axis=0))

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        if all(isIns):
            print("# All points are inside the image")
        else:
            print("# Not all points are inside the image")
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return isIns[0] if len(isIns) == 1 else tuple(isIns)


def getStatistics(img, displayInfo=False):
    """Get statistics of image

    The function gets basic statistics of an image defined as
    a SimpleITK image object. It is a wrapper for
    SimpleITK.StatisticsImageFilter routine executed on the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK filter
        SimpleITK image filter with image statistics.

    See Also
    --------
        SimpleITK.StatisticsImageFilter : more about possible statistics.

    Examples
    --------
    It is assumed that the image is a 3D dose distribution.
    Some statistics of the image, like the mean, standard deviation
    or sum can be calculated.

    >>> stat=fredtools.getStatistics(img)
    >>> stat.GetMean()
    -732.6387321958799
    >>> stat.GetSigma()
    468.4351857326367
    >>> stat.GetSum()
    -33870013138.0
    """
    import SimpleITK as sitk
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    isVector = False
    if ft._isSITK_vector(img):
        img = ft.ft_imgManipulate.sumVectorImg(img)
        isVector = True

    stat = sitk.StatisticsImageFilter()
    stat.Execute(img)
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        if isVector:
            print("# Warning: The input image is a vector image. Statistics shown for the sum of vectors.")
        print("# Image mean/std: ", stat.GetMean(), "/", stat.GetSigma())
        print("# Image min/max: ", stat.GetMinimum(), "/", stat.GetMaximum())
        print("# Image sum: ", stat.GetSum())
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return stat


def compareImgFoR(img1, img2, decimals=3, displayInfo=False):
    """Compare two images frame of reference

    The function gets two images defined as instances of a SimpleITK image
    object and compares the frame of reference, i.e. dimension, size, origin,
    spacing and direction.

    Parameters
    ----------
    img1 : SimpleITK Image
        An object of a SimpleITK image.
    img2 : SimpleITK Image
        An object of a SimpleITK image.
    decimals: int, optional
        Use rounding to a given number of decimals when comparing origin and spacing. (def. 3)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    bool
        A true/false value describing if the images are the same
        in the sense of the field of reference.

    See Also
    --------
        SimpleITK.StatisticsImageFilter : more about possible statistics.
    """
    import fredtools as ft
    import numpy as np

    ft._isSITK(img1, raiseError=True)
    ft._isSITK(img2, raiseError=True)

    # compare dimension
    dimensionMatch = img1.GetDimension() == img2.GetDimension()

    # compare size
    sizeMatch = img1.GetSize() == img2.GetSize()

    # compare origin
    originMatch = np.all(np.round(np.array(img1.GetOrigin()), decimals=decimals) == np.round(np.array(img2.GetOrigin()), decimals=decimals))

    # compare spacing
    spacingMatch = np.all(np.round(np.array(img1.GetSpacing()), decimals=decimals) == np.round(np.array(img2.GetSpacing()), decimals=decimals))

    # compare direction
    directionMatch = np.all(np.array(img1.GetDirection()) == np.array(img2.GetDirection()))

    match = (dimensionMatch and sizeMatch and originMatch and spacingMatch and directionMatch)

    # compare values if displayInfo
    if match and displayInfo:
        valuesMatch = np.all(np.round(ft.arr(img1), decimals=decimals) == np.round(ft.arr(img2), decimals=decimals))
    else:
        valuesMatch = False

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Dimension matching:      ", dimensionMatch)
        print("# Size matching:           ", sizeMatch)
        print("# Origin matching:         ", originMatch, f"({decimals} decimals tolerance)")
        print("# Spacing matching:        ", spacingMatch, f"({decimals} decimals tolerance)")
        print("# Direction matching:      ", directionMatch)
        print("# Pixel-to-pixel matching: ", valuesMatch)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return match


def transformIndexToPhysicalPoint(img, indices):
    """Transform indices to physical points.

    The function transforms an iterable of indices into a tuple of
    physical points based on the field of reference (FoR) of an image
    defined as an instance of a SimpleITK image object. The function is
    a wrapper for `TransformIndexToPhysicalPoint` SimpleITK function, but
    it works for multiple points. The shape of `indices` must be NxD, where N
    is the number of points to be converted, and D is the dimension of the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    indices : NxD array_like
        An iterable (numpy array, list of lists, etc) of N points.
        Every index must be of the image dimension size and of any
        integer type (int64, uint16, etc.).

    Returns
    -------
    tuple
        A NxD tuple of tuples with physical points.

    See Also
    --------
        transformContinuousIndexToPhysicalPoint : transform indices to physical points.
        transformPhysicalPointToIndex : transform physical points to indices.
        transformPhysicalPointToContinuousIndex : transform physical points to continuous indices.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    # transform iterable to numpy array
    indices = np.array(indices)

    # correct numpy array in case of single point
    if indices.ndim == 1:
        indices = np.expand_dims(indices, 0)

    # check if type of indices is correct
    if not np.issubdtype(indices.dtype, np.integer):
        raise AttributeError(f"The 'indices' parameter must of any integer type (int64, uint16, etc.).")

    # check if shape of indices is correct
    if indices.ndim != 2 or indices.shape[1] != img.GetDimension():
        raise AttributeError(f"The 'indices' parameter must be an iterable of Nx{img.GetDimension()} shape for {img.GetDimension()}D image.")

    return tuple(map(img.TransformIndexToPhysicalPoint, indices.tolist()))


def transformContinuousIndexToPhysicalPoint(img, indices):
    """Transform indices to physical points.

    The function transforms an iterable of indices into a tuple of
    physical points based on the field of reference (FoR) of an image
    defined as an instance of a SimpleITK image object. The function is
    a wrapper for `TransformContinuousIndexToPhysicalPoint` SimpleITK function, but
    it works for multiple points. The shape of `indices` must be NxD, where N
    is the number of points to be converted, and D is the dimension of the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    indices : NxD array_like
        An iterable (numpy array, list of lists, etc) of N points.
        Every index must be of the image dimension size and can be
        of float or integer type.

    Returns
    -------
    tuple
        A NxD tuple of tuples with physical points.

    See Also
    --------
        transformIndexToPhysicalPoint : transform indices to physical points.
        transformPhysicalPointToIndex : transform physical points to indices.
        transformPhysicalPointToContinuousIndex : transform physical points to continuous indices.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    # transform iterable to numpy array
    indices = np.array(indices)
    # correct numpy array in case of single point
    if indices.ndim == 1:
        indices = np.expand_dims(indices, 0)

    # check if shape of indices is correct
    if indices.ndim != 2 or indices.shape[1] != img.GetDimension():
        raise AttributeError(f"The 'indices' parameter must be an iterable of Nx{img.GetDimension()} shape for {img.GetDimension()}D image.")

    return tuple(map(img.TransformContinuousIndexToPhysicalPoint, indices.tolist()))


def transformPhysicalPointToIndex(img, points):
    """Transform physical points to indices.

    The function transforms an iterable of points into a tuple of
    indices based on the field of reference (FoR) of an image
    defined as an instance of a SimpleITK image object. The function is
    a wrapper for `TransformPhysicalPointToIndex` SimpleITK function, but
    it works for multiple points. The shape of `points` must be NxD, where N
    is the number of points to be converted, and D is the dimension of the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    points : NxD array_like
        An iterable (numpy array, list of lists, etc) of N points.
        Every point must be of the image dimension size.

    Returns
    -------
    tuple
        A NxD tuple of tuples with indices.

    See Also
    --------
        transformIndexToPhysicalPoint : transform indices to physical points.
        transformContinuousIndexToPhysicalPoint : transform indices to physical points.
        transformPhysicalPointToContinuousIndex : transform physical points to continuous indices.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    # transform iterable to numpy array
    points = np.array(points)
    # correct numpy array in case of single point
    if points.ndim == 1:
        points = np.expand_dims(points, 0)

    # check if shape of points is correct
    if points.ndim != 2 or points.shape[1] != img.GetDimension():
        raise AttributeError(f"The 'points' parameter must be an iterable of Nx{img.GetDimension()} shape for {img.GetDimension()}D image.")

    return tuple(map(img.TransformPhysicalPointToIndex, points.tolist()))


def transformPhysicalPointToContinuousIndex(img, points):
    """Transform physical points to continuous indices.

    The function transforms an iterable of points into a tuple of
    continuous indices based on the field of reference (FoR) of an image
    defined as an instance of a SimpleITK image object. The function is
    a wrapper for `TransformPhysicalPointToContinuousIndex` SimpleITK function, but
    it works for multiple points. The shape of `points` must be NxD, where N
    is the number of points to be converted, and D is the dimension of the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    points : NxD array_like
        An iterable (numpy array, list of lists, etc) of N points.
        Every point must be of the image dimension size.

    Returns
    -------
    tuple
        A NxD tuple of tuples with continuous indices.

    See Also
    --------
        transformIndexToPhysicalPoint : transform indices to physical points.
        transformContinuousIndexToPhysicalPoint : transform indices to physical points.
        transformPhysicalPointToIndex : transform physical points to indices.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    # transform iterable to numpy array
    points = np.array(points)
    # correct numpy array in case of single point
    if points.ndim == 1:
        points = np.expand_dims(points, 0)

    # check if shape of points is correct
    if points.ndim != 2 or points.shape[1] != img.GetDimension():
        raise AttributeError(f"The 'points' parameter must be an iterable of Nx{img.GetDimension()} shape for {img.GetDimension()}D image.")

    return tuple(map(img.TransformPhysicalPointToContinuousIndex, points.tolist()))
