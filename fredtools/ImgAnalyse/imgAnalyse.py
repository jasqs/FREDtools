from fredtools._typing import *


def getExtent(img: SITKImage, displayInfo: bool = False) -> tuple[tuple[float, float], ...]:
    """
    Get the extent of an image.

    The function calculates the extent of an image, which is defined as the coordinates of the most side voxels' borders
    in each direction. The extent is assumed to be in millimeters.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        If True, a summary of the function results will be displayed. (default: False)

    Returns
    -------
    tuple
        A tuple of extent values in the form ((xmin, xmax), (ymin, ymax), ...)

    See Also
    --------
    getExtMpl : Get the extent of a SimpleITK image object describing a slice in matplotlib format.
    """

    import numpy as np
    import fredtools as ft
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    cornerLow = img.TransformContinuousIndexToPhysicalPoint(np.zeros(img.GetDimension(), dtype="float64") - 0.5)
    cornerHigh = img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize(), dtype="float64") - 1 + 0.5)

    extent = tuple(zip(cornerLow, cornerHigh))

    if displayInfo:
        axesNames = ["x", "y", "z", "t"]
        strLog = []
        for ext, axisName in zip(extent, axesNames):
            strLog.append(f"{axisName}-spatial extent [mm] = " + ft.ImgAnalyse.imgInfo._generateExtentString(ext))
        logger.info("Image extent:\n" + "\n   ".join(strLog))

    return extent


def getSize(img: SITKImage, displayInfo: bool = False) -> tuple[float, ...]:
    """Get the size of an image.

    This function calculates the size of an image in each direction. The size is defined as the absolute difference of
    the image extent. It is assumed that the coordinate system is in millimeters.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        If True, displays a summary of the function results. (default: False)

    Returns
    -------
    tuple
        A tuple of sizes in each direction in the form (xSize, ySize, ...)

    See Also
    --------
    getExtent : Get the extent of an image.
    """
    import numpy as np
    import fredtools as ft
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    size = tuple(np.abs(np.diff(getExtent(img))).squeeze())

    if displayInfo:
        logger.info("Image size:\n" + ft.ImgAnalyse.imgInfo._generatePositionString(size, "size"))

    return size


def getImageCenter(img: SITKImage, displayInfo: bool = False) -> tuple[float, ...]:
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
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    imgCenter = tuple(np.mean(np.array(getExtent(img)), 1))

    if displayInfo:
        logger.info("Image center:\n" + ft.ImgAnalyse.imgInfo._generatePositionString(imgCenter, "center"))

    return imgCenter


def getMassCenter(img: SITKImage, displayInfo: bool = False) -> tuple[float, ...]:
    """Get the center of mass of an image.

    The function calculates the center of mass of an image defined as
    a SimpleITK image object in each direction. It is assumed that
    the coordinate system is in [mm]. Any NaN values are replaced with zeros.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    tuple
        A tuple of image center of mass coordinates in form (xMassCenter,yMassCenter,...)

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
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    if ft._imgTypeChecker.isSITK_vector(img, raiseError=False):
        logger.error("Cannot compute mass centre of a multicomponent (vector) image.")
        return tuple([np.nan]*img.GetDimension())

    # check if there are nan values in the image and replace them with zeros
    if np.any(np.isnan(sitk.GetArrayViewFromImage(img))):
        logger.debug("Found NaN values in the image. Replacing them with zeros.")
        imgMaskNaN = sitk.GetImageFromArray(np.isnan(sitk.GetArrayViewFromImage(img)).astype("uint8"))
        imgMaskNaN.CopyInformation(img)
        img = sitk.Mask(img, imgMaskNaN, outsideValue=0, maskingValue=1)

    # check if the image is filled with zeros only
    if np.all(sitk.GetArrayViewFromImage(img) == 0):
        logger.debug("Total mass of the image is zero (image filled with zeros). Returning image center as the mass center.")
        massCenter = getImageCenter(img)
    else:
        imgITK = ft.SITK2ITK(img)
        moments = itk.ImageMomentsCalculator.New(imgITK)  # type: ignore
        moments.Compute()
        massCenter = tuple(moments.GetCenterOfGravity())

    if displayInfo:
        logger.info("Image mass center:\n" + ft.ImgAnalyse.imgInfo._generatePositionString(massCenter, "mass center"))

    return massCenter


def getMaxPosition(img: SITKImage, displayInfo: bool = False) -> tuple[float, ...]:
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
    import fredtools as ft
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    maxPosition = ft.getVoxelPhysicalPoints(img == ft.getStatistics(img).GetMaximum(), insideMask=True)

    # check if only one maximum value exists and raise a warning
    if maxPosition.shape[0] != 1:
        logger.warning("More than one maximum value were found. The first one was returned.")

    maxPosition = maxPosition[0]
    maxPosition = tuple(maxPosition)
    # get maximum position
    # maxPosition = np.unravel_index(np.nanargmax(arr), arr.shape[::-1], order="F")
    # maxPosition = img.TransformIndexToPhysicalPoint([int(pos) for pos in maxPosition])

    if displayInfo:
        logger.info("Image maximum position:\n" + ft.ImgAnalyse.imgInfo._generatePositionString(maxPosition, "max position"))

    return maxPosition


def getMinPosition(img: SITKImage, displayInfo: bool = False) -> tuple[float, ...]:
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
    import fredtools as ft
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    minPosition = ft.getVoxelPhysicalPoints(img == ft.getStatistics(img).GetMinimum(), insideMask=True)

    # check if only one maximum value exists and raise a warning
    if minPosition.shape[0] != 1:
        logger.warning("More than one minimum value were found. The first one was returned.")

    minPosition = minPosition[0]
    minPosition = tuple(minPosition)
    # get maximum position
    # minPosition = np.unravel_index(np.nanargmax(arr), arr.shape[::-1], order="F")
    # minPosition = img.TransformIndexToPhysicalPoint([int(pos) for pos in minPosition])

    if displayInfo:
        logger.info("Image maximum position:\n" + ft.ImgAnalyse.imgInfo._generatePositionString(minPosition, "max position"))

    return minPosition


def getVoxelCentres(img: SITKImage, displayInfo: bool = False) -> tuple[tuple[float, ...], ...]:
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
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    voxelCentres = []
    for axis, axisSize in enumerate(img.GetSize()):
        voxelIndices = np.zeros([axisSize, img.GetDimension()], dtype=np.int32)
        voxelIndices[:, axis] = np.arange(axisSize, dtype=np.int32)
        voxelCentresAxis = np.array(ft.transformIndexToPhysicalPoint(img, voxelIndices))
        voxelCentres.append(tuple(voxelCentresAxis[:, axis]))

    if displayInfo:
        axesNames = ["x", "y", "z", "t"]
        strLog = []
        for vox, axisName in zip(voxelCentres, axesNames):
            strLog.append(f"{axisName}-spatial voxel centers [mm] = " + ft.ImgAnalyse.imgInfo._generateSpatialCentresString(vox))
        logger.info("Image voxel centers:\n" + "\n   ".join(strLog))

    return tuple(voxelCentres)


def getVoxelEdges(img: SITKImage, displayInfo: bool = False) -> tuple[tuple[float, ...], ...]:
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

    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    voxelEdges = []
    for axis, axisSize in enumerate(img.GetSize()):
        voxelIndices = np.zeros([axisSize + 1, 3], dtype=np.float64)
        voxelIndices[:, axis] = np.arange(axisSize + 1, dtype=np.float64) - 0.5
        voxelEdgesAxis = np.array(ft.transformContinuousIndexToPhysicalPoint(img, voxelIndices))
        voxelEdges.append(tuple(voxelEdgesAxis[:, axis]))

    if displayInfo:
        axesNames = ["x", "y", "z", "t"]
        strLog = []
        for vox, axisName in zip(voxelEdges, axesNames):
            strLog.append(f"{axisName}-spatial voxel edges [mm] = " + ft.ImgAnalyse.imgInfo._generateSpatialCentresString(vox))
        logger.info("Image voxel edges:\n" + "\n   ".join(strLog))

    return tuple(voxelEdges)


def getVoxelPhysicalPoints(img: SITKImage, insideMask=False, displayInfo: bool = False) -> NDArray:
    """Get physical positions of voxels.

    The function gets voxels' physical positions of an image
    defined as a SimpleITK image object. If the image is a binary mask, 
    then the voxel positions only inside the mask can be requested with
    insideMask parameter, otherwise all voxels' positions will be returned.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    insideMask : bool, optional
        Determine if only the voxels' positions inside the mask shuld be returned. 
        The `img` must describe a binary mask. (def. False)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    NxD numpy.array
        A munpy array of size NxD where N is the number of voxel 
        and D is the axis.

    See Also
    --------
    getVoxelCentres : get voxel centres.

    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # generate voxel positions for all voxels if insideMask == False
    if insideMask == False:
        imgMask = sitk.Cast(img, sitk.sitkUInt8)
        imgMask[:] = 1
    else:
        imgMask = img

    ft._imgTypeChecker.isSITK_maskBinary(imgMask, raiseError=True)

    # get all voxel positions
    PhysicalPointImageSource = sitk.PhysicalPointImageSource()
    PhysicalPointImageSource.SetReferenceImage(imgMask)
    imgMaskPhysPos = PhysicalPointImageSource.Execute()

    # get voxel positions in mask only
    arrMaskPhysPos = sitk.GetArrayViewFromImage(imgMaskPhysPos)
    voxelsIdx = np.where(sitk.GetArrayViewFromImage(imgMask))
    voxelsPos = arrMaskPhysPos[voxelsIdx]

    if displayInfo:
        logger.info(f"Voxel positions returned/all: {voxelsPos.shape[0]}/{np.prod(imgMask.GetSize())}")

    return voxelsPos


def _getAxesVectorNotUnity(img: SITKImage) -> tuple[int, ...]:
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

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    axesVectorNotUnity = [int(axis != 1) for axis in img.GetSize()]

    return tuple(axesVectorNotUnity)


def _getAxesNumberNotUnity(img: SITKImage) -> tuple[int, ...]:
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

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    axesNumberNotUnity = [axis_idx for axis_idx, axis in enumerate(img.GetSize()) if axis != 1]

    return tuple(axesNumberNotUnity)


def _getAxesNumberUnity(img: SITKImage) -> tuple[int, ...]:
    """Get axis indexes for the axes of size equal to one.

    The function calculates the indexes of the axes for which the size is equal to one.

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

    >>> fredtools.ft_imgAnalyse._getAxesNumberUnity(img)
    (1)
    """
    import fredtools as ft

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    axesNumberUnity = [axis_idx for axis_idx, axis in enumerate(img.GetSize()) if axis == 1]

    return tuple(axesNumberUnity)


def _getDirectionArray(img: SITKImage) -> NDArray:
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

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    return np.array(img.GetDirection()).reshape(img.GetDimension(), img.GetDimension())


def _checkIdentity(img: SITKImage) -> bool:
    """Check image identity.

    The function checks if the image direction represents the identity matrix
    (1 in diagonal).

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.

    Returns
    -------(
    bool
        Is it an identity or not.
    """
    import fredtools as ft
    import numpy as np

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    return bool(np.all(np.identity(img.GetDimension(), dtype="int").flatten() == img.GetDirection()))


def getExtMpl(img: SITKImage) -> tuple[float, ...]:
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

    ft._imgTypeChecker.isSITK_slice(img, raiseError=True)

    extent = [list(extent) for axis_idx, extent in enumerate(ft.getExtent(img)) if axis_idx in _getAxesNumberNotUnity(img)]
    extent[1] = extent[1][::-1]
    extent = [inner for outer in extent for inner in outer]

    return tuple(extent)


def pos(img: SITKImage) -> tuple[float, ...] | tuple[tuple[float, ...], ...]:
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

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    pos = ft.getVoxelCentres(img)
    pos = tuple([pos[i] for i in range(img.GetDimension()) if not img.GetSize()[i] == 1])

    if len(pos) == 1:
        pos = pos[0]

    return pos


def arr(img: SITKImage) -> NDArray:
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

    ft._imgTypeChecker.isSITK(img, raiseError=True)
    arr = sitk.GetArrayFromImage(img).squeeze()

    return arr


def vec(img: SITKImage) -> NDArray:
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

    ft._imgTypeChecker.isSITK_profile(img, raiseError=True)
    arr = sitk.GetArrayFromImage(img).squeeze()

    return arr


def isPointInside(img: SITKImage, point: Iterable[float] | Iterable[Iterable[float]], displayInfo: bool = False) -> bool | tuple[bool, ...]:
    """Check if a point or a list of points is inside the extent of an image.

    This function checks if a point or a list of points are inside the extent
    of an image defined as a SimpleITK image object. The points at the border
    of the image (equal to the image extent) are considered to be inside the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    point : Iterable[float] | Iterable[Iterable[float]]
        An iterable (numpy array, list of lists, etc) of N points.
        Every point must be of the image dimension size.
    displayInfo : bool, optional
        Displays a summary of the function results. (default: False)

    Returns
    -------
    bool | tuple[bool, ...]
        A single boolean value if a single point is provided,
        or a tuple of boolean values if a list of points is provided.

    See Also
    --------
    getExtent : Get the extent of the image in each direction.

    Examples
    --------
    Let's assume that the `img` is a 3D image with extent:

    >>> fredtools.getExtent(img)
    ((-175.0, 174.3),
     (-354.6, -70.9),
     (-786.2, -524.6))

    It means that the `img` expands from -175.0 to 174.3, from -354.6 to -70.9,
    and from -786.2 to -524.6 in the X, Y, and Z directions, respectively. Let's check
    if the point [0,0,0] and a list of points [[0,0,0],[0,-100,-600],[-175,-354.6,-786.2]]
    are inside the image extent:

    >>> fredtools.isPointInside(img, [0,0,0])
    False
    >>> fredtools.isPointInside(img, [[0,0,0],[0,-100,-600],[-175,-354.6,-786.2]])
    (False, True, True)
    """
    import numpy as np
    import fredtools as ft
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

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
        if all(isIns):
            logger.info("All points are inside the image.")
        else:
            logger.info("Not all points are inside the image.")

    return isIns[0] if len(isIns) == 1 else tuple(isIns)


def getStatistics(img: SITKImage, displayInfo: bool = False) -> StatisticsImageFilter:
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
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    if ft._imgTypeChecker.isSITK_vector(img):
        logger.warning("The input image is a vector image. Statistics returned for the sum of vectors.")
        img = ft.sumVectorImg(img)

    stat = sitk.StatisticsImageFilter()
    stat.Execute(img)

    if displayInfo:
        strLog = [f"Image mean/std: {stat.GetMean()}/{stat.GetSigma()}",
                  f"Image min/max: {stat.GetMinimum()}/{stat.GetMaximum()}",
                  f"Image sum: {stat.GetSum()}"]
        logger.info("Image statistics:\n" + "\n   ".join(strLog))

    return stat


def compareImgFoR(img1: SITKImage, img2: SITKImage, decimals=3, displayInfo: bool = False) -> bool:
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
    import SimpleITK as sitk
    logger = ft.getLogger()

    ft._imgTypeChecker.isSITK(img1, raiseError=True)
    ft._imgTypeChecker.isSITK(img2, raiseError=True)

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
        valuesMatch = np.allclose(sitk.GetArrayFromImage(img1), sitk.GetArrayFromImage(img2), rtol=0, atol=1 / (10**decimals), equal_nan=True)
    else:
        valuesMatch = False

    if displayInfo:
        strLog = [
            f"Dimension matching:       {dimensionMatch}",
            f"Size matching:            {sizeMatch}",
            f"Origin matching:          {originMatch} ({decimals} decimals tolerance)",
            f"Spacing matching:         {spacingMatch} ({decimals} decimals tolerance)",
            f"Direction matching:       {directionMatch}",
            f"Pixel-to-pixel matching:  {valuesMatch} ({decimals} decimals tolerance)",
        ]
        logger.info("FoRs matching:\n" + "\n   ".join(strLog))

    return bool(match)
