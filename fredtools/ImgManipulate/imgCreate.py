from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def createImg(size: Sequence[int] = [10, 20, 30], components: NonNegativeInt = 0, spacing: Sequence[Numeric] = [1, 1, 1], origin: Sequence[Numeric] = [0.5, 0.5, 0.5], centred: bool = False, fillRandom: bool = False, dtype: DTypeLike = "float32", displayInfo: bool = False) -> SITKImage:
    """Create an empty image with a given size, spacing, and origin.

    The function creates an empty image, i.e. filled with 0 values
    (or Gaussian noise if `fillRandom` is True), with a given size, spacing,
    origin, and data type. The image can be 2D or 3D.

    Parameters
    ----------
    size : Sequence[int], optional
        The size of the image in each dimension. Must be a sequence of 2 or 3 integers. (def. [10, 20, 30])
    components : NonNegativeInt, optional
        The number of components per pixel. Must be a non-negative integer.
        The value 0 creates a scalar image and a value greater than or equal to 1
        creates a vector image with that number of components. (def. 0)
    spacing : Sequence[Numeric], optional
        The spacing between pixels in each dimension. Must be a sequence of numbers. (def. [1, 1, 1])
    origin : Sequence[Numeric], optional
        The origin of the image in each dimension. Must be a sequence of numbers.
        It is ignored when `centred` is True. (def. [0.5, 0.5, 0.5])
    centred : bool, optional
        If True, the origin is centred and the `origin` parameter is ignored. (def. False)
    fillRandom : bool, optional
        If True, the image is filled with random Gaussian white noise (mean=10, std=1).
        For integer data types, the noise has integer values. (def. False)
    dtype : data-type, optional
        The data type of the image voxels (of each component for a vector image)
        in any form accepted by `numpy.dtype`, e.g. 'float32', 'f4', numpy.int16 or float.
        The supported types are (u)int8, (u)int16, (u)int32, (u)int64, float32 and float64.
        The type bool creates a 'uint8' image. (def. 'float32')
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SITKImage
        The created image.

    Raises
    ------
    ValueError
        If the size is not 2D or 3D, if `components` is a negative integer,
        or if `spacing` does not have the same length as `size`.
    TypeError
        If `dtype` is not a supported data type.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
        cropImgToMask : crop an image to mask.
        createCylinderMask : create a cylinder mask.
        createConeMask : create a cone mask.
        createEllipseMask : create an ellipse mask.
        createBoxMask : create a box mask.
        createImgFromFunction : create an image from a function of the voxel centres.
    """
    import SimpleITK as sitk
    import fredtools as ft

    # validate size and components
    if len(size) < 2 or len(size) > 3:
        error = ValueError(f"Only 2D or 3D images are supported. The parameter size = {list(size)} was used.")
        _logger.error(error)
        raise error

    if components < 0:
        error = ValueError(f"The parameter components must be a non-negative integer. The parameter components={components} was used.")
        _logger.error(error)
        raise error

    # validate dtype and get the SimpleITK pixel type
    sitkType = ft._helper.getSITKPixelID(dtype, vector=components >= 1)

    # validate spacing
    if len(spacing) != len(size):
        error = ValueError(f"The parameter spacing must have the same length as the size. The parameter spacing={list(spacing)} was used.")
        _logger.error(error)
        raise error

    img = sitk.Image(list(size), sitkType, components)
    img.SetSpacing(list(spacing))

    if centred:
        img.SetOrigin([- 0.5 * siz * spe + spe * 0.5 for spe, siz in zip(spacing, size)])
    else:
        img.SetOrigin(list(origin))

    if fillRandom:
        img = sitk.AdditiveGaussianNoise(img, standardDeviation=1, mean=10)

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(img))

    return img


def createImgFromFunction(img: SITKImage, function: Callable[..., Any], displayInfo: bool = False) -> SITKImage:
    """Create an image from a function of the voxel centres.

    The function creates a new image with the same frame of reference (size,
    spacing, origin, and direction) and pixel type as `img`, defined as a SimpleITK
    image object, and fills each voxel with the value of `function` evaluated at
    the physical position of the voxel centre, i.e. function(x, y, z) for 3D images
    and function(x, y) for 2D images. Only the frame of reference and the pixel type
    of `img` are used and its values are ignored.

    The positions of the voxel centres are calculated in float64 precision with the
    same formula as in SimpleITK, i.e. origin + direction * spacing * index, therefore,
    they are equal to the positions given by `getVoxelCentres` or by the SimpleITK
    `TransformIndexToPhysicalPoint` function.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK 2D or 3D scalar image.
    function : callable
        A function of 2 or 3 arguments, for 2D or 3D images, respectively. Each argument
        is a float64 numpy array with the voxel centre positions along an axis (x, y, z),
        of the numpy shape of the image, i.e. (nz, ny, nx) for 3D images. The function
        must return an array of the same shape or an array that can be broadcast to it,
        e.g. a scalar, which gives a constant image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image of the same frame of reference
        and pixel type as `img`, filled with the function values.

    Raises
    ------
    TypeError
        If `img` is not an instance of a SimpleITK scalar image object,
        or if `function` is not callable.
    ValueError
        If the image is not 2D or 3D, if the function result cannot be broadcast
        to the image shape, or, for images of an integer type, if the function result
        contains NaN or infinite values or values outside the type range.

    See Also
    --------
        createImg : create an empty image.
        mapStructToImg : mapping a structure to an image to create a mask.
        getVoxelCentres : get voxel centres.

    Notes
    -----
    The function values are calculated in float64 precision and converted to the pixel type
    of `img`. For images of an integer type, the values are truncated towards zero, i.e. the
    same as with `numpy.astype`, and a boolean function result gives 0/1 values.

    Examples
    --------
    Create a 3D image with a dose gradient along the X axis.

    >>> img = fredtools.createImg([60, 60, 60], spacing=[0.5, 0.5, 0.5], centred=True, dtype="float64")
    >>> imgDose = fredtools.createImgFromFunction(img, lambda x, y, z: 1.2 + 0.04 * x)

    Create a binary mask of a sphere with a radius of 5 mm.

    >>> img = fredtools.createImg([60, 60, 60], spacing=[0.5, 0.5, 0.5], centred=True, dtype="uint8")
    >>> imgMask = fredtools.createImgFromFunction(img, lambda x, y, z: x**2 + y**2 + z**2 <= 5**2)
    """
    import numpy as np
    import SimpleITK as sitk
    import fredtools as ft

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # validate image type and dimension
    if ft._imgTypeChecker.isSITK_vector(img) or ("label" in img.GetPixelIDTypeAsString()):
        error = TypeError(f"Only scalar images are supported. The image of type '{img.GetPixelIDTypeAsString()}' was used.")
        _logger.error(error)
        raise error

    if img.GetDimension() not in [2, 3]:
        error = ValueError(f"Only 2D or 3D images are supported. The image of dimension {img.GetDimension()} was used.")
        _logger.error(error)
        raise error

    # validate function
    if not callable(function):
        error = TypeError(f"The parameter function must be callable. The object of type '{type(function)}' was used.")
        _logger.error(error)
        raise error

    # calculate the voxel centre positions in the same way as ITK, i.e. for each axis i:
    # point[i] = origin[i] + sum_j direction[i, j] * spacing[j] * index[j], summed in the same order,
    # so that the positions are the same as from the SimpleITK TransformIndexToPhysicalPoint function
    dim = img.GetDimension()
    indexToPhysicalPoint = np.array(img.GetDirection()).reshape(dim, dim) * np.array(img.GetSpacing())
    voxelIndices = np.indices(img.GetSize()[::-1], dtype=np.float64, sparse=True)[::-1]  # voxelIndices[0] is the X index
    voxelCentres = []
    for axis in range(dim):
        voxelCentresAxis = img.GetOrigin()[axis]
        for indexAxis in range(dim):
            voxelCentresAxis = voxelCentresAxis + indexToPhysicalPoint[axis, indexAxis] * voxelIndices[indexAxis]
        voxelCentres.append(voxelCentresAxis)

    # evaluate the function at the voxel centres
    arr = np.asarray(function(*voxelCentres), dtype=np.float64)
    try:
        arr = np.broadcast_to(arr, voxelCentres[0].shape)
    except ValueError:
        error = ValueError(f"The function result of shape {arr.shape} cannot be broadcast to the image shape {voxelCentres[0].shape} (numpy order, i.e. (nz, ny, nx) for 3D images).")
        _logger.error(error)
        raise error

    # check the function result for images of an integer type, which would be truncated towards zero
    # and wrapped around silently by SimpleITK Cast if out of the type range
    dtypeImg = sitk.GetArrayViewFromImage(img).dtype
    if np.issubdtype(dtypeImg, np.integer):
        if not np.all(np.isfinite(arr)):
            error = ValueError(f"The function result contains NaN or infinite values, which cannot be converted to the image type '{dtypeImg}'.")
            _logger.error(error)
            raise error
        arrTrunc = np.trunc(arr)
        # the upper limit is open, because the maximum of int64 and uint64 cannot be represented in float64
        if (arrTrunc.min() < np.iinfo(dtypeImg).min) or (arrTrunc.max() >= float(np.iinfo(dtypeImg).max) + 1):
            error = ValueError(f"The function result in range [{arr.min()}, {arr.max()}] is out of the range of the image type '{dtypeImg}' [{np.iinfo(dtypeImg).min}, {np.iinfo(dtypeImg).max}].")
            _logger.error(error)
            raise error

    imgOut = sitk.GetImageFromArray(arr)
    imgOut.CopyInformation(img)
    imgOut = sitk.Cast(imgOut, img.GetPixelID())

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgOut))

    return imgOut


def createEllipseMask(img: SITKImage, point: PointLike, radii: Numeric | Sequence[Numeric], displayInfo: bool = False) -> SITKImage:
    """Create an Ellipse mask in the image field of reference.

    The function creates an ellipse mask, defined with the centre and radii
    in the frame of reference of an image defined as a SimpleITK image 
    object. Any dimension, i.e. 2D-4D, of the image is supported.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    point : array_like
        A point describing the position of the centre of the ellipse. The dimension must match the image dimension.
    radii : scalar or sequence (list/tuple)
        Radii of the ellipse for each dimension in [mm] (physical units, not voxels).
        It might be a scalar, then the same radii will be used in each direction.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object describing a binary mask (i.e. type 'uint8' with 0/1 values).

    Raises
    ------
    TypeError
        If `img` is not an instance of a SimpleITK image object, or if
        `radii` is not a scalar or an iterable.
    ValueError
        If the length of `radii` or `point` does not match the image dimension.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
        cropImgToMask : crop an image to mask.
        createCylinderMask : create a cylinder mask.
        createConeMask : create a cone mask.
        createBoxMask : create a box mask.
    """
    import itk
    import fredtools as ft
    from collections.abc import Iterable
    import numpy as np

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # convert image to ITK
    imgITK = ft.SITK2ITK(img)

    # check radii and point parameters
    if isinstance(radii, Sequence):
        radii = list(radii)
    elif np.isscalar(radii):
        radii = list([radii]*img.GetDimension())
    else:
        error = TypeError(f"The `radii` parameter must be a scalar or an iterable. The parameter {radii} was used.")
        _logger.error(error)
        raise error

    if len(radii) != img.GetDimension():
        error = ValueError(f"The `radii` parameter must be an iterable of the same length as the image dimension. Image dimension is {img.GetDimension()} but radii {radii} was used.")
        _logger.error(error)
        raise error
    if len(list(point)) != img.GetDimension():
        error = ValueError(f"The `point` parameter must be an iterable of the same length as the image dimension. Image dimension is {img.GetDimension()} but point {point} was used.")
        _logger.error(error)
        raise error

    # create ellipse and mapping objects
    EllipseSpatialObject = itk.EllipseSpatialObject[img.GetDimension()].New()  # type: ignore
    SpatialObjectToImage = itk.SpatialObjectToImageFilter[itk.SpatialObject[img.GetDimension()], itk.Image[itk.UC, img.GetDimension()]].New()  # type: ignore

    EllipseSpatialObject.SetCenterInObjectSpace(point)
    EllipseSpatialObject.SetRadiusInObjectSpace(radii)

    # map spatial object to image FoR
    SpatialObjectToImage.SetInsideValue(1)
    SpatialObjectToImage.SetOutsideValue(0)
    SpatialObjectToImage.SetInput(EllipseSpatialObject)
    SpatialObjectToImage.SetSize(imgITK.GetLargestPossibleRegion().GetSize())
    SpatialObjectToImage.SetDirection(imgITK.GetDirection())
    SpatialObjectToImage.SetOrigin(imgITK.GetOrigin())
    SpatialObjectToImage.SetSpacing(imgITK.GetSpacing())
    SpatialObjectToImage.Update()
    imgMask = SpatialObjectToImage.GetOutput()

    imgMask = ft.ITK2SITK(imgMask)

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgMask))

    return imgMask


def createConeMask(img: SITKImage, startPoint: PointLike, endPoint: PointLike, startRadius: Numeric, endRadius: Numeric, displayInfo: bool = False) -> SITKImage:
    """Create a cone mask in the image field of reference.

    The function creates a cone mask, defined with starting and ending points and radii
    in the frame of reference of an image defined as a SimpleITK image object describing a 3D image.
    Only 3D images are supported. The cone bases are flat, i.e. the ends of
    the cone are not rounded.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK 3D image.
    startPoint : array_like
        3-element point describing the position of the centre of the first cone base.
    endPoint : array_like
        3-element point describing the position of the centre of the second cone base.
    startRadius : scalar
        Radius of the first cone base in [mm] (physical units, not voxels).
    endRadius : scalar
        Radius of the second cone base in [mm] (physical units, not voxels).
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object describing a binary mask (i.e. type 'uint8' with 0/1 values).

    Raises
    ------
    TypeError
        If `img` is not an instance of a SimpleITK 3D image object.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
        cropImgToMask : crop an image to mask.
        createCylinderMask : create a cylinder mask.
        createEllipseMask : create an ellipse mask.
        createBoxMask : create a box mask.
    """
    import itk
    import fredtools as ft

    ft._imgTypeChecker.isSITK3D(img, raiseError=True)

    # convert image to ITK
    imgITK = ft.SITK2ITK(img)

    if isinstance(startPoint, Sequence):
        startPoint = list(startPoint)
    if isinstance(endPoint, Sequence):
        endPoint = list(endPoint)

    # define tube spatial object with two points
    TubeSpatialObject = itk.TubeSpatialObject[3].New()  # type: ignore

    TubeSpatialObjectPoints = [itk.TubeSpatialObjectPoint[3](),  # type: ignore
                               itk.TubeSpatialObjectPoint[3]()]  # type: ignore

    TubeSpatialObjectPoints[0].SetPositionInObjectSpace(startPoint)
    TubeSpatialObjectPoints[0].SetRadiusInObjectSpace(startRadius)
    TubeSpatialObjectPoints[1].SetPositionInObjectSpace(endPoint)
    TubeSpatialObjectPoints[1].SetRadiusInObjectSpace(endRadius)

    TubeSpatialObject.SetPoints(TubeSpatialObjectPoints)
    TubeSpatialObject.SetEndRounded(False)
    TubeSpatialObject.Update()

    # map spatial object to image FoR
    SpatialObjectToImage = itk.SpatialObjectToImageFilter[itk.SpatialObject[3], itk.Image[itk.UC, 3]].New()  # type: ignore

    SpatialObjectToImage.SetInsideValue(1)
    SpatialObjectToImage.SetOutsideValue(0)
    SpatialObjectToImage.SetInput(TubeSpatialObject)
    SpatialObjectToImage.SetSize(imgITK.GetLargestPossibleRegion().GetSize())
    SpatialObjectToImage.SetDirection(imgITK.GetDirection())
    SpatialObjectToImage.SetOrigin(imgITK.GetOrigin())
    SpatialObjectToImage.SetSpacing(imgITK.GetSpacing())
    SpatialObjectToImage.Update()
    imgMask = SpatialObjectToImage.GetOutput()

    imgMask = ft.ITK2SITK(imgMask)

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgMask))

    return imgMask


def createCylinderMask(img: SITKImage, startPoint: PointLike, endPoint: PointLike, radius: Numeric, displayInfo: bool = False) -> SITKImage:
    """Create a cylindrical Mask in the image field of reference

    The function creates a cylindrical mask with a given radius and height
    calculated from the starting and ending points of the cylinder in the frame of
    references of an image defined as a SimpleITK image object describing a 3D image.
    Only 3D images are supported. For instance, the routine might help make
    a geometrical acceptance correction of a chamber used for Bragg peak measurements.
    The routine was adapted from a GitHub repository: https://github.com/heydude1337/SimplePhantomToolkit/.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK 3D image.
    startPoint : array_like
        3-element point describing the position of the centre of the first cylinder base.
    endPoint : array_like
        3-element point describing the position of the centre of the second cylinder base.
    radius : scalar
        Radius of the cylinder in [mm] (physical units, not voxels).
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object describing a mask (i.e. type 'uint8' with 0/1 values).

    Raises
    ------
    TypeError
        If `img` is not an instance of a SimpleITK 3D image object.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
        cropImgToMask : crop an image to mask.
        createConeMask : create a cone mask.
        createEllipseMask : create an ellipse mask.
        createBoxMask : create a box mask.
    """
    import fredtools as ft
    import numpy as np

    imgMask = ft.createConeMask(img, startPoint, endPoint, radius, radius)

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgMask))

    return imgMask


def createBoxMask(img: SITKImage, point: PointLike, size: Numeric | Sequence[Numeric], displayInfo: bool = False) -> SITKImage:
    """Create a Box mask in the image field of reference.

    The function creates a box mask, defined with the centre point and size
    in the frame of reference of an image defined as a SimpleITK image
    object. Any dimension, i.e. 2D-4D, of the image is supported.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    point : array_like
        A point describing the position of the centre of the box. The dimension must match the image dimension.
    size : scalar or sequence (list/tuple)
        Size of the box for each dimension in [mm] (physical units, not voxels).
        It might be a scalar, then the same size will be used in each direction.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object describing a binary mask (i.e. type 'uint8' with 0/1 values).

    Raises
    ------
    TypeError
        If `img` is not an instance of a SimpleITK image object, or if
        `size` is not a scalar or an iterable.
    ValueError
        If the length of `size` or `point` does not match the image dimension.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
        cropImgToMask : crop an image to mask.
        createCylinderMask : create a cylinder mask.
        createEllipseMask : create an ellipse mask.
        createConeMask : create a cone mask.
    """
    import itk
    import fredtools as ft
    from collections.abc import Iterable
    import numpy as np

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # convert image to ITK
    imgITK = ft.SITK2ITK(img)

    # check size and point parameters
    if isinstance(size, Sequence):
        size = list(size)
    elif np.isscalar(size):
        size = list([size]*img.GetDimension())
    else:
        error = TypeError(f"The `size` parameter must be a scalar or an iterable. The parameter {size} was used.")
        _logger.error(error)
        raise error

    if len(size) != img.GetDimension():
        error = ValueError(f"The `size` parameter must be an iterable of the same length as the image dimension. Image dimension is {img.GetDimension()} but size {size} was used.")
        _logger.error(error)
        raise error
    if len(list(point)) != img.GetDimension():
        error = ValueError(f"The `point` parameter must be an iterable of the same length as the image dimension. Image dimension is {img.GetDimension()} but point {point} was used.")
        _logger.error(error)
        raise error

    # convert centre of box to corner
    point = np.array(point) - 0.5 * np.array(size)

    # create box and mapping objects
    BoxSpatialObject = itk.BoxSpatialObject[img.GetDimension()].New()  # type: ignore
    SpatialObjectToImage = itk.SpatialObjectToImageFilter[itk.SpatialObject[img.GetDimension()], itk.Image[itk.UC, img.GetDimension()]].New()  # type: ignore

    BoxSpatialObject.SetPositionInObjectSpace(point)
    BoxSpatialObject.SetSizeInObjectSpace(size)
    BoxSpatialObject.Update()

    # map spatial object to image FoR
    SpatialObjectToImage.SetInsideValue(1)
    SpatialObjectToImage.SetOutsideValue(0)
    SpatialObjectToImage.SetInput(BoxSpatialObject)
    SpatialObjectToImage.SetSize(imgITK.GetLargestPossibleRegion().GetSize())
    SpatialObjectToImage.SetDirection(imgITK.GetDirection())
    SpatialObjectToImage.SetOrigin(imgITK.GetOrigin())
    SpatialObjectToImage.SetSpacing(imgITK.GetSpacing())
    SpatialObjectToImage.Update()
    imgMask = SpatialObjectToImage.GetOutput()

    imgMask = ft.ITK2SITK(imgMask)

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgMask))

    return imgMask
