from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def createImg(size: Sequence[int] = [10, 20, 30], components: NonNegativeInt = 0, spacing: Sequence[Numberic] = [1, 1, 1], origin: Sequence[Numberic] = [0.5, 0.5, 0.5], centred: bool = False, fillRandom: bool = False, displayInfo: bool = False) -> SITKImage:
    """Create an empty image with a given size, spacing, and origin.

    The function creates an empty image, i.e. filled with 0 values, 
    with a given size, spacing, and origin. The image can be 2D or 3D.

    Parameters
    ----------
    size : Sequence[int], optional
        The size of the image in each dimension. Must be a sequence of 2 or 3 integers. (def. [10, 20, 30])
    components : NonNegativeInt, optional
        The number of components per pixel. Must be a non-negative integer. (def. 0)
    spacing : Sequence[Numberic], optional
        The spacing between pixels in each dimension. Must be a sequence of numbers. (def. [1, 1, 1])
    origin : Sequence[Numberic], optional
        The origin of the image in each dimension. Must be a sequence of numbers. (def. [0.5, 0.5, 0.5])
    centred : bool, optional
        If True, the origin is centred. (def. False)
    fillRandom : bool, optional
        If True, the image is filled with random Gaussian white noise (mean=10, std=1). (def. False)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SITKImage
        The created image.

    Raises
    ------
    ValueError
        If the size is not 2D or 3D, or if components is a negative integer.
    """
    import SimpleITK as sitk
    import fredtools as ft

    if len(size) < 2 or len(size) > 3:
        error = ValueError(f"Only 2D or 3D images are supported. The parameter size = {list(size)} was used.")
        _logger.error(error)
        raise error

    if components == 0:
        sitkType = sitk.sitkFloat32
    elif components >= 1:
        sitkType = sitk.sitkVectorFloat32
    else:
        error = ValueError(f"The parameter components must be a non-negative integer. The parameter components={components} was used.")
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


def createEllipseMask(img: SITKImage, point: PointLike, radii: Numberic | Sequence[Numberic], displayInfo: bool = False) -> SITKImage:
    """Create an Ellipse mask in the image field of reference.

    The function creates an ellipse mask, defined with the center and radii
    in the frame of references of an image defined as a SimpleITK image 
    object. Any dimension, i.e. 2D-4D, of the image is supported.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    point : array_like
        A point describing the position of the center of the ellipse. The dimension must match the image dimension.
    radii : scalar or array_like
        Radii of the ellipse for each dimension. It might be a scalar, then the same radii will be used in each direction.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object describing a binary mask (i.e. type 'uint8' with 0/1 values).

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
        cropImgToMask : crop an image to mask.
        createCylinderMask: create a cylinder mask.
        createConeMask : create a cone mask.
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


def createConeMask(img: SITKImage, startPoint: PointLike, endPoint: PointLike, startRadius: Numberic, endRadius: Numberic, displayInfo: bool = False) -> SITKImage:
    """Create a cone mask in the image field of reference.

    The function creates a cone mask, defined with starting and ending points and radii 
    in the frame of references of an image defined as a SimpleITK image object describing a 3D image.
    Only 3D images are supported.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK 3D image.
    startPoint : array_like
        3-element point describing the position of the center of the first cone base.
    endPoint : array_like
        3-element point describing the position of the center of the second cone base.
    startRadius : scalar
        Radious of the first cone base.
    endRadius : scalar
        Radious of the second cone base.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object describing a binary mask (i.e. type 'uint8' with 0/1 values).

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
        cropImgToMask : crop an image to mask.
        createCylinderMask: create a cylinder mask.
        createEllipseMask : create an ellipse mask.
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

    # define tube spatial object woth two points
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


def createCylinderMask(img: SITKImage, startPoint: PointLike, endPoint: PointLike, radious: Numberic, displayInfo: bool = False) -> SITKImage:
    """Create a cylindrical Mask in the image field of reference

    The function creates a cylindrical mask with a given radious and height
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
        3-element point describing the position of the center of the first cylinder base.
    endPoint : array_like
        3-element point describing the position of the center of the second cylinder base.
    radious : scalar
        Radious of the cylinder.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object describing a mask (i.e. type 'uint8' with 0/1 values).

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
        cropImgToMask : crop an image to mask.
        createConeMask: create a cone mask.
        createEllipseMask : create an ellipse mask.
    """
    import fredtools as ft
    import numpy as np

    imgMask = ft.createConeMask(img, startPoint, endPoint, radious, radious)

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgMask))

    return imgMask
