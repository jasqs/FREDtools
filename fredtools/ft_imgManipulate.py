def mapStructToImg(img, RSfileName, structName, displayInfo=False):
    """Map structure to image and create a mask.

    The function reads a `structName` structure from RS dicom file and maps it to
    the frame of reference of `img` defined a SimpleITK image object. The function
    exploits the functionality of ``gatetools``. The created mask is an image with
    the same frame of reference (origin. spacing, direction and size) as the `img`.
    The frame of reference of the `img` is not specified, in particular, the Z-spacing
    does not have to be the same as the structure Z-spacing.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    RSfileName : string
        Path String to dicom file with structures (RS file).
    structName : string
        Name of the structure to be mapped.
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image describing a mask (i.e. type 'uint8' with 0/1 values).

    See Also
    --------
        gatetools.region_of_interest: reading structure from RS file.

    Notes
    -----
    The implementation is correct but slow. It is planned to make it
    faster by implementation of multiprocessing or moving the implementation
    to GPU.
    """
    import gatetools as gt
    import numpy as np
    import fredtools as ft
    import pydicom as dicom

    if not ft._isSITK3D(img, raiseError=True):
        raise ValueError(f"The image is a SimpleITK image of dimension {img.GetDimension()}. Only mapping ROI to 3D images are supported now.")

    imgITK = ft.SITK2ITK(img)
    ROI = gt.region_of_interest(ds=dicom.read_file(RSfileName), roi_id=structName)
    imgROIITK = ROI.get_mask(imgITK, corrected=False)
    imgROI = ft.ITK2SITK(imgROIITK)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Structure name: '{:s}'".format(ROI.roiname))
        print("# Structure index: {:s}".format(ROI.roinr))
        print("# Structure volume: {:.3f} cm3".format(ft.arr(imgROI).sum() * np.prod(np.array(imgROI.GetSpacing())) / 1e3))
        ft.ft_imgAnalyse._displayImageInfo(imgROI)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return imgROI


def cropImgToMask(img, imgMask, displayInfo=False):
    """Crop image to mask boundary.

    The function calculates the boundaries of the `imgMask` defined
    as an instance of a SinmpleITK image object describing a mask
    (i.e. type uint8 and only 0/1 values) and crops the `img` defined
    as an instance of a SinmpleITK image object to these boundaries. The boundaries
    mean here the most extreme positions of positive values of the mask in
    each direction. The function exploits SimpleITK.Crop routine.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    imgMask : SimpleITK Image
        Object of a SimpleITK image describing a mask.
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image.

    See Also
    --------
        mapStructToImg: mapping a structure to image to create a mask.
    """
    import numpy as np
    import SimpleITK as sitk
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    ft._isSITK_mask(imgMask, raiseError=True)

    if not ft.compareImgFoR(img, imgMask):
        raise ValueError(f"FoR of the 'img' {img.GetSize()} must be the same as the FoR of the 'imgMask' {imgMask.GetSize()}.")

    labelStatistics = sitk.LabelStatisticsImageFilter()
    labelStatistics.Execute(imgMask, imgMask)
    boundingBox = labelStatistics.GetBoundingBox(1)  # [xmin, xmax, ymin, ymax, zmin, zmax]
    lowerBoundaryCropSize = boundingBox[0::2]
    upperBoundaryCropSize = np.array(img.GetSize()) - np.array(boundingBox[1::2]) - 1
    imgCrop = sitk.Crop(img, lowerBoundaryCropSize=lowerBoundaryCropSize, upperBoundaryCropSize=[int(i) for i in upperBoundaryCropSize])
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(imgCrop)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgCrop


def setValueMask(img, imgMask, value, outside=True, displayInfo=False):
    """Set value inside/outside mask.

    The function sets those the values of the `img` defined as an instance of
    a SinmpleITK object which are inside or outside a mask described by the
    `imgMask` defined as an instance of a SimpleITK object describing a mask
    (i.e. type uint8 and only 0/1 values). The function is a simple wrapper for
    SimpleITK.Mask routine.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    imgMask : SimpleITK Image
        Object of a SimpleITK image describing a mask.
    value : scalar
        value to be set (the type will be mapped to the type of `img`).
    outside : bool, optional
        Determine if the values should be set outside the mask
        (where mask values are equal to 0) or inside the mask
        (where mask values are equal to 1) (def. True meaning
        outside the mask)
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image.

    See Also
    --------
        mapStructToImg: mapping a structure to image to create a mask.
    """
    import SimpleITK as sitk
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    ft._isSITK_mask(imgMask, raiseError=True)

    if not ft.compareImgFoR(img, imgMask):
        raise ValueError(f"FoR of the 'img' {img.GetSize()} must be the same as the FoR of the 'imgMask' {imgMask.GetSize()}.")

    if outside:
        maskingValue = 0
    else:
        maskingValue = 1

    imgMasked = sitk.Mask(img, imgMask, outsideValue=value, maskingValue=maskingValue)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(imgMasked)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return imgMasked


def resampleImg(img, spacing, interpolation="linear", splineOrder=3, displayInfo=False):
    """Resample image to other voxel spacing.

    The function resamples an image defined as an instance of a
    SimpleITK image object to different voxel spacing using
    a specified interpolation method. The assumption is that
    the 'low extent' is not changed, i.e. the coordinates of
    the corner of the first volex preserved. The size of
    the interpolated image is calculated to fit all the voxels' centres
    in the original image extent. The function exploits the
    SimpleITK.Resample routine.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    spacing : array_like
        New spacing in each direction. The length should be the same as the `img` dimension.
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image.
    """
    import SimpleITK as sitk
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    if ft._isSITK_point(img):
        raise ValueError(
            f"The 'img' is an insntance of SimleITK image but describes single point (size of 'img' is {img.GetSize()}). Interpolation cannot be performed on images describing a single point."
        )

    # convert spacing to numpy
    spacing = np.array(spacing)

    # check if point dimension matches the img dim.
    if (spacing.size != img.GetDimension()) and spacing.size != len(ft.ft_imgAnalyse._getAxesNumberNotUnity(img)):
        raise ValueError(f"Shape of 'spacing' is {spacing} but must match the dimension of 'img' {img.GetDimension()} or number of nonunity axes {len(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))}.")
    if spacing.size == len(ft.ft_imgAnalyse._getAxesNumberNotUnity(img)):
        spacingCorr = np.array(img.GetSpacing())
        spacingCorr[np.array(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))] = spacing
    else:
        spacingCorr = spacing

    # set interplator
    interpolator = ft.ft_imgGetSubimg._setSITKInterpolator(img, interpolation=interpolation, splineOrder=splineOrder)

    # correct spacing according to image direction
    spacingCorr = np.dot(ft.ft_imgAnalyse._getDirectionArray(img).T, spacingCorr)

    # calculate new size
    newSize = np.array(ft.getSize(img)) / np.abs(spacingCorr)
    # assure that the centres of the most external voxels are inside the original image extent
    newSize = [np.round(newSizeSingle) if np.mod(newSizeSingle, newSizeSingle.astype("int")) != 0.5 else np.floor(newSizeSingle) for newSizeSingle in newSize]

    # calculate new origin to preserve the same extent
    newOrigin = np.array(ft.getExtent(img))[:, 0] + np.array(spacingCorr) / 2

    # calculate default pixel value as min value of the input image
    """comment: in principle this value is assigned when useNearestNeighborExtrapolator=False and a value is to be 
    interpolated outside the 'img' extent. Such case should not happen because it is assured in the line above that
    the centres of the most external voxels to be interpolated are inside the original image extent. However, the value
    of defaultPixelValue is set to img minimum value, in order to avoind situation that the border voxels have strange values.
    In principle, when a CT image is rescaled, the defaultPixelValue will be -1000 (or -1024) and in case of dose interpolation, 
    the defaultPixelValue will be 0 or any other minimum value in the image."""
    valueOutside = ft.getStatistics(img).GetMinimum()

    # Execute interpolation
    imgRes = sitk.Resample(
        img,
        size=[int(i) for i in newSize],
        outputOrigin=[i for i in newOrigin],
        outputSpacing=[i for i in np.abs(spacingCorr)],
        outputDirection=img.GetDirection(),
        defaultPixelValue=valueOutside,
        interpolator=interpolator,
        useNearestNeighborExtrapolator=False,
    )

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(imgRes)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return imgRes


def sumImg(imgs, displayInfo=False):
    """Sum list of images

    The function sums an iterable (list, tuple, etc.) of images
    defined as instances of a SimpleITK image object. The frame
    of references of all images must be the same.

    Parameters
    ----------
    imgs : iterable
        An iterable (list, tuple, etc.) of SimpleITK image objects.
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image.
    """
    import fredtools as ft

    # check if all images have the same FoR
    for img in imgs:
        ft._isSITK(img, raiseError=True)
        if not ft.compareImgFoR(img, imgs[0]):
            raise ValueError(f"Not all images in the input iterable 'imgs' have the same field of reference.")

    # sum images
    img = sum(imgs)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return img


def createCylindricalMask(img, startPoint, endPoint, dimension, displayInfo=False):
    """Create a cylindrical Mask in image field of reference

    The function creates a cylindrical mask with a given `dimension` and height
    calculated from the starting and ending poits of the cylinder in the frame of
    references of an image defined as SimpleITK image object describing a 3D image.
    Only 3D images are supported. The routine might be helpful for instance for making
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
    dimension : scalar
        Dimension of the cylinder.
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    SimpleITK Image
        Instance of a SimpleITK image object describing a mask (i.e. type 'uint8' with 0/1 values).

    See Also
    --------
        mapStructToImg: mapping a structure to image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
        cropImgToMask : crop an image to mask.
    """
    import SimpleITK as sitk
    import fredtools as ft
    import numpy as np

    def dot(v, w):
        # Dot product
        return sum([vi * wi for vi, wi in zip(v, w)])

    def cross(u, v):
        # Cross product
        s1 = float(u[1]) * v[2] - float(u[2]) * v[1]
        s2 = float(u[2]) * v[0] - float(u[0]) * v[2]
        s3 = float(u[0]) * v[1] - float(u[1]) * v[0]
        return (s1, s2, s3)

    def grid_from_image(image):
        # Similar to numpy.meshgrid using sitk. Grids will be in world (physical) space.
        imsize = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        grid = sitk.PhysicalPointSource(sitk.sitkVectorFloat64, imsize, origin, spacing, direction)

        dim = image.GetDimension()
        grid = [sitk.VectorIndexSelectionCast(grid, i) for i in range(0, dim)]

        for gi in grid:
            gi.CopyInformation(image)
        return grid

    ft._isSITK3D(img, raiseError=True)

    if not isinstance(startPoint, list):
        startPoint = list(startPoint)

    if not isinstance(endPoint, list):
        endPoint = list(endPoint)

    startPoint = np.array(startPoint, dtype="double")
    endPoint = np.array(endPoint, dtype="double")

    heightVector = (np.array(startPoint) - np.array(endPoint)).astype("double")
    height = np.sqrt(heightVector.dot(heightVector))
    radius = dimension / 2

    x, y, z = grid_from_image(img)

    u = (float(startPoint[0]) - x, float(startPoint[1]) - y, float(startPoint[2]) - z)

    dxyz = cross(heightVector, u)

    d = (sitk.Sqrt(sum([dxyzS ** 2 for dxyzS in dxyz])) / height) <= radius
    side1 = dot(heightVector.tolist(), (x - float(startPoint[0]), y - float(startPoint[1]), z - float(startPoint[2]))) <= 0
    side2 = dot(heightVector.tolist(), (x - float(endPoint[0]), y - float(endPoint[1]), z - float(endPoint[2]))) >= 0

    imgMask = d * side1 * side2

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Cylinder height/dimension [mm]: {height:.2f} / {dimension:.2f}")
        print("# Cylinder volume theoretical/real [cm3]: {:.2f} / {:.2f}".format(height * np.pi * radius ** 2 / 1e3, np.prod(imgMask.GetSpacing()) * sitk.GetArrayFromImage(imgMask).sum() / 1e3))
        ft.ft_imgAnalyse._displayImageInfo(imgMask)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return imgMask
