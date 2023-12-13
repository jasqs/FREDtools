def mapStructToImg(img, RSfileName, structName, binaryMask=False, areaFraction=0.5, CPUNo="auto", displayInfo=False):
    """Map structure to image and create a mask.

    The function reads a `structName` structure from the RS dicom file and maps it to
    the frame of reference of `img` defined as a SimpleITK image object. The created
    mask is an image with the same frame of reference (origin. spacing, direction
    and size) as the `img` with values larger than 0 for voxels inside the contour
    and values 0 outside. In primary usage, the function produces floating masks, i.e., the value
    of each voxel describes its fractional occupancy by the structure.
    It is assumed that the image is 3D and has a unitary direction, which means that
    the axes describe X, Y and Z directions, respectively. The frame of reference of
    the `img` is not specified, in particular, the Z-spacing does not have to be
    the same as the structure Z-spacing.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK 3D image.
    RSfileName : string
        Path String to dicom file with structures (RS file).
    structName : string
        Name of the structure to be mapped.
    binaryMask : bool, optional
        Determine binary mask production using `areaFraction` parameter. (def. False)
    areaFraction : scalar, optional
        Fraction of pixel area occupancy to calculate binary mask.
        Used only if binaryMask==True. (def. 0.5)
    CPUNo : {'auto', 'none'}, scalar or None, optional
        Define if the multiprocessing should be used and how many cores should
        be exploited. It can be None, and then no multiprocessing will be used,
        a string 'auto', then the number of cores will be determined by os.cpu_count(),
        or a scalar defining the number of CPU cores to be used. (def. 'auto')
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image describing a floating or binary mask.

    See Also
    --------
        cropImgToMask : crop image to mask boundary.
        getDVHMask : calculate DVH for the mask.

    Notes
    -----
    1. the functionality of ``gatetools`` package has been tested, but it turned
    out that it does not consider holes and detached structures. Therefore, a new
    approach has been implemented

    2. The mapping is done for each contour separately and based on the direction (CW or CCW)
    of the contour, it is treated as an inclusive (mask, CW) or exclusive (hole, CCW) contour.
    The mapping of each contour is done in 2D, meaning slice by slice. The resulting image has
    the voxel size and shape the same as the input `img` in X and Y directions. The voxel size
    in the Z direction is calculated based on the contour slice distances, taking into account
    gaps, holes and detached contours. The shape of the image in the Z direction is equal to
    the contour boundings in the Z direction, enlarged by 1 px (``sliceAddNo`` parameter). Such
    image mask is then resampled to the frame of reference of the input `img`. In fact,
    the resampling is applied only to the Z direction, because the frame of reference of X and Y
    directions are the same as the input `img`.
    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk
    import warnings
    from multiprocessing import Pool

    if not ft._isSITK3D(img, raiseError=True):
        raise ValueError(f"The image is a SimpleITK image of dimension {img.GetDimension()}. Only mapping ROI to 3D images are supported now.")

    if not ft.ft_imgIO.dicom_io._isDicomRS(RSfileName):
        raise ValueError(f"The file {RSfileName} is not a proper dicom describing structures.")

    # set the number of CPUs to be used
    CPUNo = ft.getCPUNo(CPUNo)

    # check if fraction area is correct
    if binaryMask:
        if not np.isscalar(areaFraction) or areaFraction <= 0 or areaFraction > 1:
            raise ValueError(f"The parameter 'areaFraction' must be a scalar larger than 0 and less or equal to 1.")

    # check if the structName is in the RS dicom
    if not structName in ft.getRSInfo(RSfileName).ROIName.tolist():
        raise ValueError(f"The structure '{structName}' can not be found in the dicom RS file {RSfileName}")

    # get structure contour and structure info
    StructureContours, StructInfo = ft.dicom_io._getStructureContoursByName(RSfileName, structName)

    # check if the StructureContours is empty and return an empty mask (filled with 0) if true
    if len(StructureContours) == 0:
        # make an empty mask (filled with 0)
        imgMask = sitk.Cast(img, sitk.sitkUInt8)
        imgMask *= 0
        # set additional metadata
        imgMask.SetMetaData("ROIColor", str(StructInfo["Color"]))
        imgMask.SetMetaData("ROIName", StructInfo["Name"])
        imgMask.SetMetaData("ROINumber", str(StructInfo["Number"]))
        imgMask.SetMetaData("ROIGenerationAlgorithm", StructInfo["GenerationAlgorithm"])
        imgMask.SetMetaData("ROIType", StructInfo["Type"])

        if displayInfo:
            print(f"### {ft._currentFuncName()} ###")
            print("# Warning: no 'StructureContours' was defined for this structure and an empty mask was returned")
            print("# Structure name (type): '{:s}' ({:s})".format(StructInfo["Name"], StructInfo["Type"]))
            print("# Structure volume: {:.3f} cm3".format(ft.arr(imgMask).sum() * np.prod(np.array(imgMask.GetSpacing())) / 1e3))
            ft.ft_imgAnalyse._displayImageInfo(imgMask)
            print("#" * len(f"### {ft._currentFuncName()} ###"))

        return imgMask

    # add the first point at the end of the contour for each contour (to make sure that the contour is closed)
    StructureContours = [np.append(StructureContour, np.expand_dims(StructureContour[0], axis=0), axis=0) for StructureContour in StructureContours]

    # check if all Z positions are the same for each contour
    for StructureContour in StructureContours:
        if not len(np.unique(StructureContour[:, 2])) == 1:
            raise ValueError(f"Not all Z (depth) position in controur are the same.")

    # get depth for each contour and sort the depths
    StructureContoursDepths = np.array([StructureContour[0, 2] for StructureContour in StructureContours])

    # get contour spacing in Z direction as the minimum spacing between individual contours excluding 0.
    """
    note: spacing 0 means that holes or detached contours exist in the structure
    note: more than single spacing (excluding 0) means that a gap exists in the structure
    """
    StructureContoursSpacing = np.unique(np.round(np.diff(np.sort(StructureContoursDepths)), decimals=3))
    StructureContoursSpacing = StructureContoursSpacing[StructureContoursSpacing != 0].min()

    # get CW (True) or CCW (False) direction for each contour
    StructureContoursDirection = [ft.ft_imgIO.dicom_io._checkContourCWDirection(StructureContour) for StructureContour in StructureContours]

    # get structure bounding box in Z
    """
    note: this is not extent because it takes the positions of vertices and not the positions of voxels' borders
    """
    StructureContoursBBoxZ = (StructureContoursDepths.min(), StructureContoursDepths.max())

    # calculate the mask depths
    # it is larger in Z direction of +/- StructureContoursSpacing
    MaskDepths = np.round(np.arange(StructureContoursBBoxZ[0] - StructureContoursSpacing, StructureContoursBBoxZ[1] + StructureContoursSpacing, StructureContoursSpacing), 3)

    # verify if all contour depths are present in the mask depths
    if not set(StructureContoursDepths).issubset(MaskDepths):
        raise RuntimeError(f"Not all depths defined in coutour are represented in the created mask.")

    # validate if all contour vertices are defined inside the image extent
    StructureContoursExtent = np.stack((np.min(np.concatenate(StructureContours, axis=0), axis=0), np.max(np.concatenate(StructureContours, axis=0), axis=0)))
    if not all(ft.isPointInside(img, StructureContoursExtent)):
        warnings.warn(f"Warning: Some vertices of the structure '{structName}' are defined outside the image extent.\n" +
                      f"The image extent is {ft.getExtent(img)}\n" +
                      f"and the contour extent is {StructureContoursExtent.T.tolist()}.")

    # calculate mask size
    arrMaskSize = np.array([len(MaskDepths), img.GetSize()[1], img.GetSize()[0]])

    # prepare an empty mask
    arrMask = np.zeros(arrMaskSize, dtype="float")
    imgMask = sitk.GetImageFromArray(arrMask)
    imgMask.SetOrigin([img.GetOrigin()[0], img.GetOrigin()[1], MaskDepths.min()])
    imgMask.SetSpacing([img.GetSpacing()[0], img.GetSpacing()[1], StructureContoursSpacing])

    # convert all structure contour coordinates from the real world to pixel
    StructureContoursPx = []
    for StructureContour in StructureContours:
        StructureContoursPx.append(np.array(ft.transformPhysicalPointToContinuousIndex(imgMask, StructureContour)))

    # map all structure contours to structure arrays using multiprocessing
    """
    The 2D arrays are of various sizes depending on the structure contour bounding box. 
    The 2D arrays are in floating numbers in the range 0-1, showing the voxel occupancy by the structure contour.
    """
    with Pool(CPUNo) as p:
        StructureContoursArrays = p.map(_getStructureContourArray, StructureContoursPx)

    # correct each structure contour array for the size
    for StructureContoursArrayIdx in range(len(StructureContoursArrays)):
        StructureContoursArray = StructureContoursArrays[StructureContoursArrayIdx]
        StructureContoursArray = StructureContoursArray[0: arrMaskSize[1], 0: arrMaskSize[2]]  # clip array to given shape if needed
        StructureContoursArray = np.pad(StructureContoursArray, ((0, arrMaskSize[1] - StructureContoursArray.shape[0]), (0, arrMaskSize[2] - StructureContoursArray.shape[1])))  # pad array to given shape if needed
        StructureContoursArrays[StructureContoursArrayIdx] = StructureContoursArray

    # merge the structure contours arrays into a single mask
    """
    The 3D array is in floating numbers in the range 0-1, showing the voxel occupancy by the structure contour.
    The positive-direction contours are added, and the negative-direction contours are subtracted. The routine proceeds 
    in the order of appearance of the structure contour in the structure RS dicom. 
    """
    for MaskDepth_idx, MaskDepth in enumerate(MaskDepths):
        # get list of indices of StructureContours for MaskDepth
        StructureContoursIdx = np.where(StructureContoursDepths == MaskDepth)[0]

        # skip calculation of the mask if no structure exists for MaskDepth
        if StructureContoursIdx.size == 0:
            continue

        for StructureContourIdx in StructureContoursIdx:
            if StructureContoursDirection[StructureContourIdx]:
                arrMask[MaskDepth_idx, :, :] += StructureContoursArrays[StructureContourIdx]
            else:
                arrMask[MaskDepth_idx, :, :] -= StructureContoursArrays[StructureContourIdx]

    # prepare SimpleITK mask
    imgMask = sitk.GetImageFromArray(arrMask)
    imgMask.SetOrigin(
        [img.GetOrigin()[0], img.GetOrigin()[1], MaskDepths.min()])
    imgMask.SetSpacing([img.GetSpacing()[0], img.GetSpacing()[
                       1], StructureContoursSpacing])

    # interpolate mask to input image
    imgMask = sitk.Resample(
        imgMask, img, interpolator=ft.ft_imgGetSubimg._setSITKInterpolator(interpolation="linear"))

    # prepare binary mask if requested, with fraction area threshold
    if binaryMask:
        try:
            imgMask = ft.floatingToBinaryMask(imgMask, threshold=areaFraction)
        except TypeError:
            raise RuntimeError("The structure was mapped but the binary mask is incorrect.")
    else:
        imgMask = sitk.Cast(imgMask, sitk.sitkFloat64)
        # make sure the image is a floating mask
        try:
            ft._isSITK_maskFloating(imgMask, raiseError=True)
        except TypeError:
            raise RuntimeError("The structure was mapped but the floating mask is incorrect.")

    # set additional metadata
    imgMask.SetMetaData("ROIColor", str(StructInfo["Color"]))
    imgMask.SetMetaData("ROIName", StructInfo["Name"])
    imgMask.SetMetaData("ROINumber", str(StructInfo["Number"]))
    imgMask.SetMetaData("ROIGenerationAlgorithm",
                        StructInfo["GenerationAlgorithm"])
    imgMask.SetMetaData("ROIType", StructInfo["Type"])

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Structure name (type): '{:s}' ({:s})".format(StructInfo["Name"], StructInfo["Type"]))
        print("# Structure volume: {:.3f} cm3".format(ft.arr(imgMask).sum() * np.prod(np.array(imgMask.GetSpacing())) / 1e3))
        ft.ft_imgAnalyse._displayImageInfo(imgMask)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgMask


def _getLineSegmentPixels(lineSegmentStart, lineSegmentEnd):
    """
    Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line segment, given two points lineSegmentStart and lineSegmentEnd.
    See also: http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
    """
    import numpy as np

    lineSegment = np.stack((lineSegmentStart, lineSegmentEnd))
    dxy = np.diff(lineSegment.T)[:, 0]
    steep = np.abs(dxy[0]) < np.abs(dxy[1])

    if steep:
        lineSegment = lineSegment[:, ::-1]
        dxy = dxy[::-1]
    if not (lineSegment[1, 0] - lineSegment[0, 0]) > 0:
        lineSegment = lineSegment[::-1, :]

    if dxy[0] == 0:  # the dxy[0]==0 when the starting and ending points of a segment line are the same
        gradient = 0
    else:
        gradient = dxy[1] / dxy[0]

    # handle the first and the last endpoint
    xyend = np.array((np.round(lineSegment[0, 0]), lineSegment[0, 1] + gradient * (np.round(lineSegment[0, 0]) - lineSegment[0, 0])))
    xends = np.round(lineSegment[:, 0])
    yends = lineSegment[:, 1] + gradient * (xends - lineSegment[:, 0])
    xycoordsEnds = np.array(((xends[0], yends[0]), (xends[1], yends[1]), (xends[0], yends[0] + 1), (xends[1], yends[1] + 1)), dtype=int)

    # main loop
    xycoordsInside = np.empty((0, 2), dtype=int)

    intery = xyend[1] + gradient
    for px in range(xycoordsEnds[0, 0] + 1, xycoordsEnds[1, 0]):
        xycoordsInside = np.append(xycoordsInside, np.array([[px, int(intery)]]), axis=0)
        xycoordsInside = np.append(xycoordsInside, np.array([[px, int(intery) + 1]]), axis=0)
        intery = intery + gradient

    # merge inside and start/end pixels
    xycoords = np.concatenate((xycoordsEnds, xycoordsInside))

    if steep:
        xycoords = xycoords[:, ::-1]

    return xycoords


def _getStructureContourBorderPixels(StructureContourPx):
    """
    Calculates all pixels along a polygon.
    """
    import numpy as np

    StructureContourBorderPixels = np.empty((0, 2), dtype=int)
    for lineSegmentIdx in range(len(StructureContourPx) - 1):
        StructureContourBorderPixels = np.append(StructureContourBorderPixels, _getLineSegmentPixels(StructureContourPx[lineSegmentIdx], StructureContourPx[lineSegmentIdx + 1]), axis=0)

    # remove duplicated pixels
    StructureContourBorderPixels = np.unique(StructureContourBorderPixels, axis=0)
    return StructureContourBorderPixels


def _getStructureContourArray(StructureContourPx):
    """
    Calculates array with unit values inside the structure polygon and
    floating values in the range 0-1 at the contour border, describing
    the voxel occupancy fraction.
    """
    import shapely as sph
    import scipy.ndimage as ndimage
    import numpy as np

    # get a list of pixels at the contour border
    StructureContourBorderPixels = _getStructureContourBorderPixels(StructureContourPx[:, 0:2])

    # convert the contour and contour pixels to polygons
    StructureContourBorderPixelsPolygons = [sph.Polygon(StructureContourBorderPixel + np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])) for StructureContourBorderPixel in StructureContourBorderPixels]
    StructureContourPxPolygon = sph.Polygon(StructureContourPx[:, 0:2])

    # remove remaining pixels at the contour border that do not intersect with the contour
    StructureContourBorderPixelsPolygonsIntersects = [StructureContourBorderPixelsPlygon.intersects(StructureContourPxPolygon.boundary) for StructureContourBorderPixelsPlygon in StructureContourBorderPixelsPolygons]
    StructureContourBorderPixels = StructureContourBorderPixels[StructureContourBorderPixelsPolygonsIntersects]
    StructureContourBorderPixelsPolygons = np.array(StructureContourBorderPixelsPolygons)[StructureContourBorderPixelsPolygonsIntersects].tolist()

    # generate contour slice
    arr = np.zeros(StructureContourBorderPixels.max(axis=0)[::-1] + 1)

    # set pixels at the contour border to 1
    arr[StructureContourBorderPixels[:, 1], StructureContourBorderPixels[:, 0]] = 1

    # fill pixels inside the contour
    arr = ndimage.binary_fill_holes(arr.astype("bool")).astype("float")

    # calculate the contour pixels area inside the contour
    StructureContourBorderPixelsArea = [StructureContourBorderPixelsPolygon.intersection(StructureContourPxPolygon).area for StructureContourBorderPixelsPolygon in StructureContourBorderPixelsPolygons]

    # set contour pixels with the area inside the contour
    arr[StructureContourBorderPixels[:, 1], StructureContourBorderPixels[:, 0]] = StructureContourBorderPixelsArea

    return arr


def floatingToBinaryMask(imgMask, threshold=0.5, displayInfo=False):
    """Convert floating mask to binary mask.

    The function converts an image defined as an instance of a SimpleITK
    image object describing a floating mask to a binary mask image, based on
    a given threshold.

    Parameters
    ----------
    imgMask : SimpleITK Image
        An object of a SimpleITK image describing a binary mask.
    threshold : scalar, optional
        The threshold to calculate the binary mask. (def. 0.5)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
    """
    import fredtools as ft
    import SimpleITK as sitk
    import numpy as np

    ft._isSITK_maskFloating(imgMask, raiseError=True)

    # check if the threshold is correct
    if not np.isscalar(threshold) or threshold <= 0 or threshold > 1:
        raise ValueError(f"The parameter 'threshold' must be a scalar larger than 0 and less or equal to 1.")

    imgMaskBinary = sitk.BinaryThreshold(imgMask, lowerThreshold=threshold, upperThreshold=ft.getStatistics(imgMask).GetMaximum() + 1, insideValue=1, outsideValue=0)
    imgMaskBinary = sitk.Cast(imgMaskBinary, sitk.sitkUInt8)
    imgMaskBinary = ft._copyImgMetaData(imgMask, imgMaskBinary)

    # make sure the image is a binary mask
    try:
        ft._isSITK_maskBinary(imgMaskBinary, raiseError=True)
    except TypeError:
        raise RuntimeError("The the input image is a floating mask but the binary mask is incorrect.")

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(imgMaskBinary)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgMaskBinary


def cropImgToMask(img, imgMask, displayInfo=False):
    """Crop image to mask boundary.

    The function calculates the boundaries of the `imgMask` defined
    as an instance of a SimpleITK image object describing a binary or floating mask
    and crops the `img` defined as an instance of a SimpleITK image object to
    these boundaries. The boundaries mean here the most extreme positions of
    positive values (1 for binary mask and above 0 for floating mask) of
    the mask in each direction. The function exploits the SimpleITK.Crop routine.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    imgMask : SimpleITK Image
        An object of a SimpleITK image describing a mask.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
    """
    import numpy as np
    import SimpleITK as sitk
    import fredtools as ft
    import copy

    ft._isSITK(img, raiseError=True)
    ft._isSITK_mask(imgMask, raiseError=True)

    # check FoR of the image and the mask
    if not ft.compareImgFoR(img, imgMask):
        raise ValueError(f"FoR of the 'img' {img.GetSize()} must be the same as the FoR of the 'imgMask' {imgMask.GetSize()}.")

    # convert the floating mask to binary with a minimum threshold larger than 0
    if ft._isSITK_maskFloating(imgMask):
        imgMaskNonZero = copy.copy(imgMask)
        imgMaskNonZero[imgMaskNonZero == 0] = np.nan
        imgMask = ft.floatingToBinaryMask(imgMask, threshold=ft.getStatistics(imgMaskNonZero).GetMinimum())

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

    The function sets the values of the `img` defined as an instance of
    a SimpleITK object that are inside or outside a binary or floating mask
    described by the `imgMask`, defined as an instance of a SimpleITK object
    describing a mask. The inside of the mask is defined for voxels with 1 for the binary mask
    and above 0 for the floating mask. The function is a simple wrapper for the SimpleITK.Mask routine.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    imgMask : SimpleITK Image
        An object of a SimpleITK image describing a mask.
    value : scalar
        value to be set (the type will be mapped to the type of `img`).
    outside : bool, optional
        Determine if the values should be set outside the mask
        (where mask values are equal to 0) or inside the mask
        (where mask values are above 0) (def. True meaning
        outside the mask)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
    """
    import SimpleITK as sitk
    import fredtools as ft
    import copy
    import numpy as np

    ft._isSITK(img, raiseError=True)
    ft._isSITK_mask(imgMask, raiseError=True)

    # check FoR of the image and the mask
    if not ft.compareImgFoR(img, imgMask):
        raise ValueError(f"FoR of the 'img' {img.GetSize()} must be the same as the FoR of the 'imgMask' {imgMask.GetSize()}.")

    # convert the floating mask to binary with a minimum threshold larger than 0
    if ft._isSITK_maskFloating(imgMask):
        imgMaskNonZero = copy.copy(imgMask)
        imgMaskNonZero[imgMaskNonZero == 0] = np.nan
        imgMask = ft.floatingToBinaryMask(imgMask, threshold=ft.getStatistics(imgMaskNonZero).GetMinimum())

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
    the corner of the first voxel are preserved. The size of
    the interpolated image is calculated to fit all the voxels' centers
    in the original image extent. The function exploits the
    SimpleITK.Resample routine.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    spacing : array_like
        New spacing in each direction. The length should be the same as the `img` dimension.
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in the range 0-5. (def. 3)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image.
    """
    import SimpleITK as sitk
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    if ft._isSITK_point(img):
        raise ValueError(
            f"The 'img' is an instance of SimleITK image but describes a single point (size of 'img' is {img.GetSize()}). Interpolation cannot be performed on images describing a single point."
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

    # set interpolator
    interpolator = ft.ft_imgGetSubimg._setSITKInterpolator(interpolation=interpolation, splineOrder=splineOrder)

    # correct spacing according to image direction
    spacingCorr = np.dot(ft.ft_imgAnalyse._getDirectionArray(img).T, spacingCorr)

    # calculate new size
    newSize = np.array(ft.getSize(img)) / np.abs(spacingCorr)
    # assure that the centres of the most external voxels are inside the original image extent
    newSize = [np.round(newSizeSingle) if np.mod(newSizeSingle, newSizeSingle.astype("int")) != 0.5 else np.floor(newSizeSingle) for newSizeSingle in newSize]

    # calculate new origin to preserve the same extent
    newOrigin = np.array(ft.getExtent(img))[:, 0] + np.array(spacingCorr) / 2

    # calculate default pixel value as min value of the input image
    """comment: In principle, this value is assigned when using NearestNeighborExtrapolator=False, and a value is to be 
    interpolated outside the 'img' extent. Such a case should not happen because it is assured in the line above that
    the centers of the most external voxels to be interpolated are inside the original image extent. However, the value
    of defaultPixelValue is set to the img minimum value, to avoid the situation that the border voxels have strange values.
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
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image.
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
    """Create a cylindrical Mask in the image field of reference

    The function creates a cylindrical mask with a given `dimension` and height
    calculated from the starting and ending points of the cylinder in the frame of
    references of an image defined as a SimpleITK image object describing a 3D image.
    Only 3D images are supported. The routine might be helpful for instance for making
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
    dimension : scalar
        Dimension of the cylinder.
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

    d = (sitk.Sqrt(sum([dxyzS**2 for dxyzS in dxyz])) / height) <= radius
    side1 = dot(heightVector.tolist(), (x - float(startPoint[0]), y - float(startPoint[1]), z - float(startPoint[2]))) <= 0
    side2 = dot(heightVector.tolist(), (x - float(endPoint[0]), y - float(endPoint[1]), z - float(endPoint[2]))) >= 0

    imgMask = d * side1 * side2

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Cylinder height/dimension [mm]: {height:.2f} / {dimension:.2f}")
        print("# Cylinder volume theoretical/real [cm3]: {:.2f} / {:.2f}".format(height * np.pi * radius**2 / 1e3, np.prod(imgMask.GetSpacing()) * sitk.GetArrayFromImage(imgMask).sum() / 1e3))
        ft.ft_imgAnalyse._displayImageInfo(imgMask)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return imgMask


def sumVectorImg(img, displayInfo=False):
    """Sum vector image.

    The function sums all elements of a vector in a vector image
    defined as instances of a SimpleITK vector image object.
    The resulting image has the same frame of reference but
    is a scalar image.

    Parameters
    ----------
    img : SimpleITK Vector Image
        An object of a SimpleITK vector image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image.
    """
    import fredtools as ft
    import SimpleITK as sitk

    ft._isSITK_vector(img, raiseError=True)

    imgs = []
    for componentIdx in range(img.GetNumberOfComponentsPerPixel()):
        imgs.append(sitk.VectorIndexSelectionCast(img, componentIdx))

    imgSum = ft.sumImg(imgs)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(imgSum)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return imgSum


def _getFORTransformed(img, transform):
    """Calculate image FOR after transformation.

    The function calculates a new Field of Reference (FOR) for an image defined
    as an instance of a SimpleITK image object. The purpose of this calculation is
    that transformed images can be 'cropped'. This function calculates new FOR based
    on the positions of the transformed image corners.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    transform : SimpleITK Transform
        An object of a SimpleITK transform.

    Returns
    -------
    size, origin, spacing, direction
        Calculated image size, origin, and spacing (the same as the original)
        and direction (identity).

    Notes
    -----
    The implementation was adapted from the idea presented in
    https://discourse.itk.org/t/dont-lose-data-with-rotation/3325/2
    """
    from itertools import product
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    ft._isSITK_transform(transform, raiseError=True)

    startPX = [0, 0, 0]
    endPX = [int(size) for size in np.array(img.GetSize()) - 1]
    cornersPX = list(product(*zip(startPX, endPX)))
    cornersRW = [img.TransformIndexToPhysicalPoint(cornerPX) for cornerPX in cornersPX]
    cornersRWTransformed = [transform.TransformPoint(cornerRW) for cornerRW in cornersRW]
    sizeRW = np.max(cornersRWTransformed, 0) - np.min(cornersRWTransformed, 0)
    originRW = np.min(cornersRWTransformed, 0)
    sizePX = [int(size) for size in np.ceil((sizeRW + 1) / img.GetSpacing())]
    spacingRW = img.GetSpacing()
    directionIdentity = np.identity(img.GetDimension()).flatten().tolist()

    return sizePX, originRW, spacingRW, directionIdentity


def getImgBEV(img, isocentrePosition, gantryAngle, couchAngle, defaultPixelValue="auto", interpolation="linear", splineOrder=3, displayInfo=False):
    """Transform an image to Beam's Eye View (BEV).

    The function transforms a 3D image defined as a SimpleITK 3D image object to
    the Beam's Eye View (BEV) based on the given isocentre position,
    gantry angle and couch rotation, using a defined interpolation method.
    The BEV Field of Reference (FOR) means that the Z+ direction is along the field
    (along the beam of relative position [0,0]) and X/Y positions are consistent with
    the DICOM and FRED Monte Carlo definitions.

    Parameters
    ----------
    img : SimpleITK 3D Image
        Object of a SimpleITK 3D image.
    isocentrePosition : array_like, (3x1)
        Position of the isocentre with respect to the `img` FOR.
    gantryAngle : scalar
        Rotation of the gantry around the isocentre position in [deg].
    couchAngle : scalar
        Rotation of the couch around the isocentre position in [deg].
    defaultPixelValue : 'auto' or scalar, optional
        The value to fill the voxels with, outside the original `img`.
        If 'auto', then the value will be calculated automatically as the
        minimum value of the `img`. (def. 'auto')
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in the range 0-5. (def. 3)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK 3D Image
        An object of a transformed SimpleITK 3D image.

    Notes
    -----
    The basic workflow follows:

        1. translate the image to the isocentre as to have the isocentre at zero position,
        2. rotate the couch around the isocentre,
        3. rotate the gantry around the isocentre,
        4. rotate and flip the image to get BEV.

    Note that the isocentre of the transformed image is at the zero point.

    Note that the isocentre defined in the delivery sequence of the FRED rtplan is
    a negative isocentre defined in the DICOM RN plan, the couch rotation defined
    in the delivery sequence of the FRED rtplan is a negative couch rotation defined
    in the DICOM RN plan, but the gantry rotation defined in the delivery sequence
    of the FRED rtplan is equal to the gantry rotation of the DICOM RN plan.
    """
    import numpy as np
    import SimpleITK as sitk
    import fredtools as ft

    ft._isSITK3D(img, raiseError=True)

    # set interpolator
    interpolator = ft.ft_imgGetSubimg._setSITKInterpolator(interpolation=interpolation, splineOrder=splineOrder)

    # check if isocentrePosition dimension matches the img dim.
    if len(isocentrePosition) != img.GetDimension():
        raise ValueError(f"Dimension of 'isocentrePosition' {isocentrePosition} does not match 'img' dimension {img.GetDimension()}.")

    # determine default pixel value
    if isinstance(defaultPixelValue, str) and defaultPixelValue.lower() == "auto":
        defaultPixelValue = ft.getStatistics(img).GetMinimum()
    elif not np.isscalar(defaultPixelValue):
        raise ValueError(f"The parameter 'defaultPixelValue' must be a scalar or 'auto'")

    # define translation to isocentre (to have isocentre at the zero position)
    translationTransformIsocentre = sitk.TranslationTransform(img.GetDimension())
    translationTransformIsocentre.SetOffset(isocentrePosition)

    # define gantry rotation around the zero position
    eulerTransformGantry = sitk.Euler3DTransform()
    eulerTransformGantry.SetRotation(angleX=0, angleY=0, angleZ=np.deg2rad(gantryAngle))

    # define couch rotation around the zero position
    eulerTransformCouch = sitk.Euler3DTransform()
    eulerTransformCouch.SetRotation(angleX=0, angleY=np.deg2rad(couchAngle), angleZ=0)

    # define rotation and flipping to get BEV (Z+ along the field) and to be consistent with FRED coordinate system of PB
    rotateBEVTransform = sitk.Euler3DTransform()
    rotateBEVTransform.SetRotation(angleX=np.deg2rad(-90), angleY=0, angleZ=0)
    flipXYTransform = sitk.ScaleTransform(img.GetDimension(), (1, -1, 1))
    compositTransformBEV = sitk.CompositeTransform(img.GetDimension())
    compositTransformBEV.AddTransform(rotateBEVTransform)
    compositTransformBEV.AddTransform(flipXYTransform)

    # define composite transform
    compositTransform = sitk.CompositeTransform(img.GetDimension())
    compositTransform.AddTransform(translationTransformIsocentre)
    compositTransform.AddTransform(eulerTransformCouch)
    compositTransform.AddTransform(eulerTransformGantry)
    compositTransform.AddTransform(compositTransformBEV)

    # calculate new FOR for a new, not cropped image
    size, origin, spacing, direction = ft.ft_imgManipulate._getFORTransformed(img, compositTransform.GetInverse())

    # transform with resampling
    imgBEV = sitk.Resample(
        img, transform=compositTransform, size=size, outputOrigin=origin, outputSpacing=spacing, interpolator=interpolator, outputDirection=direction, defaultPixelValue=defaultPixelValue  # type: ignore
    )

    # copy metadata if they exist
    imgBEV = ft._copyImgMetaData(img, imgBEV)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Isocentre position [mm]: ", np.array(isocentrePosition))
        print("# Gantry angle [deg]: {:.1f}".format(gantryAngle))
        print("# Couch angle [deg]: {:.1f}".format(couchAngle))
        ft.ft_imgAnalyse._displayImageInfo(imgBEV)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgBEV


def overwriteCTPhysicalProperties(
    img,
    RSfileName,
    areaFraction=0.5,
    CPUNo="auto",
    relElecDensCalib=[
        [-1024, -1000, -777.82, -495.34, -64.96, -34.39, -3.87, 51.92, 56.99, 226.05, 857.65, 1313, 8513, 12668, 25332],
        [0, 0, 0.190, 0.489, 0.949, 0.976, 1, 1.043, 1.053, 1.117, 1.456, 1.696, 3.76, 6.58, 9.09],
    ],
    HUrange=[-2000, 50000],
    displayInfo=False,
):
    """Overwrite HU values in a CT image based on structures' physical properties.

    The function searches in a structure RS dicom file for structures with
    the physical property defined, maps each structure to the CT image
    defined as an instance of a SimpleITK 3D image, and replaces the Hounsfield Units (HU)
    values for voxels inside the structure. Only the relative electronic density physical
    property ('REL_ELEC_DENSITY') is implemented now, and it is converted to a HU value
    based on relative electronic density to HU calibration, given as `relElecDensCalib`
    parameter, whereas the missing values are interpolated linearly and rounded to
    the nearest integer HU value.

    Parameters
    ----------
    img : SimpleITK 3D Image
        Object of a SimpleITK 3D image.
    RSfileName : string
        Path String to dicom file with structures (RS file).
    areaFraction : scalar, optional
        Fraction of pixel area occupancy to calculate binary mask. See `mapStructToImg` function for more information. (def. 0.5)
    CPUNo : {'auto', 'none'}, scalar or None, optional
        Define if the multiprocessing should be used and how many cores should
        be exploited. See `mapStructToImg` function for more information. (def. 'auto')
    relElecDensCalib : array_like, optional
        2xN iterable (e.g. 2xN numpy array or list of two equal size lists) describing
        the calibration between HU values and relative electronic density. The first element (column)
        is describing the HU values and the second the relative electronic density. The missing values
        are interpolated linearly and if the user would like to use a different interpolation
        like spline or polynomial, it is advised to provide it explicitly for each HU value.
        The structures with the relative electronic density outside the calibration range will be skipped
        and a warning will be displayed. (def. [[-1024, -1000, -777.82, -495.34, -64.96, -34.39, -3.87, 51.92, 56.99, 226.05, 857.65, 1313, 8513, 12668, 25332]
        , [0, 0, 0.190, 0.489, 0.949, 0.976, 1, 1.043, 1.053, 1.117, 1.456, 1.696, 3.76, 6.58, 9.09]])
    HUrange : 2-element array_like, optional
        2-element iterable of HU range to overwrite the physical properties. Only the structures that
        the HU values, derived from the calibration, are within the range (including the boundaries)
        will be overwritten. No warning will be displayed. (def. [-2000, 50000])
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK 3D Image
        An object of a transformed SimpleITK 3D image.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        setValueMask : setting values of the image inside/outside a mask.
    """
    from scipy.interpolate import interp1d
    import numpy as np
    import fredtools as ft
    import warnings
    import pydicom as dicom

    ft._isSITK3D(img, raiseError=True)

    # check if dicom is RN
    ft.ft_imgIO.dicom_io._isDicomRS(RSfileName, raiseError=True)

    # check HURange
    if not len(HUrange) == 2 or not HUrange[0] <= HUrange[1]:
        raise ValueError(f"The 'HUrange' parameter must be a 2-element iterable were the first element is less or equal to the second.")
    # get structures' info
    structsInfo = ft.getRSInfo(RSfileName)
    structsInfo.dropna(inplace=True)

    # prepare calibration from Rel. Electronic Density to HU
    relElecDensCalib = np.array(relElecDensCalib)
    relElecDensCalib = interp1d(relElecDensCalib[1], relElecDensCalib[0], bounds_error=True)

    # check if all Rel. Electronic Density are withing the calibration
    if not structsInfo.ROIPhysicalPropertyValue.between(relElecDensCalib.x.min(), relElecDensCalib.x.max()).all():
        warnings.warn(f"Warning: some of the structure physical property values are not within the calibration range [{relElecDensCalib.x.min()}, {relElecDensCalib.x.max()}]. They will be skipped.")
        structsInfo = structsInfo[structsInfo.ROIPhysicalPropertyValue.between(relElecDensCalib.x.min(), relElecDensCalib.x.max())]

    # check if all ROIPhysicalProperty are ["REL_ELEC_DENSITY"] (only REL_ELEC_DENSITY is supported for now).
    if not all(structsInfo.ROIPhysicalProperty.isin(["REL_ELEC_DENSITY"])):
        warnings.warn(f"Warning: some of the structure physical property are not in the supported list ['REL_ELEC_DENSITY']. They will be skipped.")
        structsInfo = structsInfo.loc[structsInfo.ROIPhysicalProperty.isin(["REL_ELEC_DENSITY"])]

    # calculate HU from Rel. Electronic Density
    structsInfo["ROIPhysicalHUValue"] = np.round(relElecDensCalib(structsInfo.ROIPhysicalPropertyValue))
    structsInfo = structsInfo.astype({"ROIPhysicalHUValue": "int"})

    # remove mapped structures outside the HU range
    structsInfo = structsInfo[structsInfo.ROIPhysicalHUValue.between(HUrange[0], HUrange[1])]

    for _, structInfo in structsInfo.iterrows():
        # map structure to img
        roiStruct = ft.mapStructToImg(img, RSfileName=RSfileName, structName=structInfo.ROIName, binaryMask=True, areaFraction=areaFraction, CPUNo=CPUNo)

        # set HU values inside the structure
        img = ft.setValueMask(img, roiStruct, value=structInfo.ROIPhysicalHUValue, outside=False)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        if len(structsInfo) > 0:
            print(f"# Overwritten HU values for {len(structsInfo)}", "structure" if len(structsInfo) <= 1 else "structures:")
            for ID, structInfo in structsInfo.iterrows():
                print(f"# Structure '{structInfo.ROIName}' (ID: {ID}) with HU={structInfo.ROIPhysicalHUValue} (Rel. Elec. Dens. {structInfo.ROIPhysicalPropertyValue})")
        else:
            print(f"# No structures found to overwrite HU values.")
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return img


def setIdentityDirection(img, displayInfo=False):
    """Set an identity direction for the image.

    The function sets an identity direction of an image defined as an instance of a
    SimpleITK image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Object SimpleITK image with identity direction.
    """
    import numpy as np
    import fredtools as ft

    ft._isSITK(img, raiseError=True)

    img.SetDirection(np.identity(img.GetDimension()).flatten())

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return img


def addMarginToMask(imgMask, marginLateral, marginProximal, marginDistal, lateralKernelType="circular", displayInfo=False):
    """Add lateral, proximal and distal margins to mask.

    The function adds lateral, proximal and/or distal margins to a binary mask defined as an
    instance of a SimpleITK 3D image describing a binary mask. The lateral directions are defined
    in the X and Y axes, whereas the distal and proximal are along the Z axis. It is the user's
    responsibility to transform the image into the correct view. Usually the 'getImgBEV'
    routine can be used to get the beam's eye view.

    Parameters
    ----------
    imgMask : SimpleITK 3D Image
        An object of a SimpleITK 3D image describing a binary mask.
    marginLateral : scalar
        Lateral margin in the mask unit, usually in [mm]
    marginProximal : scalar
        Proximal margin in the mask unit, usually in [mm]
    marginDistal : scalar
        Distal margin in the mask unit, usually in [mm]
    lateralKernelType : {'circular', 'box', 'cross'}, optional
        Kernel type for the lateral dilatation. (def. 'circular')
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK 3D Image
        Object of a SimpleITK 3D image describing the diluted mask.

    See Also
    --------
        mapStructToImg : mapping a structure to an image to create a mask.
        getImgBEV : transform an image to Beam's Eye View (BEV).
    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk

    ft._isSITK3D(imgMask, raiseError=True)
    ft._isSITK_maskBinary(imgMask, raiseError=True)

    # set kernel type
    if lateralKernelType.lower() == "circular":
        lateralKernelTypeEnum = sitk.sitkBall
    elif lateralKernelType.lower() == "box":
        lateralKernelTypeEnum = sitk.sitkBox
    elif lateralKernelType.lower() == "cross":
        lateralKernelTypeEnum = sitk.sitkCross
    else:
        raise ValueError(f"Lateral Kernel Type type '{lateralKernelType}' cannot be recognized. Only 'circular', 'box' and 'cross' are supported.")

    # get an interpolator suitable for mask interpolation
    interpolator = ft.ft_imgGetSubimg._setSITKInterpolator(interpolation="linear")

    # get pixel spacing
    pixelSpacing = imgMask.GetSpacing()

    # calculate margins in pixel
    marginLateralXPixel = int(np.round(marginLateral / pixelSpacing[0]))
    marginLateralYPixel = int(np.round(marginLateral / pixelSpacing[1]))
    marginDistalPixel = int(np.round(marginDistal / pixelSpacing[2]))
    marginProximalPixel = int(np.round(marginProximal / pixelSpacing[2]))

    # pad image with 0 values in proximal and distal directions
    imgMask = sitk.ConstantPad(imgMask, [0, 0, marginDistalPixel], [0, 0, marginProximalPixel], 0)

    # get lateral margin
    if marginLateral > 0:
        imgExtLateral = sitk.BinaryDilate(imgMask, kernelRadius=[marginLateralXPixel, marginLateralYPixel, 0], kernelType=lateralKernelTypeEnum)
    else:
        imgExtLateral = imgMask

    # get distal margin
    imgExtProximalDistal = sitk.BinaryDilate(imgExtLateral, kernelRadius=[0, 0, marginDistalPixel], kernelType=sitk.sitkBox)
    translationTransform = sitk.TranslationTransform(imgExtProximalDistal.GetDimension(), [0, 0, -marginDistal])
    imgExtDistal = sitk.Resample(imgExtProximalDistal, transform=translationTransform, interpolator=interpolator)
    imgExtDistal = sitk.BinaryFillhole(imgExtDistal)
    imgExtDistal = sitk.And(imgExtProximalDistal, imgExtDistal)

    # get proximal margin
    imgExtProximalDistal = sitk.BinaryDilate(imgExtLateral, kernelRadius=[0, 0, marginProximalPixel], kernelType=sitk.sitkBox)
    translationTransform = sitk.TranslationTransform(imgExtProximalDistal.GetDimension(), [0, 0, marginProximal])
    imgExtProximal = sitk.Resample(imgExtProximalDistal, transform=translationTransform, interpolator=interpolator)
    imgExtProximal = sitk.BinaryFillhole(imgExtProximal)
    imgExtProximal = sitk.And(imgExtProximalDistal, imgExtProximal)

    # merge all margins to a single structure mask
    imgExtProximalDistal = sitk.Or(imgExtDistal, imgExtProximal)
    imgExt = sitk.Or(imgExtLateral, imgExtProximalDistal)

    # crop image in proximal and distal directions to the original size
    imgExt = sitk.Crop(imgExt, [0, 0, marginDistalPixel], [0, 0, marginProximalPixel])

    # copy and modify information about the mask
    ft._copyImgMetaData(imgMask, imgExt)
    if "ROIName" in imgExt.GetMetaDataKeys():
        imgExt.SetMetaData("ROIName", imgExt.GetMetaData("ROIName") + " Margin")

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Added lateral (X/Y) margins: {marginLateral} mm of {lateralKernelType.lower()} type")
        print(f"# Added distal (+Z) margin: {marginDistal} mm")
        print(f"# Added proximal (-Z) margin: {marginProximal} mm")
        ft.ft_imgAnalyse._displayImageInfo(imgExt)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgExt
