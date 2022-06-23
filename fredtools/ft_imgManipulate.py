def mapStructToImg(img, RSfileName, structName, method="centreInside", algorithm="smparallel", CPUNo="auto", displayInfo=False):
    """Map structure to image and create a mask.

    The function reads a `structName` structure from RS dicom file and maps it to
    the frame of reference of `img` defined a SimpleITK image object. The created
    mask is an image with the same frame of reference (origin. spacing, direction
    and size) as the `img` with voxels inside the contour and 0 outside. It is
    assumed that the image is 3D and  has an unitary direction, which means that
    the axes describe X,Y and Z directions, respectively. The frame of reference
    of the `img` is not specified, in particular, the Z-spacing does not have to
    be the same as the structure Z-spacing.

    Two methods of mapping voxels are available of are available: 'allinside', which
    maps the voxels which are all inside the contour (including voxel size and edges),
    and 'centreInside' (default method), which maps only the centre of the voxels.

    Two algorithms of mapping are available: 'matplotlib', which utilizes the
    matplotlib.path.Path.contains_points functionality, and 'smparallel', which exploits
    the algorithm described in [1]_.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK 3D image.
    RSfileName : string
        Path String to dicom file with structures (RS file).
    structName : string
        Name of the structure to be mapped.
    method: {'centreInside', 'allInside'}, optional
        Method of calculation (def. 'centreInside'):

            -  'centreInside' : map the voxels which are all inside the contour.
            -  'allInside' : map only the centre of the voxels.

    algorithm: {'smparallel', 'matplotlib'}, optional
        Algorithm of calculation (def. 'smparallel'):

            -  'smparallel' : use matplotlib to calculate the voxels inside contours.
            -  'matplotlib' : use multiprocessing sm algorithm to calculate the voxels inside contours.

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
        Object of a SimpleITK image describing a mask (i.e. type 'uint8' with 0/1 values).

    See Also
    --------
        cropImgToMask: Crop image to mask boundary.
        getDVH: calculate DVH for structure

    Notes
    -----
    1. the functionality of ``gatetools`` package has been tested but it turned
    out that it does not consider holes and detached structures. Therefore a new
    approach has been implemented

    2. Although the implementation exploits multiple CPU computation using numba
    functionality, it would be recommended to move this functionality to GPU. Such implementation
    is described in [1]_ but has not been tested yet.

    3. Two methods of mapping voxels are available of are available: 'allinside', which
    maps the voxels which are all inside the contour (including voxel size and edges),
    and 'centreInside' (default method), which maps only the centre of the voxels.
    Obviously, the 'centreInside' method is faster. On the other hand it usually calculates
    the volume of the structure slightly larger than the real volume. Contrary, the 'allinside'
    method should always calculate smaller volume and should converge to the real volume while
    the `img` resolution increases.

    4. Two algorithms of mapping are available: 'matplotlib', which utilizes the
    matplotlib.path.Path.contains_points functionality, and 'smparallel', which exploits
    the algorithm described in [1]_ (search for 'Comparison of different methods') along with numba functionality of
    multiprocessing [2]_ (default algorithm). The 'matplotlib'
    algorithm calculates on a single CPU thread and is the slowest but it does not require
    any specific modules to be installed (basically the matplotlib) and should work on any platform.
    The 'smparallel' has been adapted from the above mentioned conversations and no significant
    changes have been made. Nevertheless, it has been tested against clinical Treatment Planning System
    (Varian Eclipse 15.6) and the standard 'matplotlib' method, showing no significant difference.
    Because, the 'smparallel' method utilizes numba module to speed and parallelise the computation,
    it might happen that it will not work on all platforms. Basically, the numba and tbb packages
    should be installed, but no testing on other platforms has been done.

    5. The mapping is done for each contour separately and based on the direction (CW or CCW)
    of the contour it is treated as an inclusive (mask, CW) or exclusive (hole, CCW) contour.
    The mapping of each contour is done in 2D, meaning slice by slice. The resulting image has
    the voxel size and shape the same as the input `img` in X and Y directions. The voxel size
    in Z direction is calculated based on the contour slice distances, taking into account
    gaps, holes and detached contours. The shape of the image in Z direction is equal to
    the contour boundings in Z direction, enlarged by 1 px (``sliceAddNo`` parameter). Such
    image mask, is then resampled to the frame of reference of the input `img`. In fact,
    the resampling is applied only to Z direction, because the frame of reference of X and Y
    directions are the same as the input `img`.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    .. [2] http://numba.pydata.org/
    """
    import numpy as np
    import fredtools as ft
    import SimpleITK as sitk

    if algorithm.lower() in ["smparallel", "sm"]:
        from numba import set_num_threads
        from .smparallel import is_inside_sm_parallel

        # set number of CPU to be used by numba
        if not CPUNo:
            set_num_threads(ft._getCPUNo(1))
        else:
            set_num_threads(ft._getCPUNo(CPUNo))
    elif algorithm.lower() in ["matplotlib", "mlp"]:
        import matplotlib.path as path
    else:
        raise ValueError(f"The algorithm '{algorithm}'' can not be recognised.")

    if not ft._isSITK3D(img, raiseError=True):
        raise ValueError(f"The image is a SimpleITK image of dimension {img.GetDimension()}. Only mapping ROI to 3D images are supported now.")

    if not ft.ft_imgIO.dicom_io._isDicomRS(RSfileName):
        raise ValueError(f"The file {RSfileName} is not a proper dicom describing structures.")

    # check if method is correct
    if not method.lower() in ["centreinside", "centerinside", "centre", "center", "allinside", "all"]:
        raise ValueError(f"The method '{method}' can not be recognised. Only ('centerinside','allinside') are possible")

    # check if the structName is in the RS dicom
    if not structName in ft.getRSInfo(RSfileName).ROIName.tolist():
        raise ValueError(f"The structure '{structName}' can not be found in the dicom RS file {RSfileName}")

    # get structure contour and structure info
    StructureContours, StructInfo = ft.dicom_io._getStructureContoursByName(RSfileName, structName)

    # check if the StructureContours is empty and return empty mask (filled with 0) if true
    if len(StructureContours)==0:
        # make empty mask (filled with 0)
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
            print("# Warrning: no 'StructureContours' was defined for this structure and an empty mask was returned")
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
    # get depth for each contour
    StructureContoursDepth = np.array([StructureContour[0, 2] for StructureContour in StructureContours])
    # get controur spacing in Z direction as the minimum spacing between individual contours excluding 0.
    """
    note: spacing 0 means that holes or detached contours exist in the structure
    note: more than single spacing (excluding 0) means that a gap exists in the structure
    """
    StructureContoursSpacing = np.unique(np.round(np.diff(np.sort(StructureContoursDepth)), decimals=3))
    StructureContoursSpacing = StructureContoursSpacing[StructureContoursSpacing != 0].min()
    # get structure bounding box, i.e. the min-max positions of vertices.
    """
    note: this is not extent because it takes the positions of vertices and not the positions of voxels' borders
    """
    StructureBBox = (
        (np.array([StructureContour[:, 0].min() for StructureContour in StructureContours]).min(), np.array([StructureContour[:, 0].max() for StructureContour in StructureContours]).max()),
        (np.array([StructureContour[:, 1].min() for StructureContour in StructureContours]).min(), np.array([StructureContour[:, 1].max() for StructureContour in StructureContours]).max()),
        (StructureContoursDepth.min(), StructureContoursDepth.max()),
    )
    # get CW (True) or CCW (False) direction for each contour
    StructureContoursDirection = [ft.ft_imgIO.dicom_io._checkContourCWDirection(StructureContour) for StructureContour in StructureContours]

    ### prepare mask
    # enlarge mask for depths +/- min/max of depths (add sliceAddNo depth slices at the beginning and ad the end)
    sliceAddNo = 1
    MaskDepths = np.round(
        np.arange(StructureContoursDepth.min() - StructureContoursSpacing * sliceAddNo, StructureContoursDepth.max() + StructureContoursSpacing * sliceAddNo, StructureContoursSpacing), 3
    )
    # prepare an empty mask
    StructureContoursMask = np.zeros([len(MaskDepths), img.GetSize()[1], img.GetSize()[0]], dtype="bool")
    imgMask = sitk.GetImageFromArray(StructureContoursMask.astype("uint8"))
    imgMask.SetOrigin([img.GetOrigin()[0], img.GetOrigin()[1], MaskDepths.min()])
    imgMask.SetSpacing([img.GetSpacing()[0], img.GetSpacing()[1], StructureContoursSpacing])

    imgX_px, imgY_px = np.meshgrid(np.arange(imgMask.GetSize()[0]), np.arange(imgMask.GetSize()[1]))
    imgPos_px = np.concatenate((np.expand_dims(imgX_px.flatten(), 1), np.expand_dims(imgY_px.flatten(), 1)), axis=1)

    # prepare empty mask
    StructureContoursMask = np.zeros((MaskDepths.size, imgMask.GetSize()[1], imgMask.GetSize()[0]), dtype="bool")

    for MaskDepth_idx, MaskDepth in enumerate(MaskDepths):
        # get list of indices of StructureContours for MaskDepth
        StructureContoursIdx = np.where(StructureContoursDepth == MaskDepth)[0]

        # skip calculation of the mask if no structure exists for MaskDepth
        if not np.any(StructureContoursIdx):
            continue

        for StructureContourIdx in StructureContoursIdx:
            StructureContour_rw = StructureContours[StructureContourIdx]

            # convert vertices from real world coordinate to pixel coordinate
            StructureContour_px = np.array([imgMask.TransformPhysicalPointToContinuousIndex(Vertex) for Vertex in StructureContour_rw])

            ### remove all pixels from Mask that for sure are not inside the contour based on the contour boundaries
            """
            note: pixel coordinates of the contour bounding box (x1, x2, y1, y2) enlarged by StructureContourBBoxEnlarge_px px (floored/ceiled to int)
            """
            StructureContourBBoxEnlarge_px = 1
            StructureContourBBox_px = np.array(
                [
                    np.floor(StructureContour_px[:, 0].min()) - StructureContourBBoxEnlarge_px,
                    np.ceil(StructureContour_px[:, 0].max()) + StructureContourBBoxEnlarge_px,
                    np.floor(StructureContour_px[:, 1].min()) - StructureContourBBoxEnlarge_px,
                    np.ceil(StructureContour_px[:, 1].max()) + StructureContourBBoxEnlarge_px,
                ]
            )
            # correct StructureContourBBox_px in case when the structure is larger than the image
            StructureContourBBox_px[StructureContourBBox_px < 0] = 0
            StructureContourBBox_px[1] = imgPos_px[:, 0].max() if StructureContourBBox_px[1] > imgPos_px[:, 0].max() else StructureContourBBox_px[1]
            StructureContourBBox_px[3] = imgPos_px[:, 1].max() if StructureContourBBox_px[3] > imgPos_px[:, 1].max() else StructureContourBBox_px[3]
            # correct StructureContourBBox_px in case when the structure is not inside the image at all
            StructureContourBBox_px[0] = StructureContourBBox_px[1] if StructureContourBBox_px[0] > StructureContourBBox_px[1] else StructureContourBBox_px[0]
            StructureContourBBox_px[2] = StructureContourBBox_px[3] if StructureContourBBox_px[2] > StructureContourBBox_px[3] else StructureContourBBox_px[2]

            StructureContourBBox_px = StructureContourBBox_px.astype("uint32")
            # logical vector of image positions inside the enlarged contour bounding box
            imgPosInsideStructureContourBBox_px = np.logical_and(
                np.logical_and(imgPos_px[:, 0] >= StructureContourBBox_px[0], imgPos_px[:, 0] <= StructureContourBBox_px[1]),
                np.logical_and(imgPos_px[:, 1] >= StructureContourBBox_px[2], imgPos_px[:, 1] <= StructureContourBBox_px[3]),
            )

            # get only those image pixel position that are inside the enlarged contour bounding box
            imgPosToMap_px = imgPos_px[imgPosInsideStructureContourBBox_px, :]

            # contour bounding box shape
            StructureContourBBoxShape_px = (np.diff(StructureContourBBox_px[[0, 1]])[0] + 1, np.diff(StructureContourBBox_px[[2, 3]])[0] + 1)

            if method.lower() in ["allinside", "all"]:
                # calculate mask only for pixels all inside (for CW contour direction) or all outside (for CCW contour direction) the contour
                StructureContourMask = []
                for shift_px in [[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]:
                    if algorithm.lower() in ["matplotlib", "mlp"]:
                        StructureContourPath_px = path.Path(StructureContour_px[:, 0:2] + shift_px, closed=True)
                        StructureContourMask.append(np.reshape(StructureContourPath_px.contains_points(imgPosToMap_px, radius=1e-9), StructureContourBBoxShape_px[::-1]))
                    elif algorithm.lower() in ["smparallel", "sm"]:
                        StructureContourMask.append(np.reshape(is_inside_sm_parallel(imgPosToMap_px, StructureContour_px[:, 0:2] + shift_px), StructureContourBBoxShape_px[::-1]))
                if StructureContoursDirection[StructureContourIdx]:
                    StructureContourMask = np.logical_and.reduce(StructureContourMask)
                else:
                    StructureContourMask = np.logical_or.reduce(StructureContourMask)
            elif method.lower() in ["centreinside", "centerinside", "centre", "center"]:
                if algorithm.lower() in ["matplotlib", "mlp"]:
                    StructureContourPath_px = path.Path(StructureContour_px[:, 0:2], closed=True)
                    StructureContourMask = np.reshape(StructureContourPath_px.contains_points(imgPosToMap_px, radius=1e-9), StructureContourBBoxShape_px[::-1])
                elif algorithm.lower() in ["smparallel", "sm"]:
                    StructureContourMask = np.reshape(is_inside_sm_parallel(imgPosToMap_px, StructureContour_px[:, 0:2]), StructureContourBBoxShape_px[::-1])

            # add pad to StructureContourMask to obtain the same shape in X/Y as the imgMask
            StructureContourMask = np.pad(
                StructureContourMask,
                pad_width=((StructureContourBBox_px[2], imgMask.GetSize()[1] - StructureContourBBox_px[3] - 1), (StructureContourBBox_px[0], imgMask.GetSize()[0] - StructureContourBBox_px[1] - 1)),
                constant_values=False,
            )

            # add mask for slice (2D image) to StructureContoursMask (3D image) taking into account if the contour is CW (structure) or CCW (hole)
            if StructureContoursDirection[StructureContourIdx]:
                StructureContoursMask[MaskDepth_idx, :, :] = np.logical_or(StructureContoursMask[MaskDepth_idx, :, :], StructureContourMask)
            else:
                StructureContoursMask[MaskDepth_idx, :, :] = np.logical_xor(StructureContoursMask[MaskDepth_idx, :, :], StructureContourMask)

    # make SimpleITK mask
    imgMask = sitk.GetImageFromArray(StructureContoursMask.astype("uint8"))
    imgMask.SetOrigin([img.GetOrigin()[0], img.GetOrigin()[1], MaskDepths.min()])
    imgMask.SetSpacing([img.GetSpacing()[0], img.GetSpacing()[1], StructureContoursSpacing])

    # interpolate mask to input image
    imgMask = sitk.Cast(imgMask, sitk.sitkFloat32)
    imgMask = sitk.Resample(imgMask, img, interpolator=ft.ft_imgGetSubimg._setSITKInterpolator(interpolation="linear"))
    imgMask = sitk.BinaryThreshold(imgMask, lowerThreshold=0.5, upperThreshold=100, insideValue=1, outsideValue=0)
    imgMask = sitk.Cast(imgMask, sitk.sitkUInt8)

    # set additional metadata
    imgMask.SetMetaData("ROIColor", str(StructInfo["Color"]))
    imgMask.SetMetaData("ROIName", StructInfo["Name"])
    imgMask.SetMetaData("ROINumber", str(StructInfo["Number"]))
    imgMask.SetMetaData("ROIGenerationAlgorithm", StructInfo["GenerationAlgorithm"])
    imgMask.SetMetaData("ROIType", StructInfo["Type"])

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Structure name (type): '{:s}' ({:s})".format(StructInfo["Name"], StructInfo["Type"]))
        print("# Structure volume: {:.3f} cm3".format(ft.arr(imgMask).sum() * np.prod(np.array(imgMask.GetSpacing())) / 1e3))
        ft.ft_imgAnalyse._displayImageInfo(imgMask)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return imgMask


def cropImgToMask(img, imgMask, displayInfo=False):
    """Crop image to mask boundary.

    The function calculates the boundaries of the `imgMask` defined
    as an instance of a SimpleITK image object describing a mask
    (i.e. type uint8 and only 0/1 values) and crops the `img` defined
    as an instance of a SimpleITK image object to these boundaries. The boundaries
    mean here the most extreme positions of positive values of the mask in
    each direction. The function exploits SimpleITK.Crop routine.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    imgMask : SimpleITK Image
        Object of a SimpleITK image describing a mask.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

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
    a SimpleITK object which are inside or outside a mask described by the
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
        Displays a summary of the function results. (def. False)

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
    the corner of the first voxel preserved. The size of
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
        Displays a summary of the function results. (def. False)

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
        Displays a summary of the function results. (def. False)

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
    calculated from the starting and ending points of the cylinder in the frame of
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
        Displays a summary of the function results. (def. False)

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


def sumVectorImg(img, displayInfo=False):
    """Sum vector image.

    The function sums all elements of vector in a vector image
    defined as instances of a SimpleITK vector image object.
    The resulting image have the same frame of reference but
    is a scalar image.

    Parameters
    ----------
    img : SimpleITK Vector Image
        Object of a SimpleITK vector image.
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

    The function is calculating a new Field of Reference (FOR) for an image defined
    as an instance od a SimpleITK image object. The purpose of this calculation is
    that transformed images can be 'cropped'. This function is calculating new FOR based
    of the positions of the transformed image corners.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    transform : SimpleITK Transform
        Object of a SimpleITK transform.

    Returns
    -------
    size, origin, spacing, direction
        Calculated image size, origin, spacing (the same as original)
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
    gantry angle and couch rotation, using defined interpolation method.
    The BEV Field of Reference (FOR) means that the Z+ direction is along the field
    (along the beam of relative position [0,0]) and X/Y positions are consistend with
    the DICOM and FRED Monte Carlo definitions.

    Parameters
    ----------
    img : SimpleITK 3D Image
        Object of a SimpleITK 3D image.
    isocentrePosition : array_like, (3x1)
        Position of the isocentre with respect to the `img` FOR.
    gantryAngle : scalar
        Rotation of the gantry around the isocentre position in [deg].
    couchAngle: scalar
        Rotation of the couch around the isocentre position in [deg].
    defaultPixelValue: 'auto' or scalar, optional
        The value to fill the voxels with, outside the original `img`.
        If 'auto', then the value will be calculated automatically as the
        minimum value of the `img`. (def. 'auto')
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK 3D Image
        Object of a transformed SimpleITK 3D image.

    Notes
    -----
    The basic workflow follows:

        1. translate the image to the isocentre so to have the isocentre at zero position,
        2. rotate the couch around the isocentre,
        3. rotate the gantry around the isocentre,
        4. rotate and flip image to get BEV.

    Note that the isocentre of the transformed image is at the zero point.

    Note that the isocentre defined in the delivery sequence of the FRED rtplan is
    a negative isocentre defined in the DICOM RN plan, the couch rotation defined
    in the delivery sequence of the FRED rtplan is a negative couch rotation defined
    in the DICOM RN plan, but the gantry rotation defined in the delivery sequence
    of the FRED rtplan is equal to the gantry rotation the DICOM RN plan.
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

    # determine default pixelvalue
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
    flipXYTransform = sitk.ScaleTransform(img.GetDimension(), (-1, -1, 1))
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

    # make transformation with resampling
    imgBEV = sitk.Resample(
        img, transform=compositTransform, size=size, outputOrigin=origin, outputSpacing=spacing, interpolator=interpolator, outputDirection=direction, defaultPixelValue=defaultPixelValue
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


def overwriteCTPhysicalProperties(img, RSfileName, method="centreInside", algorithm="smparallel", CPUNo="auto", relElecDensCalib=[[-1000, 100, 1000, 6000], [0, 1.1, 1.532, 3.920]], displayInfo=False):
    """Overwrite HU values in a CT image based on structures physical properties.

    The function searches in a structure RS dicom file for structures with
    the physical property defined, maps each structure to the CT image
    defined as an instance of a SimpleITK 3D image and replaces the Hounsfield Units (HU)
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
    method: {'centreInside', 'allInside'}, optional
        Method of calculation. See `mapStructToImg` function for more information. (def. 'centreInside')
    algorithm: {'smparallel', 'matplotlib'}, optional
        Algorithm of calculation. See `mapStructToImg` function for more information. (def. 'smparallel')
    CPUNo : {'auto', 'none'}, scalar or None, optional
        Define if the multiprocessing should be used and how many cores should
        be exploited. See `mapStructToImg` function for more information. (def. 'auto')
    relElecDensCalib: array_like, optional
        2xN iterable (e.g. 2xN numpy array or list of two equal size lists) describing
        the calibration between HU values and relative electronic density. The first element (column)
        is describing the HU values and the second the relative electronic density. The missing values
        are interpolated linearly and if the user would like to use a different interpolation
        like spline or polynominal, it is advised to provide it explicitely for each HU value.
        (def. [[-1000, 100, 1000, 6000], [0, 1.1, 1.532, 3.920]] )

    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK 3D Image
        Object of a transformed SimpleITK 3D image.

    See Also
    --------
        mapStructToImg: mapping a structure to image to create a mask.
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

    # get structures' info
    structsInfo = ft.getRSInfo(RSfileName)
    structsInfo.dropna(inplace=True)

    # prepare calibration from Rel. Electronic Density to HU
    relElecDensCalib = np.array(relElecDensCalib)
    relElecDensCalib = interp1d(relElecDensCalib[1], relElecDensCalib[0], bounds_error=False, fill_value="extrapolate")

    # calculate HU from Rel. Electronic Density
    structsInfo["ROIPhysicalHUValue"] = np.round(relElecDensCalib(structsInfo.ROIPhysicalPropertyValue))
    structsInfo = structsInfo.astype({"ROIPhysicalHUValue": "int"})

    # check if all ROIPhysicalProperty are ["REL_ELEC_DENSITY"] (only REL_ELEC_DENSITY is supported for now).
    if not all(structsInfo.ROIPhysicalProperty.isin(["REL_ELEC_DENSITY"])):
        warnings.warn(f"Some of the structure physical property are not in the supported list ['REL_ELEC_DENSITY']. They will be skipped.")
        structsInfo = structsInfo.loc[structsInfo.ROIPhysicalProperty.isin(["REL_ELEC_DENSITY"])]

    # read dicom tags
    dicomTags = dicom.read_file(RSfileName)

    for _, structInfo in structsInfo.iterrows():
        # map structure to img
        roiStruct = ft.mapStructToImg(img, RSfileName=RSfileName, structName=structInfo.ROIName, method=method, algorithm=algorithm, CPUNo=CPUNo)

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
