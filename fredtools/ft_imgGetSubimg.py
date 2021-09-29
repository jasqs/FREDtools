def _setSITKInterpolator(interpolation="linear", splineOrder=3):
    """Set SimpleITK interpolator for interpolation method.

    The function is setting a specific interpolation method for
    SimpleITK image objects.

    Parameters
    ----------
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)

    Returns
    -------
    interpolator
        Object of a SimpleITK interpolator.
    """
    import SimpleITK as sitk

    # set interpolation method
    if interpolation.lower() == "linear":
        return sitk.sitkLinear
    elif interpolation.lower() == "nearest":
        return sitk.sitkNearestNeighbor
    elif interpolation.lower() == "spline":
        if splineOrder > 5 or splineOrder < 0:
            raise ValueError(f"Spline order must be in range 0-5.")
        if splineOrder == 0:
            return sitk.sitkBSplineResampler
        elif splineOrder == 1:
            return sitk.sitkBSplineResamplerOrder1
        elif splineOrder == 2:
            return sitk.sitkBSplineResamplerOrder2
        elif splineOrder == 3:
            return sitk.sitkBSplineResamplerOrder3
        elif splineOrder == 4:
            return sitk.sitkBSplineResamplerOrder4
        elif splineOrder == 5:
            return sitk.sitkBSplineResamplerOrder5
    else:
        raise ValueError(f"Interpolation type '{interpolation}' cannot be recognized. Only 'linear', 'nearest' and 'spline' are supported.")


def getSlice(img, point, plane="XY", interpolation="linear", splineOrder=3, raiseWarning=True, displayInfo=False):
    """Get 2D slice from image.

    The function calculates a 2D slice image through a specified
    `point` in a specified `plane` from an image defined as SimpleITK
    image object. The slice is returned as an instance of a SimpleITK image
    object of the same dimension but describing a slice (the dimension
    of only two axes are different than one). The slice through a specified
    point is calculated with a specified `interpolation` type.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    point : array_like
        Point to generate the 2D slice through. It should have length
        of the image dimension. A warning will be generated if the point is
        not inside the image extent.
    plane : str, optional
        Plane to generate the 2D slice given as a string. The string
        should have form of two letters from [XYZT] set with +/- signs
        (if no sign provided, then + is assumed). For instance it can be:
        `XY`,`ZY`,`-YX`, `Y-T`, etc. If the minus sign is found, then the
        image is flipped in the following direction. The order of axis is
        important and the output will be generated in this way to be consistent
        with the axes displayed with matplotlib.pyplot.imshow. For instance
        plane `Z-X` will display Z-axis on X-axis in imshow and Y-axis of
        of imshow will be a reversed X-axis of the image. (def. 'XY')
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)
    raiseWarning : bool, optional
        Raise warnings. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Instance of a SimpleITK image object describing a slice.

    See Also
    --------
        matplotlib.pyplot.imshow: displaying 2D images.
        getExtent: get extent of the image in each direction.
        arr: get squeezed array from image.

    Examples
    --------
    The example below shows how to generate and display a 2D slice along 'ZX'
    (with reversed X) axes from a 3D image through a specified point of the 3D image.

    >>> img3D = fredtools.readMHD('imageCT.mhd', displayInfo = True)
    ### readMHD ###
    # dims (xyz) =  [511 415 218]
    # pixel size [mm] =  [0.68359375 0.68359375 1.2       ]
    # origin [mm]     =  [-174.65820312 -354.28710938 -785.6       ]
    # x-spatial voxel centre [mm] =  [  -174.658203,  -173.974609, ...,   173.291016,   173.974609 ]
    # y-spatial voxel centre [mm] =  [  -354.287109,  -353.603516, ...,   -71.962891,   -71.279297 ]
    # z-spatial voxel centre [mm] =  [  -785.600000,  -784.400000, ...,  -526.400000,  -525.200000 ]
    # x-spatial extent [mm] =  [  -175.000000 ,   174.316406 ] =>   349.316406
    # y-spatial extent [mm] =  [  -354.628906 ,   -70.937500 ] =>   283.691406
    # z-spatial extent [mm] =  [  -786.200000 ,  -524.600000 ] =>   261.600000
    # volume = 25924053.15 mm3  =>  25.92 litre
    # data type:  16-bit signed integer
    # range: from  -1024  to  3071
    # sum = -33870013138 , mean = -732.6387321958799 ( 468.4351806663016 )
    # non-zero (dose=0)  voxels  = 46188861 (99.91%) => 25.90 litre
    # non-air (HU>-1000) voxels  = 15065800 (32.59%) => 8.45 litre
    ###############
    >>> sl2D = fredtools.getSlice(img3D, point=[0,-212.42,-654.8], plane='Z-X', displayInfo=True)
    ### getSlice ###
    # Point:  [   0.   -212.42 -654.8 ]
    # Plane: 'Z-X'
    # dims (xyz) =  [218   1 511]
    # pixel size [mm] =  [1.2        0.68359375 0.68359375]
    # origin [mm]     =  [-174.65820312 -212.42       -525.2       ]
    # x-spatial voxel centre [mm] =  [  -525.200000,  -526.400000, ...,  -784.400000,  -785.600000 ]
    # y-spatial voxel centre [mm] =  [  -212.420000 ]
    # z-spatial voxel centre [mm] =  [  -174.658203,  -173.974609, ...,   173.291016,   173.974609 ]
    # x-spatial extent [mm] =  [  -524.600000 ,  -786.200000 ] =>   261.600000
    # y-spatial extent [mm] =  [  -212.761797 ,  -212.078203 ] =>     0.683594
    # z-spatial extent [mm] =  [  -175.000000 ,   174.316406 ] =>   349.316406
    # volume = 62467.60 mm3  =>  0.06 litre
    # data type:  16-bit signed integer
    # range: from  -1024  to  1803
    # sum = -53645122 , mean = -481.56270310059426 ( 559.2583473799535 )
    # non-zero (dose=0)  voxels  = 111051 (99.69%) => 0.06 litre
    # non-air (HU>-1000) voxels  = 57831 (51.91%) => 0.03 litre
    ################
    >>> matplotlib.pyplot.imshow(fredtools.arr(sl2D), extent=fredtools.getExtMpl(sl2D))
    >>> matplotlib.pyplot.xlabel('Z [mm]')
    >>> matplotlib.pyplot.ylabel('X [mm] (reversed)')
    """
    import re
    import warnings
    import SimpleITK as sitk
    import fredtools as ft
    import numpy as np

    if not (ft._isSITK3D(img) or ft._isSITK4D(img)):
        raise TypeError(f"The object '{type(img)}' is not an instance of a 3D or 4D SimpleITK image.")

    # check if point dimension matches the img dim.
    if len(point) != img.GetDimension():
        raise ValueError(f"Dimension of 'point' {point} does not match 'img' dimension {img.GetDimension()}.")

    # set interpolator
    interpolator = ft.ft_imgGetSubimg._setSITKInterpolator(interpolation=interpolation, splineOrder=splineOrder)

    # check if point is inside the image
    if not ft.isPointInside(img, point) and raiseWarning:
        warnings.warn(f"Warning: the point {point} is not inside the image extent: {ft.getExtent(img)}.")

    # check if plane is in proper format
    plane = plane.upper()
    if not {"X", "Y", "Z", "T", "-", "+"}.issuperset(plane):
        raise ValueError(f"Plane parameter '{plane}' cannot be recognized. Only letters 'X','Y','Z','T','-','+' are supported.")
    if len(plane) > 4:
        raise ValueError(f"Plane parameter '{plane}' cannot be recognized. The length of the plane parameter should less or equal than 4.")

    # remove all signs from the plane definition
    planeSimple = re.sub("[-+]", "", plane)

    # determine available axis names based on img dimension
    axesNameAvailable = ["X", "Y", "Z", "T"][0 : img.GetDimension()]

    # check if plane definition is correct for img dimension
    for planeSimpleAxis in planeSimple:
        if not planeSimpleAxis in axesNameAvailable:
            raise ValueError(f"Axis '{planeSimple}' cannot be recongised for 'img' of dimension {img.GetDimension()}. Only combination of {axesNameAvailable} is possible.")

    # determine axes for slice
    axisNo = []
    for planeSimpleAxis in planeSimple:
        axisNo.append([i for i, x in enumerate([planeSimpleAxis == i for i in axesNameAvailable]) if x][0])
    axisVec = np.zeros(img.GetDimension())
    axisVec[axisNo] = 1

    # determine shape
    slSize = np.array(img.GetSize()) * axisVec
    slSize[slSize == 0] = 1

    # determine origin 3D
    slOrigin3D = np.array(point, dtype="float64")
    slOrigin3D[axisNo] = np.array(img.GetOrigin())[axisNo]

    # generate slice
    sl = sitk.Resample(
        img,
        size=[int(x) for x in slSize],
        outputSpacing=img.GetSpacing(),
        outputDirection=img.GetDirection(),
        outputOrigin=slOrigin3D,
        interpolator=interpolator,
    )

    # swap axes if requested
    if axisNo != sorted(axisNo):
        permuteOrder = list(range(img.GetDimension()))
        permuteOrder[axisNo[0]], permuteOrder[axisNo[1]] = permuteOrder[axisNo[1]], permuteOrder[axisNo[0]]
        sl = sitk.PermuteAxes(sl, order=permuteOrder)

    # flip axes if requested
    if "-" in plane:
        axesFlip = [False] * img.GetDimension()
        for axesNameAvailable_idx, axesNameAvailable in enumerate(axesNameAvailable):
            if not re.findall("(-){:s}".format(axesNameAvailable), plane):
                continue
            else:
                axesFlip[axesNameAvailable_idx] = True
        sl = sitk.Flip(sl, flipAxes=axesFlip)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Point: ", np.array(point))
        print("# Plane: '{:s}'".format(plane))
        ft.ft_imgAnalyse._displayImageInfo(sl)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return sl


def getProfile(img, point, axis="X", interpolation="linear", splineOrder=3, raiseWarning=True, displayInfo=False):
    """Get 1D profile from image along axis.

    The function calculates a 1D profile image through a specified
    `point` in a specified `axis` from an image defined as SimpleITK
    image object. The profile is returned as an instance of a SimpleITK image
    object of the same dimension but describing a profile (the dimension
    of only one axes is different than one). The profile through a specified
    point is calculated with a specified `interpolation` type.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    point : array_like
        Point to generate the 2D slice through. It should have length
        of the image dimension. A warning will be generated if the point is
        not inside the image extent.
    axis : str, optional
        Axis to generate the 1D profile given as a string. The string
        should have form of one letter from [XYZT] set with +/- signs
        (if no sign provided, then + is assumed). For instance it can be:
        `X`,`Y`,`-Z`, etc. If the minus sign is found, then the
        image is flipped in the following direction.
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)
    raiseWarning : bool, optional
        Raise warnings. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Instance of a SimpleITK image object describing a profile.

    See Also
    --------
        matplotlib.pyplot.plot: displaying profiles.
        pos: get voxels' centres for axes of size different than one.
        vec: get vector with values for the img describing a profile.

    Examples
    --------
    The example below shows how to generate and display a 1D profile along 'Y'
    (reversed) axis from a 3D image through a specified point of the 3D image.

    >>> img3D = fredtools.readMHD('imageCT.mhd', displayInfo = True)
    ### readMHD ###
    # dims (xyz) =  [511 415 218]
    # pixel size [mm] =  [0.68359375 0.68359375 1.2       ]
    # origin [mm]     =  [-174.65820312 -354.28710938 -785.6       ]
    # x-spatial voxel centre [mm] =  [  -174.658203,  -173.974609, ...,   173.291016,   173.974609 ]
    # y-spatial voxel centre [mm] =  [  -354.287109,  -353.603516, ...,   -71.962891,   -71.279297 ]
    # z-spatial voxel centre [mm] =  [  -785.600000,  -784.400000, ...,  -526.400000,  -525.200000 ]
    # x-spatial extent [mm] =  [  -175.000000 ,   174.316406 ] =>   349.316406
    # y-spatial extent [mm] =  [  -354.628906 ,   -70.937500 ] =>   283.691406
    # z-spatial extent [mm] =  [  -786.200000 ,  -524.600000 ] =>   261.600000
    # volume = 25924053.15 mm3  =>  25.92 litre
    # data type:  16-bit signed integer
    # range: from  -1024  to  3071
    # sum = -33870013138 , mean = -732.6387321958799 ( 468.4351806663016 )
    # non-zero (dose=0)  voxels  = 46188861 (99.91%) => 25.90 litre
    # non-air (HU>-1000) voxels  = 15065800 (32.59%) => 8.45 litre
    ###############
    >>> pr1D = fredtools.getProfile(img3D, point=[0,-212.42,-654.8], axis='-Y', displayInfo=True)
    ### getProfile ###
    # Point:  [   0.   -212.42 -654.8 ]
    # Axis: '-Y'
    # dims (xyz) =  [  1 415   1]
    # pixel size [mm] =  [0.68359375 0.68359375 1.2       ]
    # origin [mm]     =  [   0.          -71.27929688 -654.8       ]
    # x-spatial voxel centre [mm] =  [     0.000000 ]
    # y-spatial voxel centre [mm] =  [   -71.279297,   -71.962891, ...,  -353.603516,  -354.287109 ]
    # z-spatial voxel centre [mm] =  [  -654.800000 ]
    # x-spatial extent [mm] =  [    -0.341797 ,     0.341797 ] =>     0.683594
    # y-spatial extent [mm] =  [   -70.937500 ,  -354.628906 ] =>   283.691406
    # z-spatial extent [mm] =  [  -655.400000 ,  -654.200000 ] =>     1.200000
    # volume = 232.72 mm3  =>  0.00 litre
    # data type:  16-bit signed integer
    # range: from  -1023  to  1336
    # sum = -131818 , mean = -317.63373493975905 ( 487.9811134201045 )
    # non-zero (dose=0)  voxels  = 414 (99.76%) => 0.00 litre
    # non-air (HU>-1000) voxels  = 354 (85.30%) => 0.00 litre
    ##################
    >>> matplotlib.pyplot.plot(fredtools.pos(pr1D), fredtools.vec(pr1D))
    >>> matplotlib.pyplot.xlabel('Y [mm] (reversed)')
    >>> matplotlib.pyplot.ylabel('Values')
    """
    import re
    import warnings
    import SimpleITK as sitk
    import fredtools as ft
    import numpy as np

    ft._isSITK(img, raiseError=True)

    # check if img is already a profile
    if ft._isSITK_profile(img):
        raise TypeError(f"The object '{type(img)}' is already an instance SimpleITK image describing a profile.")

    # check if point dimension matches the img dim.
    if (len(point) != img.GetDimension()) and (len(point) != len(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))):
        raise ValueError(
            f"Dimension of 'point' {point} is {len(point)}. The 'img' is of dimension {img.GetDimension()} but is describing a {len(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))}D image. The 'point' should have dimension {len(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))} or {img.GetDimension()}."
        )

    # correct point is needed
    if len(point) < img.GetDimension():
        pointCorr = np.array(img.GetOrigin())
        pointCorr[list(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))] = point
        point = pointCorr

    # set interpolator
    interpolator = ft.ft_imgGetSubimg._setSITKInterpolator(interpolation=interpolation, splineOrder=splineOrder)

    # check if point is inside the image
    if not ft.isPointInside(img, point) and raiseWarning:
        warnings.warn(f"Warning: the point {point} is not inside the image extent: {ft.getExtent(img)}.")

    # check if axis is in proper format
    axis = axis.upper()
    if not {"X", "Y", "Z", "T", "-", "+"}.issuperset(axis):
        raise ValueError(f"Axis parameter {axis} cannot be recognized. Only letters 'X','Y','Z','T','-','+' are supported.")
    if len(axis) > 2:
        raise ValueError(f"Axis parameter {axis} cannot be recognized. The length of the plane parameter should less or equal to 2.")

    # remove all signs from the plane definition
    axisSimple = re.sub("[-+]", "", axis)

    # determine available axis names based on img dimension
    axesNameAvailable = ["X", "Y", "Z", "T"][0 : img.GetDimension()]

    # check if profile definition is correct for img dimension
    if not axisSimple in axesNameAvailable:
        raise ValueError(f"Axis '{axisSimple}' cannot be recongised for 'img' of dimension {img.GetDimension()}. Only {axesNameAvailable} are possible.")

    # determine axis for profile
    axisNo = [i for i, x in enumerate([axisSimple == i for i in axesNameAvailable]) if x][0]
    axisVec = np.zeros(img.GetDimension())
    axisVec[axisNo] = 1

    # determine shape
    prSize = np.array(img.GetSize()) * axisVec
    prSize[prSize == 0] = 1

    # determine origin
    prOrigin = list(point)
    prOrigin[axisNo] = img.GetOrigin()[axisNo]

    # generate profile
    prof = sitk.Resample(
        img,
        size=[int(x) for x in prSize],
        outputSpacing=img.GetSpacing(),
        outputDirection=img.GetDirection(),
        outputOrigin=prOrigin,
        interpolator=interpolator,
    )

    # flip axes if requested
    if "-" in axis:
        axesFlip = [False] * img.GetDimension()
        axesFlip[axisNo] = True
        prof = sitk.Flip(prof, flipAxes=axesFlip)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Point: ", np.array(point))
        print("# Axis: '{:s}'".format(axis))
        ft.ft_imgAnalyse._displayImageInfo(prof)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return prof


def getPoint(img, point, interpolation="linear", splineOrder=3, raiseWarning=True, displayInfo=False):
    """Get point value from image.

    The function calculates a point value in a specified `point` from an
    image defined as SimpleITK image object. The point is returned as an
    instance of a SimpleITK image object of the same dimension but describing
    a point (the dimension of all axes is equal to one). The point value in
    a specified point is calculated with a specified `interpolation` type.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    point : array_like
        Point to generate the value. It should have length of the image
        dimension. A warning will be generated if the point is not inside
        the image extent.
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)
    raiseWarning : bool, optional
        Raise warnings. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Instance of a SimpleITK image object describing a point.

    Examples
    --------
    The example below shows how to generate a point value for various interpolation
    types from a 3D image in a specified point of the 3D image.

    >>> img3D = fredtools.readMHD('imageCT.mhd', displayInfo = True)
    ### readMHD ###
    # dims (xyz) =  [511 415 218]
    # pixel size [mm] =  [0.68359375 0.68359375 1.2       ]
    # origin [mm]     =  [-174.65820312 -354.28710938 -785.6       ]
    # x-spatial voxel centre [mm] =  [  -174.658203,  -173.974609, ...,   173.291016,   173.974609 ]
    # y-spatial voxel centre [mm] =  [  -354.287109,  -353.603516, ...,   -71.962891,   -71.279297 ]
    # z-spatial voxel centre [mm] =  [  -785.600000,  -784.400000, ...,  -526.400000,  -525.200000 ]
    # x-spatial extent [mm] =  [  -175.000000 ,   174.316406 ] =>   349.316406
    # y-spatial extent [mm] =  [  -354.628906 ,   -70.937500 ] =>   283.691406
    # z-spatial extent [mm] =  [  -786.200000 ,  -524.600000 ] =>   261.600000
    # volume = 25924053.15 mm3  =>  25.92 litre
    # data type:  16-bit signed integer
    # range: from  -1024  to  3071
    # sum = -33870013138 , mean = -732.6387321958799 ( 468.4351806663016 )
    # non-zero (dose=0)  voxels  = 46188861 (99.91%) => 25.90 litre
    # non-air (HU>-1000) voxels  = 15065800 (32.59%) => 8.45 litre
    ###############
    >>> pointValue = fredtools.getPoint(img3D, point=[0,-212.42,-654.8], displayInfo=True)
    ### getPoint ###
    # Point:  [   0.   -212.42 -654.8 ]
    # Value:  45
    # dims (xyz) =  [1 1 1]
    # pixel size [mm] =  [0.68359375 0.68359375 1.2       ]
    # origin [mm]     =  [   0.   -212.42 -654.8 ]
    # x-spatial voxel centre [mm] =  [     0.000000 ]
    # y-spatial voxel centre [mm] =  [  -212.420000 ]
    # z-spatial voxel centre [mm] =  [  -654.800000 ]
    # x-spatial extent [mm] =  [    -0.341797 ,     0.341797 ] =>     0.683594
    # y-spatial extent [mm] =  [  -212.761797 ,  -212.078203 ] =>     0.683594
    # z-spatial extent [mm] =  [  -655.400000 ,  -654.200000 ] =>     1.200000
    # volume = 0.56 mm3  =>  0.00 litre
    # data type:  16-bit signed integer
    # range: from  45  to  45
    # sum = 45 , mean = 45.0 ( 0.0 )
    # non-zero (dose=0)  voxels  = 1 (100.00%) => 0.00 litre
    # non-air (HU>-1000) voxels  = 1 (100.00%) => 0.00 litre
    ################
    >>> ft.arr(pointValue)
    array(45, dtype=int16)
    >>> fredtools.arr(fredtools.getPoint(img3D, point=[0,-212.42,-654.8], interpolation='nearest'))
    array(37, dtype=int16)
    >>> fredtools.arr(fredtools.getPoint(img3D, point=[0,-212.42,-654.8], interpolation='linear'))
    array(45, dtype=int16)
    >>> fredtools.arr(fredtools.getPoint(img3D, point=[0,-212.42,-654.8], interpolation='spline', splineOrder=5))
    array(43, dtype=int16)
    """
    import warnings
    import SimpleITK as sitk
    import fredtools as ft
    import numpy as np

    ft._isSITK(img, raiseError=True)

    # check if img is already a point
    if ft._isSITK_point(img):
        raise TypeError(f"The object '{type(img)}' is already an instance SimpleITK image describing a point.")

    # check if point dimension matches the img dim.
    if (len(point) != img.GetDimension()) and (len(point) != len(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))):
        raise ValueError(
            f"Dimension of 'point' {point} is {len(point)}. The 'img' is of dimension {img.GetDimension()} but is describing a {len(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))}D image. The 'point' should have dimension {len(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))} or {img.GetDimension()}."
        )

    # correct point if needed
    if len(point) < img.GetDimension():
        pointCorr = np.array(img.GetOrigin(), dtype="float64")
        pointCorr[list(ft.ft_imgAnalyse._getAxesNumberNotUnity(img))] = point
        point = pointCorr

    # set interpolator
    interpolator = ft.ft_imgGetSubimg._setSITKInterpolator(interpolation=interpolation, splineOrder=splineOrder)

    # check if point is inside the image
    if not ft.isPointInside(img, point) and raiseWarning:
        warnings.warn(f"Warning: the point {point} is not inside the image extent: {ft.getExtent(img)}.")

    # generate point value
    pointVal = sitk.Resample(
        img,
        size=[1] * img.GetDimension(),
        outputSpacing=img.GetSpacing(),
        outputDirection=img.GetDirection(),
        outputOrigin=[float(x) for x in point],
        interpolator=interpolator,
    )

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Point: ", np.array(point))
        print("# Value: ", ft.arr(pointVal))
        ft.ft_imgAnalyse._displayImageInfo(pointVal)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return pointVal


def getInteg(img, axis="X", displayInfo=False):
    """Get 1D integral profile from image.

    The function calculates a 1D integral profile image along specified `axis`
    from an image defined as SimpleITK image object. The integral profile is
    returned as an instance of a SimpleITK image object of the same dimension
    but describing a profile (the dimension of only one axes is different than one).
    The integral means the sum of the voxel values multiplied by the voxel volume.
    The routine is usefull to calculate an integral depth dose (IDD) distributions.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    axis : str, optional
        Axis to generate the 1D integral profile given as a string. The string
        should have form of one letter from [XYZT] set with +/- signs
        (if no sign provided, then + is assumed). For instance it can be:
        `X`,`Y`,`-Z`, etc. If the minus sign is found, then the
        image is flipped in the following direction.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Instance of a SimpleITK image object describing a profile.

    See Also
    --------
        matplotlib.pyplot.plot: displaying profiles.
        pos: get voxels' centres for axes of size different than one.
        vec: get vector with values for the img describing a profile.

    Examples
    --------
    The example below shows how to generate and display a 1D integral profile
    along 'Y' axis from a 3D image.

    >>> img3D = fredtools.readMHD('imageCT.mhd', displayInfo = True)
    ### readMHD ###
    # dims (xyz) =  [511 415 218]
    # pixel size [mm] =  [0.68359375 0.68359375 1.2       ]
    # origin [mm]     =  [-174.65820312 -354.28710938 -785.6       ]
    # x-spatial voxel centre [mm] =  [  -174.658203,  -173.974609, ...,   173.291016,   173.974609 ]
    # y-spatial voxel centre [mm] =  [  -354.287109,  -353.603516, ...,   -71.962891,   -71.279297 ]
    # z-spatial voxel centre [mm] =  [  -785.600000,  -784.400000, ...,  -526.400000,  -525.200000 ]
    # x-spatial extent [mm] =  [  -175.000000 ,   174.316406 ] =>   349.316406
    # y-spatial extent [mm] =  [  -354.628906 ,   -70.937500 ] =>   283.691406
    # z-spatial extent [mm] =  [  -786.200000 ,  -524.600000 ] =>   261.600000
    # volume = 25924053.15 mm3  =>  25.92 litre
    # data type:  16-bit signed integer
    # range: from  -1024  to  3071
    # sum = -33870013138 , mean = -732.6387321958799 ( 468.4351806663016 )
    # non-zero (dose=0)  voxels  = 46188861 (99.91%) => 25.90 litre
    # non-air (HU>-1000) voxels  = 15065800 (32.59%) => 8.45 litre
    ###############
    >>> in1D = fredtools.getProfile(img3D, axis='Y', displayInfo=True)
    ### getInteg ###
    # Axis: 'Y'
    # dims (xyz) =  [  1 415   1]
    # pixel size [mm] =  [349.31640625   0.68359375 261.6       ]
    # origin [mm]     =  [ 1.46800622e+09 -3.54287109e+02 -7.85000000e+02]
    # x-spatial voxel centre [mm] =  [ 1468006225.000000 ]
    # y-spatial voxel centre [mm] =  [  -354.287109,  -353.603516, ...,   -71.962891,   -71.279297 ]
    # z-spatial voxel centre [mm] =  [  -785.000000 ]
    # x-spatial extent [mm] =  [ 1468006050.341797 , 1468006399.658203 ] =>   349.316406
    # y-spatial extent [mm] =  [  -354.628906 ,   -70.937500 ] =>   283.691406
    # z-spatial extent [mm] =  [  -915.800000 ,  -654.200000 ] =>   261.600000
    # volume = 25924053.15 mm3  =>  25.92 litre
    # data type:  64-bit float
    # range: from  -91384297.26562846  to  -43597831.757814154
    # sum = -27783995152.26668 , mean = -66949385.90907634 ( 17496118.62771702 )
    # non-zero (dose=0)  voxels  = 415 (100.00%) => 25.92 litre
    # non-air (HU>-1000) voxels  = 0 (0.00%) => 0.00 litre
    ################
    >>> matplotlib.pyplot.plot(fredtools.pos(in1D), fredtools.vec(in1D))
    >>> matplotlib.pyplot.xlabel('Y [mm]')
    >>> matplotlib.pyplot.ylabel('Values per unitary volume')
    """
    import re
    import SimpleITK as sitk
    import fredtools as ft
    import numpy as np

    ft._isSITK(img, raiseError=True)

    # check if img is already a profile or integral
    if ft._isSITK_profile(img):
        raise TypeError(f"The object '{type(img)}' is already an instance SimpleITK image describing a profile or integral.")

    # check if axis is in proper format
    axis = axis.upper()
    if not {"X", "Y", "Z", "T", "-", "+"}.issuperset(axis):
        raise ValueError(f"Axis parameter {axis} cannot be recognized. Only letters 'X','Y','Z','T','-','+' are supported.")
    if len(axis) > 2:
        raise ValueError(f"Axis parameter {axis} cannot be recognized. The length of the plane parameter should less or equal to 2.")

    # remove all signs from the plane definition
    axisSimple = re.sub("[-+]", "", axis)

    # determine available axis names based on img dimension
    axesNameAvailable = ["X", "Y", "Z", "T"][0 : img.GetDimension()]

    # check if profile definition is correct for img dimension
    if not axisSimple in axesNameAvailable:
        raise ValueError(f"Axis '{axisSimple}' cannot be recongised for 'img' of dimension {img.GetDimension()}. Only {axesNameAvailable} are possible.")

    # determine axis to accumulate and axis of integral
    if not axisSimple in axesNameAvailable:
        raise ValueError(f"Axis '{axisSimple}' cannot be recongised for 'img' of dimension {img.GetDimension()}. Only {axesNameAvailable} are possible.")
    axesAcc = [i for i, x in enumerate([axisSimple == i for i in axesNameAvailable]) if not x]
    axesInteg = [i for i, x in enumerate([axisSimple == i for i in axesNameAvailable]) if x][0]

    # generate integ profile
    integ = img
    for axisAcc in axesAcc:
        integ = sitk.SumProjection(integ, projectionDimension=axisAcc)
    integ *= np.prod(np.array(img.GetSpacing())[axesAcc])

    # determine and set new origin
    origin = list(ft.getImageCenter(img))
    origin[ft.ft_imgAnalyse._getAxesNumberNotUnity(integ)[0]] = img.GetOrigin()[ft.ft_imgAnalyse._getAxesNumberNotUnity(integ)[0]]
    integ.SetOrigin(origin)

    # flip axes if requested
    if "-" in axis:
        axesFlip = [False] * img.GetDimension()
        axesFlip[axesInteg] = True
        integ = sitk.Flip(integ, flipAxes=axesFlip)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Axis: '{:s}'".format(axis))
        ft.ft_imgAnalyse._displayImageInfo(integ)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return integ
