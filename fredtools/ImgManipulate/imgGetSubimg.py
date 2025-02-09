from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def getSlice(img: SITKImage, point: PointLike, plane: str = "XY", displayInfo: bool = False, **kwargs) -> SITKImage:
    """Get 2D slice from image.

    The function calculates a 2D slice image through a specified
    `point` in a specified `plane` from an image defined as a SimpleITK
    image object. The slice is returned as an instance of a SimpleITK image
    object of the same dimension but describing a slice (the dimension
    of only two axes are different than one). The slice through a specified
    point is calculated with a specified `interpolation` type.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    point : array_like
        Point to generate the 2D slice through. It should have the length
        of the image dimension. A warning will be generated if the point is
        not inside the image extent.
    plane : str, optional
        Plane to generate the 2D slice given as a string. The string
        should have the form of two letters from [XYZT] set with +/- signs
        (if no sign is provided, then + is assumed). For instance, it can be:
        `XY`,`ZY`,`-YX`, `Y-T`, etc. If the minus sign is found, then the
        image is flipped in the following direction. The order of the axis is
        important and the output will be generated in this way to be consistent
        with the axes displayed with matplotlib.pyplot.imshow. For instance,
        plane `Z-X` will display Z-axis on X-axis in imshow and Y-axis of
        of imshow will be a reversed X-axis of the image. (def. 'XY')
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)
    **kwargs : interpolation type, optional
        Determine the interpolation method. The following keyword arguments
        are available:
            interpolation : {'linear', 'nearest', 'spline'}
                Determine the interpolation method. (def. 'linear')
            splineOrder : int
                Order of spline interpolation. Must be in range 0-5. (def. 3)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object describing a slice.

    See Also
    --------
        matplotlib.pyplot.imshow : displaying 2D images.
        getExtent : get extent of the image in each direction.
        arr : get squeezed array from image.

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
    import SimpleITK as sitk
    import fredtools as ft
    import numpy as np

    if not (ft._imgTypeChecker.isSITK3D(img) or ft._imgTypeChecker.isSITK4D(img)):
        error = TypeError(f"The object '{type(img)}' is not an instance of a 3D or 4D SimpleITK image.")
        _logger.error(error)
        raise error

    # check if point dimension matches the img dim.
    if len(list(point)) != img.GetDimension():
        error = AttributeError(f"Dimension of 'point' {point} does not match 'img' dimension {img.GetDimension()}.")
        _logger.error(error)
        raise error

    # set interpolator
    interpolator = ft._helper.setSITKInterpolator(**kwargs)

    # check if plane is in proper format
    plane = plane.upper()
    if not {"X", "Y", "Z", "T", "-", "+"}.issuperset(plane):
        error = AttributeError(f"Plane parameter '{plane}' cannot be recognized. Only letters 'X','Y','Z','T','-','+' are supported.")
        _logger.error(error)
        raise error
    if len(plane) > 4:
        error = AttributeError(f"Plane parameter '{plane}' cannot be recognized. The length of the plane parameter should less or equal than 4.")
        _logger.error(error)
        raise error

    # remove all signs from the plane definition
    planeSimple = re.sub("[-+]", "", plane)

    # determine available axis names based on img dimension
    axesNameAvailable = ["X", "Y", "Z", "T"][0: img.GetDimension()]

    # check if plane definition is correct for img dimension
    for planeSimpleAxis in planeSimple:
        if not planeSimpleAxis in axesNameAvailable:
            error = AttributeError(f"Axis '{planeSimpleAxis}' cannot be recongised for 'img' of dimension {img.GetDimension()}. Only {axesNameAvailable} are possible.")
            _logger.error(error)
            raise error

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
    sl = sitk.Resample(img,
                       size=[int(x) for x in slSize],  # type: ignore
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
        if not ft.isPointInside(img, point):
            _logger.warning(f"Warning: the point {point} is not inside the image extent: {ft.getExtent(img)}.")
        _logger.info(f"Getting {plane} slice through point {np.array(point)}" + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(sl))

    return sl


def getProfile(img: SITKImage, point: PointLike, axis: str = "X", displayInfo: bool = False, **kwargs) -> SITKImage:
    """Get 1D profile from image along an axis.

    The function calculates a 1D profile image through a specified
    `point` in a specified `axis` from an image defined as a SimpleITK
    image object. The profile is returned as an instance of a SimpleITK image
    object of the same dimension but describing a profile (the dimension
    of only one axes is different than one). The profile through a specified
    point is calculated with a specified `interpolation` type.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    point : array_like
        Point to generate the 2D slice through. It should have the length
        of the image dimension. A warning will be generated if the point is
        not inside the image extent.
    axis : str, optional
        Axis to generate the 1D profile given as a string. The string
        should have the form of one letter from [XYZT] set with +/- signs
        (if no sign is provided, then + is assumed). For instance, it can be:
        `X`,`Y`,`-Z`, etc. If the minus sign is found, then the
        image is flipped in the following direction.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)
    **kwargs : interpolation type, optional
        Determine the interpolation method. The following keyword arguments
        are available:
            interpolation : {'linear', 'nearest', 'spline'}
                Determine the interpolation method. (def. 'linear')
            splineOrder : int
                Order of spline interpolation. Must be in range 0-5. (def. 3)

    Returns
    -------
    SimpleITK Image        cumsum_img = ft.getCumSum(self.img3D, axis='X', displayInfo=True)
        self.assertEqual(cumsum_img.GetDimension(), 3)
        self.assertEqual(cumsum_img.GetSize(), self.img3D.GetSize())

    --------
        matplotlib.pyplot.plot : displaying profiles.
        pos : get voxels' centers for axes of size different than one.
        vec : get a vector with values for the img describing a profile.

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

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # check if img is already a profile
    if ft._imgTypeChecker.isSITK_profile(img):
        error = TypeError(f"The object '{type(img)}' is already an instance SimpleITK image describing a profile.")
        _logger.error(error)
        raise error

    # check if point dimension matches the img dim.
    point = list(point)
    if (len(point) != img.GetDimension()) and (len(point) != len(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(img))):
        error = AttributeError(f"Dimension of 'point' {point} is {len(point)}. The 'img' is of dimension {img.GetDimension()} but is describing a {len(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(img))}D image. The 'point' should have dimension {len(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(img))} or {img.GetDimension()}.")
        _logger.error(error)
        raise error

    # correct point is needed
    if len(point) < img.GetDimension():
        pointCorr = np.array(img.GetOrigin())
        pointCorr[list(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(img))] = point
        point = list(pointCorr)

    # set interpolator
    interpolator = ft._helper.setSITKInterpolator(**kwargs)

    # check if axis is in proper format
    axis = axis.upper()
    if not {"X", "Y", "Z", "T", "-", "+"}.issuperset(axis):
        error = AttributeError(f"Axis parameter '{axis}' cannot be recognized. Only letters 'X','Y','Z','T','-','+' are supported.")
        _logger.error(error)
        raise error
    if len(axis) > 2:
        error = AttributeError(f"Axis parameter '{axis}' cannot be recognized. The length of the plane parameter should less or equal to 2.")
        _logger.error(error)
        raise error

    # remove all signs from the plane definition
    axisSimple = re.sub("[-+]", "", axis)

    # determine available axis names based on img dimension
    axesNameAvailable = ["X", "Y", "Z", "T"][0: img.GetDimension()]

    # check if profile definition is correct for img dimension
    if not axisSimple in axesNameAvailable:
        error = AttributeError(f"Axis '{axisSimple}' cannot be recongised for 'img' of dimension {img.GetDimension()}. Only {axesNameAvailable} are possible.")
        _logger.error(error)
        raise error

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
    prof = sitk.Resample(img,
                         size=[int(x) for x in prSize],  # type: ignore
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
        if not ft.isPointInside(img, point):
            _logger.warning(f"Warning: the point {point} is not inside the image extent: {ft.getExtent(img)}.")
        _logger.info(f"Getting {axis} profile through point {np.array(point)}" + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(prof))

    return prof


def getPoint(img: SITKImage, point: PointLike, displayInfo: bool = False, **kwargs):
    """Get point value from image.

    The function calculates a point value in a specified `point` from an
    image defined as a SimpleITK image object. The point is returned as an
    instance of a SimpleITK image object of the same dimension but describing
    a point (the dimension of all axes is equal to one). The point value in
    a specified point is calculated with a specified `interpolation` type.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    point : array_like
        Point to generate the value. It should have the length of the image
        dimension. A warning will be generated if the point is not inside
        the image extent.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)
    **kwargs : interpolation type, optional
        Determine the interpolation method. The following keyword arguments
        are available:
            interpolation : {'linear', 'nearest', 'spline'}
                Determine the interpolation method. (def. 'linear')
            splineOrder : int
                Order of spline interpolation. Must be in range 0-5. (def. 3)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object describing a point.

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

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # check if img is already a point
    if ft._imgTypeChecker.isSITK_point(img):
        error = TypeError(f"The object '{type(img)}' is already an instance SimpleITK image describing a point.")
        _logger.error(error)
        raise error

    # check if point dimension matches the img dim.
    point = list(point)
    if (len(point) != img.GetDimension()) and (len(point) != len(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(img))):
        error = AttributeError(f"Dimension of 'point' {point} is {len(point)}. The 'img' is of dimension {img.GetDimension()} but is describing a {len(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(img))}D image. The 'point' should have dimension {len(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(img))} or {img.GetDimension()}.")
        _logger.error(error)
        raise error

    # correct point if needed
    if len(point) < img.GetDimension():
        pointCorr = np.array(img.GetOrigin(), dtype="float64")
        pointCorr[list(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(img))] = point
        point = list(pointCorr)

    # set interpolator
    interpolator = ft._helper.setSITKInterpolator(**kwargs)

    # generate point value
    pointVal = sitk.Resample(img,
                             size=[1] * img.GetDimension(),
                             outputSpacing=img.GetSpacing(),
                             outputDirection=img.GetDirection(),
                             outputOrigin=[float(x) for x in point],
                             interpolator=interpolator,
                             )

    if displayInfo:
        if not ft.isPointInside(img, point):
            _logger.warning(f"Warning: the point {point} is not inside the image extent: {ft.getExtent(img)}.")
        if ft._imgTypeChecker.isSITK_vector(img):
            _logger.info(f"Getting value at point {np.array(point)}: vector of {len(ft.arr(pointVal))} values" + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(pointVal))
            _logger.debug(f"Vector of point values:\n {ft.arr(pointVal)}")
        else:
            _logger.info(f"Getting value at point {np.array(point)}: {ft.arr(pointVal)}" + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(pointVal))

    return pointVal


def getInteg(img: SITKImage, axis: str = "X", displayInfo: bool = False) -> SITKImage:
    """Get 1D integral profile from an image.

    The function calculates a 1D integral profile image along the specified `axis`
    from an image defined as a SimpleITK image object. The integral profile is
    returned as an instance of a SimpleITK image object of the same dimension
    but describing a profile (the dimension of only one axes is different than one).
    The integral means the sum of the voxel values multiplied by the voxel volume.
    The routine is useful to calculate integral depth dose (IDD) distributions.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    axis : str, optional
        Axis to generate the 1D integral profile given as a string. The string
        should have the form of one letter from [XYZT] set with +/- signs
        (if no sign is provided, then + is assumed). For instance, it can be:
        `X`, `Y`, `-Z`, etc. If the minus sign is found, then the
        image is flipped in the following direction. (def. "X")
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Instance of a SimpleITK image object describing a profile.

    See Also
    --------
        matplotlib.pyplot.plot : displaying profiles.
        pos : get voxels' centers for axes of size different than one.
        vec : get a vector with values for the img describing a profile.

    Examples
    --------
    The example below shows how to generate and display a 1D integral profile
    along Y axis from a 3D image.

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

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # check if img is already a profile or integral
    if ft._imgTypeChecker.isSITK_profile(img):
        error = TypeError(f"The object '{type(img)}' is already an instance SimpleITK image describing a profile or integral.")
        _logger.error(error)
        raise error

    # check if axis is in proper format
    axis = axis.upper()
    if not {"X", "Y", "Z", "T", "-", "+"}.issuperset(axis):
        error = AttributeError(f"Axis parameter '{axis}' cannot be recognized. Only letters 'X','Y','Z','T','-','+' are supported.")
        _logger.error(error)
        raise error
    if len(axis) > 2:
        error = AttributeError(f"Axis parameter '{axis}' cannot be recognized. The length of the plane parameter should less or equal to 2.")
        _logger.error(error)
        raise error

    # remove all signs from the plane definition
    axisSimple = re.sub("[-+]", "", axis)

    # determine available axis names based on img dimension
    axesNameAvailable = ["X", "Y", "Z", "T"][0: img.GetDimension()]

    # check if profile definition is correct for img dimension
    if not axisSimple in axesNameAvailable:
        error = AttributeError(f"Axis '{axisSimple}' cannot be recongised for 'img' of dimension {img.GetDimension()}. Only {axesNameAvailable} are possible.")
        _logger.error(error)
        raise error

    # determine axis to accumulate and axis of integral
    if not axisSimple in axesNameAvailable:
        error = AttributeError(f"Axis '{axisSimple}' cannot be recognized for 'img' of dimension {img.GetDimension()}. Only {axesNameAvailable} are possible.")
        _logger.error(error)
        raise error
    axesAcc = [i for i, x in enumerate([axisSimple == i for i in axesNameAvailable]) if not x]
    axesInteg = [i for i, x in enumerate([axisSimple == i for i in axesNameAvailable]) if x][0]

    # generate integ profile
    integ = img
    for axisAcc in axesAcc:
        integ = sitk.SumProjection(integ, projectionDimension=axisAcc)
    integ *= np.prod(np.array(img.GetSpacing())[axesAcc])

    # determine and set new origin
    origin = list(ft.getImageCenter(img))
    origin[ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(integ)[0]] = img.GetOrigin()[ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(integ)[0]]
    integ.SetOrigin(origin)

    # flip axes if requested
    if "-" in axis:
        axesFlip = [False] * img.GetDimension()
        axesFlip[axesInteg] = True
        integ = sitk.Flip(integ, flipAxes=axesFlip)

    if displayInfo:
        _logger.info(f"Getting integral profile along {axis} axis" + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(integ))

    return integ


def getCumSum(img: SITKImage, axis: str = "X", displayInfo: bool = False) -> SITKImage:
    """Get cumulative sum image.

    The function calculates a cumulative sum image along the specified `axis`
    from an image defined as a SimpleITK image object. The cumulative sum image is
    returned as an instance of a SimpleITK image object of the same dimension.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    axis : str, optional
        An axis is used to generate the cumulative sum given as a string. The string
        should have the form of one letter from [XYZT] set. (def. "X")
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Instance of a SimpleITK image object describing a profile.

    See Also
    --------
        getInteg : get 1D integral profile from an image.
    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk

    # validate imgCT
    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # check if axis is in proper format
    axis = axis.upper()
    if not {"X", "Y", "Z", "T"}.issuperset(axis):
        error = AttributeError(f"Axis parameter '{axis}' cannot be recognized. Only letters 'X', 'Y', 'Z', 'T' are supported.")
        _logger.error(error)
        raise error
    if len(axis) > 1:
        error = AttributeError(f"Axis parameter '{axis}' cannot be recognized. The length of the plane parameter should less or equal to 1.")
        _logger.error(error)
        raise error

    # determine available axis names based on img dimension
    axesNameAvailable = ["X", "Y", "Z", "T"][0: img.GetDimension()]

    # check if the profile definition is correct for img dimension
    if not axis in axesNameAvailable:
        error = AttributeError(f"Axis '{axis}' cannot be recongised for 'img' of dimension {img.GetDimension()}. Only {axesNameAvailable} are possible.")
        _logger.error(error)
        raise error

    # determine axis along which to calculate the cumulative sum
    axisxyz = [i for i, x in enumerate([axis.upper() == i for i in axesNameAvailable]) if x][0]  # in xyz convention for simpleITK
    axisijk = [i for i, x in enumerate([axis.upper() == i for i in axesNameAvailable[::-1]]) if x][0]  # in ijk convention for numpy

    # calculate CT WET along axisWET
    arr = sitk.GetArrayFromImage(img)
    arrCumSum = np.cumsum(arr, axis=axisijk)
    imgCumSum = sitk.GetImageFromArray(arrCumSum)
    imgCumSum.CopyInformation(img)
    ft._helper.copyImgMetaData(img, imgCumSum)

    if displayInfo:
        _logger.info(f"Getting cumulative sum along {axis} axis" + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(imgCumSum))

    return imgCumSum
