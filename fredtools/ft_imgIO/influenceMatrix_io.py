def getInmFREDBaseImg(fileName, dtype=float, displayInfo=False):
    """Get base image defined in FRED influence matrix.

    The function reads an influence matrix file produced by the FRED Monte Carlo
    and builds the basic image of a given type, defined as an instance of a SimpleITK image object, 
    with the frame of reference defined in the influence matrix. 

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.
    dtype : data-type, optional
        The desired data-type for the output image, e.g., `numpy.uint32` or `float32`. (def. numpy.float64)        
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image.

    See Also
    --------
        getInmFREDInfo : get information from an influence matrix produced by FRED Monte Carlo.
        getInmFREDPoint : get a vector of interpolated values in a point from an influence matrix produced by FRED Monte Carlo.
        getInmFREDVectorImage : get a vector image from an influence matrix produced by FRED Monte Carlo.
        getInmFREDSumImage : get FRED influence matrix image to a sum SimpleITK image object.
    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk
    import pandas as pd
    import struct

    with open(fileName, "rb") as file_h:
        # get FoR of the Inm image and pencil beam number
        [fileVersion, sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(48))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=4) * 10
        offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=4) * 10
        origin = offset + spacing / 2

        # create empty basic image with FoR defined in Inm
        imgBase = sitk.GetImageFromArray(np.zeros(shape[::-1], dtype=dtype))
        imgBase.SetOrigin(origin)
        imgBase.SetSpacing(spacing)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(imgBase)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgBase


def getInmFREDSumImage(fileName, inmInfo=None, threshold=None, dtype=float, displayInfo=False):
    """Read the FRED influence matrix to sum up the SimpleITK image object.

    The function reads an influence matrix file produced by
    the FRED Monte Carlo to an instance of a SimpleITK image object by summing
    the requested pencil beams with weights if requested. By default,
    all the pencil beams saved to the Inm influence matrix are read with the unitary weights
    for all pencil beams. Still, the user can ask for selected pencil beams providing influence 
    matrix info pandas DataFrame, which must include at least columns 'PBID' and 'FID', and 
    can include column 'weight', which will be used for weight calculation.

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.
    inmInfo : pandas.DataFrame, optional
        A pandas DataFrame with at least columns 'PBID' and 'FID'. (def. None)
    threshold : scalar, array_like or None, optional
        The threshold for which the values are filtered, defined as the fraction 
        of maximum value for each bencim beam. It can be a scalar for a single component
        influence matrix or an iterable of the same size as the number of components. (def. None)
    dtype : data-type, optional
        The desired data-type for the output image, e.g., `numpy.uint32` or `float32`. (def. numpy.float64)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image.

    See Also
    --------
        getInmFREDBaseImg : get base image defined in FRED influence matrix.
        getInmFREDInfo : get information from an influence matrix produced by FRED Monte Carlo.
        getInmFREDPoint : get a vector of interpolated values in a point from an influence matrix produced by FRED Monte Carlo.
        getInmFREDVectorImage : get a vector image from an influence matrix produced by FRED Monte Carlo.
    """
    import struct
    import SimpleITK as sitk
    import numpy as np
    import fredtools as ft
    from collections.abc import Iterable

    fileHeaderSize = 48

    # get number of PBs and FoR saved to Inm
    with open(fileName, "rb") as file_h:
        [fileVersion, sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(fileHeaderSize))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=4) * 10  # [mm]
        offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=4) * 10  # [mm]
        origin = offset + spacing / 2  # [mm]

    # validate threshold
    if threshold:
        if np.isscalar(threshold):
            threshold = tuple([threshold])
        else:
            threshold = tuple(threshold)

        if len(threshold) != componentNo:
            raise AttributeError(f"The influence matrix describes {componentNo} components but threshold is {threshold}. The size of threshold must be the same as the number of components.")

    # get requested pencil beams info
    inmInfoRequested = ft.ft_imgIO.influenceMatrix_io._mergeInmInfo(inmInfo, fileName)

    # create empty vector
    arrVec = [np.zeros(size, dtype="float64") for _ in range(componentNo)]

    with open(fileName, "rb") as file_h:
        file_h.seek(fileHeaderSize, 1)  # skip header
        for _, inmInfoRow in inmInfoRequested.iterrows():
            # jump to PB data file target
            file_h.seek(int(inmInfoRow.PBfileTarget))

            # read voxel indices and values
            voxelsNo = int(inmInfoRow.voxelsNo)
            voxelIndices = np.frombuffer(file_h.read(voxelsNo * 4), dtype="uint32", count=voxelsNo)
            voxelValues = np.frombuffer(file_h.read(voxelsNo * componentNo * 4), dtype="float32", count=voxelsNo*componentNo)

            # filter voxels for low signal
            # voxelsFilter=np.where(voxelValues>=(np.max(voxelValues)*0.1))
            # voxelIndices=
            # add values to array for each component
            for component in range(componentNo):

                voxelValuesComponent = voxelValues[component::componentNo]
                # filter values if requested
                """The filterring is done for values above or equal to a given fraction of the maximum value, defined with the threshold for a given component"""
                if threshold:
                    voxelsFilter = np.where(voxelValuesComponent >= (np.max(voxelValuesComponent)*threshold[component]))
                    voxelValuesComponent = voxelValuesComponent[voxelsFilter]
                    voxelIndices = voxelIndices[voxelsFilter]

                arrVec[component][voxelIndices] += voxelValuesComponent * inmInfoRow.weight  # add values to array

    # reshape each array
    for component in range(componentNo):
        arrVec[component] = np.reshape(arrVec[component], shape, order="F")
    # stack array for each component to single array
    arr = np.stack(arrVec, axis=3)
    arr = arr.astype(dtype)
    # squeeze axis=3 if only one component
    if arr.shape[3] == 1:
        arr = arr[:, :, :, 0]
    arr = np.moveaxis(arr, list(range(len(shape))), list(range(len(shape)))[::-1])
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Number of PBs: ", len(inmInfoRequested))
        print("# Number of fields: ", len(inmInfoRequested.FID.unique()))
        print("# Number of components: ", componentNo)
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return img


def getInmFREDPoint(fileName, point, inmInfo=None, dtype=float, interpolation="linear", raiseMemError=True, displayInfo=False):
    """Get vector of interpolated values in a point from FRED influence matrix.

    The function reads an influence matrix file produced by the FRED Monte Carlo
    and interpolates the signal value for a given point or list of points. By default, the interpolated
    values for all the pencil beams saved to the Inm influence matrix will be calculated with the unitary weights
    for all pencil beams. Still, the user can ask for selected pencil beams providing influence 
    matrix info pandas DataFrame, which must include at least columns 'PBID' and 'FID', and 
    can include column 'weight', which will be used for weight calculation.

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.
    point : Nx3 array_like
        3-element iterable or an N-element iterable of 3-element iterables.
    inmInfo : pandas.DataFrame, optional
        A pandas DataFrame with at least columns 'PBID' and 'FID'. (def. None)        
    dtype : data-type, optional
        The desired data-type for the output image, e.g., `numpy.uint32` or `float32`. (def. numpy.float64)        
    interpolation : {'linear', 'nearest', 'cubic'}, optional
       Determine the interpolation method. (def. 'linear')
    raiseMemError : bool, optional
        Raise an error if the expected size of the calculated array is larger than the available RAM space. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    CxPxN numpy array
        A numpy array of shape CxPxN where C is the component number, P is the number of pencil beam, and N is the number of point.

    See Also
    --------
        getInmFREDBaseImg : get base image defined in FRED influence matrix.
        getInmFREDInfo : get information from an influence matrix produced by FRED Monte Carlo.
        getInmFREDVectorImage : get a vector image from an influence matrix produced by FRED Monte Carlo.
        getInmFREDSumImage : get FRED influence matrix image to a sum SimpleITK image object.

    Notes
    -----
    1. The function exploits the scipy RegularGridInterpolator [1]_ to interpolate the point value.

    2. The function calculates the values in a given point or list of points for each pencil beam and each component
    saved in the Inm influence matrix. In particular, it can be used to get the values for each pencil beam for
    each voxel by supplying the position of the center of each voxel (interpolation "nearest" can be set
    to speed up the calculations). The user should be aware of the memory usage in such cases.

    3. Some vectors (when calculating for multiple points) are often filled with zero 
    for each component. This means that no pencil beam saved in the influence matrix delivered a signal to those voxels. 
    Filtering the function's output for such cases and recording the zero-signal voxels is recommended.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

    Examples
    --------
    The output array is a 3-dimensional array showing the signal for each component, each pencil beam, and each point. 
    The following examples help with understanding the order of dimensions:

    >> out[0, :, :] - get signals for all pencil beams at each point for component 0. It is useful for single-component scorers like dose or Edep.

    >> out[1, :, 2] - get signals from all pencil beams for component 1 and point 2.

    >> out[0, 2, 5] - get signal from the component 0, pencil beam 2 at point 5.

    >> out[0, :, 2].sum()/out[1,:,2].sum() - calculate the ratio of the signals' sum from all pencil beams of component 0 to component 1. 
    In particular, this might be the LETd value at point 2, where the numerator is saved to component 0 and the denominator to component 1.
    """
    import struct
    import fredtools as ft
    import numpy as np
    import psutil
    import SimpleITK as sitk
    from scipy.interpolate import RegularGridInterpolator

    fileHeaderSize = 48

    # convert points to 3xN numpy array and check the shape
    points = np.array(point)
    if points.ndim == 1:
        points = np.expand_dims(points, 0)
    if points.ndim != 2 or points.shape[1] != 3:
        raise TypeError("Parameter 'point' must be a 3-element iterable (for a single point) or an N-element iterable of 3 elements iterables (for N points).")
    pointsNo = points.shape[0]

    # check interpolation
    if interpolation.lower() not in ["nearest", "linear", "cubic"]:
        raise ValueError(f"Interpolation type '{interpolation}' cannot be recognized. Only 'linear', 'nearest' and 'cubic' are supported.")

    # get number of PBs and FoR saved to Inm
    with open(fileName, "rb") as file_h:
        [fileVersion, sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(fileHeaderSize))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]) * 10, decimals=4)  # [mm]
        offset = np.around(np.array([offsetX, offsetY, offsetZ]) * 10, decimals=4)  # [mm]
        origin = offset + spacing / 2  # [mm]

    # check the memory
    if raiseMemError:
        expectedMemorySize = componentNo * pencilBeamNo * pointsNo * np.array(1, dtype=dtype).nbytes  # bytes
        if psutil.virtual_memory().available < (componentNo * pencilBeamNo * pointsNo * np.array(1, dtype=dtype).nbytes):
            raise MemoryError(f"Requested to calculate signal of type {dtype} in {pointsNo} points for {pencilBeamNo} pencil beams for {componentNo} components which is expected to use {(expectedMemorySize/1024**3):.2f} GB of RAM but only {psutil.virtual_memory().available/1024**3:.2f} GB is available.")

    # get requested pencil beams info
    inmInfoRequested = _mergeInmInfo(inmInfo, fileName)

    # create an empty basic image with FoR defined in Inm
    imgBase = sitk.GetImageFromArray(np.zeros(shape[::-1], dtype="uint8"))
    imgBase.SetOrigin(origin)
    imgBase.SetSpacing(spacing)

    # generate scipy RegularGridInterpolator
    voxelCentres = ft.getVoxelCentres(imgBase)
    arrVec = np.zeros(size, dtype="float64")
    rgi = RegularGridInterpolator(voxelCentres, np.reshape(arrVec, shape, order="F"), method=interpolation, bounds_error=False, fill_value=np.nan)

    with open(fileName, "rb") as file_h:
        file_h.seek(fileHeaderSize, 1)  # skip header
        # interpolate point value for BP
        pointValues = []
        for _, inmInfoRow in inmInfoRequested.iterrows():
            # jump to PB data file target
            file_h.seek(int(inmInfoRow.PBfileTarget))

            # read voxel indices and values
            voxelsNo = int(inmInfoRow.voxelsNo)
            voxelIndices = np.frombuffer(file_h.read(voxelsNo * 4), dtype="uint32", count=voxelsNo)
            voxelValues = np.frombuffer(file_h.read(voxelsNo * componentNo * 4), dtype="float32", count=voxelsNo*componentNo)

            pointValuesPB = []
            for component in range(componentNo):
                arrVec = np.zeros(size, dtype="float64")
                arrVec[voxelIndices] = voxelValues[component::componentNo] * inmInfoRow.weight  # replace values in array
                rgi.values = np.reshape(arrVec, shape, order="F")
                pointValuesPB.append(rgi(points).astype(dtype))
            pointValues.append(pointValuesPB)

    pointValues = np.stack(pointValues, axis=1)  # [component, pb, point]

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Number of points: ", pointsNo)
        print("# Number of PBs: ", len(inmInfoRequested))
        print("# Number of fields: ", len(inmInfoRequested.FID.unique()))
        print("# Number of components: ", componentNo)
        print(f"# Number of no-signal points: {(pointValues.sum(axis=0).sum(axis=0)==0).sum()} ({((pointValues.sum(axis=0).sum(axis=0)==0).sum()/pointsNo)*100:.2f}%)")  # no signal in any component
        print(f"# Memory used: {(pencilBeamNo*pointsNo*4/1024**3):.2f} GB")
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return pointValues


def getInmFREDPointSparse(fileName, point, dtype="float64", inmInfo=None, displayInfo=False):
    """Get a sparse array with values in a point from FRED influence matrix.

    The function reads an influence matrix file produced by the FRED Monte Carlo to a list of row-based list 
    of lists sparse arrays, defined as instances of a scipy.sparse.lil_array objects. The list length equals 
    the number of components in the influence matrix file, i.e. each component is read to a separate sparse array.
    By default, all the pencil beams saved to the Inm influence matrix will be read with the unitary weights
    for all pencil beams. Still, the user can ask for selected pencil beams providing influence 
    matrix info pandas DataFrame, which must include at least columns 'PBID' and 'FID', and 
    can include column 'weight', which will be used for weight calculation.

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.
    point : Nx3 array_like
        3-element iterable or an N-element iterable of 3-element iterables describing the points in 
        the physical coordinates. The points do not have to be defined in the voxel centers. However, 
        they will be converted internally to pixel coordinates.
    inmInfo : pandas.DataFrame, optional
        A pandas DataFrame with at least columns 'PBID' and 'FID'. (def. None)        
    dtype : data-type, optional
        The desired data-type for the output sparse array, e.g., `numpy.uint32` or `float32`. (def. 'float64')        
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    list of scipy.sparse.lil_array
        A list of scipy sparse arrays in row-based list of lists format, describing the influence from each 
        pencil beam to a given voxel. The number of rows in each sparse array will equal the number of 
        pencil beams saved in the Inm influence matrix file, and the number of columns will equal 
        the number of points requested.

    See Also
    --------
        getInmFREDBaseImg : get base image defined in FRED influence matrix.
        getInmFREDInfo : get information from an influence matrix produced by FRED Monte Carlo.
        getInmFREDVectorImage : get a vector image from an influence matrix produced by FRED Monte Carlo.
        getInmFREDSumImage : get FRED influence matrix image to a sum SimpleITK image object.

    Notes
    -----
    1. The function exploits the scipy.sparse [2]_ package.

    2. The function does not check the memory occupancy of the sparse arrays, as this depends strictly 
    on the number of requested points. The user is advised to use it carefully as it can quickly 
    fill all the machine's memory when a large influence matrix is loaded.

    References
    ----------
    .. [2] https://docs.scipy.org/doc/scipy/reference/sparse.html

    Examples
    --------
    The output is a list of 2D sparse arrays. The following examples help with understanding the order of slicing:

    >> out[0] - get a sparse matrix of the first component. It is helpful for single-component scorers like dose or Edep.

    >> out[1][:,[2]] - get signals from all pencil beams for component 1 and point 2.

    >> out[5][[7],:] - get signals from the pencil beam no 7 for component 5 in all points.

    >> out[0][[2], [5]] - get signal from the component 0, pencil beam 2 at point 5.
    """
    import struct
    import fredtools as ft
    import numpy as np
    from scipy import sparse

    fileHeaderSize = 48

    # convert points to 3xN numpy array and check the shape
    points = np.array(point)
    if points.ndim == 1:
        points = np.expand_dims(points, 0)
    if points.ndim != 2 or points.shape[1] != 3:
        raise TypeError("Parameter 'point' must be a 3-element iterable (for a single point) or an N-element iterable of 3 elements iterables (for N points).")
    pointsNo = points.shape[0]

    # get number of PBs and FoR saved to Inm
    with open(fileName, "rb") as file_h:
        [fileVersion, sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(fileHeaderSize))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=4) * 10  # [mm]
        offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=4) * 10  # [mm]
        origin = offset + spacing / 2  # [mm]

    # get requested pencil beams info
    inmInfoRequested = ft.ft_imgIO.influenceMatrix_io._mergeInmInfo(inmInfo, fileName)

    # create an empty basic image with FoR defined in Inm
    imgBase = ft.getInmFREDBaseImg(fileName, dtype="uint8")

    # convert physical points to indices
    points = np.round((points-imgBase.GetOrigin())/imgBase.GetSpacing()).astype(int)  # equivalent to transformPhysicalPointToIndex but faster
    indices = np.ravel_multi_index(np.array(points).T, imgBase.GetSize(), order="F")

    with open(fileName, "rb") as file_h:
        file_h.seek(fileHeaderSize, 1)  # skip header
        # list of sparse matrices of point values for each component
        pointValuesComponents = [sparse.lil_array((pencilBeamNo, size), dtype=dtype) for _ in range(componentNo)]

        for PBIdx, inmInfoRow in inmInfoRequested.iterrows():
            # jump to PB data file target
            file_h.seek(int(inmInfoRow.PBfileTarget))

            # get number of voxels to read
            voxelsNo = int(inmInfoRow.voxelsNo)
            # read voxel indices
            voxelIndices = np.frombuffer(file_h.read(voxelsNo * 4), dtype="uint32", count=voxelsNo)

            # generate mask of voxelIndices length with marked positions overlapping with requested indices
            voxelIndicesMask = np.isin(voxelIndices, indices, assume_unique=True)
            # get only voxelIndices that overly with the requested indices
            voxelIndices = voxelIndices[voxelIndicesMask]
            if voxelIndices.size == 0:
                continue

            # read voxel values
            voxelValues = np.frombuffer(file_h.read(voxelsNo * componentNo * 4), dtype="float32", count=voxelsNo*componentNo)

            for component in range(componentNo):
                pointValuesComponents[component][PBIdx, voxelIndices] = voxelValues[component::componentNo][voxelIndicesMask] * inmInfoRow.weight

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Number of points: ", pointsNo)
        print("# Number of PBs: ", len(inmInfoRequested))
        print("# Number of fields: ", len(inmInfoRequested.FID.unique()))
        print("# Number of components: ", componentNo)
        print("# Stored elements per component: ", [inmValues.nnz for inmValues in pointValuesComponents])
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return pointValuesComponents


def getInmFREDInfo(fileName, displayInfo=False):
    """Read basic information from FRED influence matrix.

    The function reads an influence matrix file produced by the FRED Monte Carlo
    and gets the basic information about the pencil beams and fields saved.

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    DataFrame
        Pandas DataFrame with pencil beams and field numbers.

    See Also
    --------
        getInmFREDBaseImg : get base image defined in FRED influence matrix.
        getInmFREDPoint : get a vector of interpolated values in a point from an influence matrix produced by FRED Monte Carlo.
        getInmFREDVectorImage : get a vector image from an influence matrix produced by FRED Monte Carlo.
        getInmFREDSumImage : get FRED influence matrix image to a sum SimpleITK image object.
    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk
    import pandas as pd
    import struct

    with open(fileName, "rb") as file_h:
        # get FoR of the Inm image and pencil beam number
        [fileVersion, sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(48))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=4) * 10
        offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=4) * 10
        origin = offset + spacing / 2

        # create empty basic image with FoR defined in Inm
        imgBase = sitk.GetImageFromArray(np.zeros(shape[::-1], dtype="float32"))
        imgBase.SetOrigin(origin)
        imgBase.SetSpacing(spacing)

        PBIDs = []
        FIDs = []
        voxelsNos = []
        try:
            for _ in range(pencilBeamNo):
                [PBTag, voxelsNo] = struct.unpack("2i", file_h.read(8))
                PBIDs.append(PBTag % 1000000)
                FIDs.append(int(PBTag / 1000000))
                voxelsNos.append(voxelsNo)

                file_h.seek(voxelsNo * 4 * (componentNo+1), 1)  # jump to the next PB
        except:
            raise TypeError('Could not parse the whole structure of the influence matrix.')

    InmInfo = pd.DataFrame({"PBID": PBIDs, "FID": FIDs, "voxelsNo": voxelsNos})

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Influence file version: ", fileVersion/10)
        print("# Number of PBs: ", InmInfo.PBID.size)
        print("# Number of fields: ", InmInfo.FID.unique().size)
        print("# Number of components: ", componentNo)
        print(f"# Number of voxels (min/max/mean): {InmInfo.voxelsNo.min()}/{InmInfo.voxelsNo.max()}/{InmInfo.voxelsNo.mean():.0f}")
        print(f"# Percent of voxels (min/max/mean): {InmInfo.voxelsNo.min()/size*100:.2f}%/{InmInfo.voxelsNo.max()/size*100:.2f}%/{InmInfo.voxelsNo.mean()/size*100:.2f}%")
        print("# FoR of the image:")
        ft.ft_imgAnalyse._displayImageInfo(imgBase)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return InmInfo


def _getInmFREDPBfileTarget(fileName):
    """Read target in FRED influence matrix for each pencil beam.

    The function reads the file target for each pencil beam (PB) produced by the FRED 
    Monte Carlo. The function is a helper in the influence matrix file reading,
    providing with the file starting position of each PB data (excluding the PB tag).

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.

    Returns
    -------
    tuple
        Tupe with the file positions.
    """
    import struct

    fileHeaderSize = 48
    PBHeaderSize = 8

    with open(fileName, "rb") as file_h:

        [_, _, _, _, _, _, _, _, _, _, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(fileHeaderSize))

        PBfileTarget = []
        try:
            for _ in range(pencilBeamNo):
                [_, voxelsNo] = struct.unpack("2i", file_h.read(PBHeaderSize))
                PBfileTarget.append(file_h.tell())
                file_h.seek(voxelsNo * 4 * (componentNo+1), 1)  # jump to the next PB
        except:
            raise TypeError('Could not parse the whole structure of the influence matrix.')

    return tuple(PBfileTarget)


def _mergeInmInfo(inmInfo, fileName):
    """Merge inmInfo with the influence matrix info.

    The function merges the field and pencil beam IDs from the inmInfo defined as a pandas DataFrame
    with the influence matrix info taken from the influence matrix file defined by the fileName. 
    The inmInfo DataFrame must include at least columns 'PBID' and 'FID', and can include column 'weight'
    which will be used for weights calculation. If no 'weight' column is provided, then a unit weight 
    will be used for all requested pencil beams. Additionally, the pencil beam file targets will be calculated.

    Parameters
    ----------
    inmInfo : pandas.DataFrame
        A pandas DataFrame with at least columns 'PBID' and 'FID'.
    fileName : path
        Path to FRED influence matrix file.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with requested pencil beams.
    """
    import fredtools as ft
    import numpy as np
    import pandas as pd
    import copy

    # get influence matrix info for all PBs
    inmInfoAll = ft.getInmFREDInfo(fileName)
    inmInfoAll.set_index(["FID", "PBID"], inplace=True)

    # get PB data file target for all PBs
    inmInfoAll["PBfileTarget"] = _getInmFREDPBfileTarget(fileName)

    # merge influence matrix info for all PBs with the influence matrix provided
    if inmInfo is not None:
        inmInfoInput = copy.copy(inmInfo)

        # convert to DataFrame if Series
        if isinstance(inmInfo, pd.Series):
            inmInfoInput = inmInfoInput.to_frame().T

        # validate if required columns exist in inmInfoInput
        if not {"PBID", "FID"}.issubset(inmInfoInput.columns):
            raise ValueError(f"Missing columns or wrong column names of 'inmInfo'. Must include at least 'PBID' and 'FID'")

        inmInfoInput.set_index(["FID", "PBID"], inplace=True)

        if not np.all(inmInfoInput.index.isin(inmInfoAll.index)):
            raise ValueError(f"Not all pencil beam or field IDs are present in the influence matrix file.")

        if "weight" in inmInfoInput.columns:
            inmInfoAll = pd.concat([inmInfoAll, inmInfoInput["weight"]], axis=1)
        else:
            inmInfoAll.loc[inmInfoInput.index, "weight"] = 1
    else:
        inmInfoAll["weight"] = 1
    inmInfoAll.reset_index(inplace=True)

    # remove all rows for which weights is NaN
    inmInfoAll.dropna(axis=0, inplace=True)

    return inmInfoAll
