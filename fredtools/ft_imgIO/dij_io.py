def getDijFREDVectorImage(dijFileName, FNo=None, PBNo=None, returnOrder=False, raiseMemError=True, displayInfo=False):
    """Read FRED Dij image to SimpleITK vector image object.

    The function reads a Dij (influence matrix) file produced by
    the FRED Monte Carlo to an instance of a SimpleITK vector image object.
    The default type of the SimpleITK vector image object is "float32". Each
    element of the vectors in voxels is a single pencil beam signal. By default,
    all the pencil beams saved to the Dij influence matrix are read, but the user can
    ask for selected pencil beams (the field will be read automatically) or for
    all pencil beams for a given field or list of fields. See the Notes for more details.
    The order of the vectors is the order of the pencil beam's signal saved in
    the Dij matrix and can alter from the one requested by the user. It is recommended
    to get the true order by `returnOrder=True` option, to return it.
    The size of the SimpleITK vector image object in the memory can be significant,
    therefore memory checking is implemented, and it is recommended to use it.

    Parameters
    ----------
    dijFileName : path
        Path to FRED Dij file to read.
    FNo : scalar or array_like, optional
        The number of field IDs. (def. None)
    PBNo : scalar or array_like, optional
        The number of pencil beams IDs.  (def. None)
    returnOrder : bool, optional
        If True, the true order of the pencil beams and field IDs will be returned,
        along with the SimpleITK vector image. (def. False)
    raiseMemError : bool, optional
        Raise an error if the expected size of the imported array is larger than the available RAM space. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Vector Image (optionally with pandas DataFrame)
        An object of a SimpleITK vector image, or SimpleITK Vector Image along with the pandas DataFtame
        describing the order of the pencil beams and fields.

    See Also
    --------
        getDijFREDInfo : get information from a Dij influence matrix produced by FRED Monte Carlo.
        getDijFREDPoint : get a vector of interpolated values in a point from a Dij influence matrix produced by FRED Monte Carlo.
        getDijFREDSumImage : get FRED Dij image to a sum SimpleITK image object.

    Notes
    -----
    It is possible to request various combinations of pencil beams and fields. For instance:

        - FNo=None, PBNo=2: will get just the pencil beam no 2 and the field ID will be calculated automatically,
        - FNo=None, PBNo=[20,30,40]: will get just the pencil beams no 20, 30 and 40 and the field ID will be calculated automatically,
        - FNo=1, PBNo=None: will get all the pencil beams from field 1,
        - FNo=[3,1], PBNo=None: will get all the pencil beams from fields 1 and 3.

    The binary influence matrix file format had changed from FRED 3.69.3 version. The function has been aligned to this format 
    but will not work for the previous format. Use FREDtools v. 0.7.6 to read the old binary influence matrix file format or 
    contact the FREDtools developers.
    """
    import struct
    import SimpleITK as sitk
    import numpy as np
    import fredtools as ft
    import psutil
    import pandas as pd
    from collections.abc import Iterable

    # get number of PBs and FoR saved to Dij
    with open(dijFileName, "rb") as dijFile_h:
        [sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, pencilBeamNo] = struct.unpack("<3i6f1i", dijFile_h.read(40))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=7) * 10  # [mm]
        offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=7) * 10  # [mm]
        origin = offset + spacing / 2  # [mm]

    # get PBIDs and FIDs
    DijInfo = ft.getDijFREDInfo(dijFileName, displayInfo=False)

    # check if FNo, PBNo are None, if they are iterable and correct for iterable if needed
    if FNo and not isinstance(FNo, Iterable):
        FNo = [FNo]
    if PBNo and not isinstance(PBNo, Iterable):
        PBNo = [PBNo]
    if FNo and PBNo and not len(FNo) == len(PBNo):
        raise ValueError("The parameters 'PBNo' and 'FNo' must be both scalars or iterables of the same length.")

    # use all FIDs and PBIDs if FNo and PBNo are None
    if not PBNo and not FNo:
        PBInfo = DijInfo.copy()
    # get PBInfo if PBNo=None and FNo is provided (all PBs for given fields)
    if not PBNo and FNo:
        PBInfo = DijInfo.loc[DijInfo.FID.isin(FNo)].copy()
        if len(PBInfo) == 0:
            raise ValueError(f"Cannot find any of the FNo={FNo} in the Dij file.")
        for FID in FNo:
            if FID not in PBInfo.FID.tolist():
                raise ValueError(f"Cannot find the FNo={FID} in the Dij file.")
    # get PBInfo if FNo=None and PBNo is provided (selected PBs without specifying the field)
    if not FNo and PBNo:
        PBInfo = DijInfo.loc[DijInfo.PBID.isin(PBNo)].copy()
        if len(PBInfo) == 0:
            raise ValueError(f"Cannot find any of the PBNo={PBNo} in the Dij file.")
        for PBID in PBNo:
            if PBID not in PBInfo.PBID.tolist():
                raise ValueError(f"Cannot find the PBNo={PBID} in the Dij file.")
            else:
                if len(PBInfo.loc[PBInfo.PBID == PBID]) > 1:
                    raise ValueError(
                        f"Ambiguous definition of the PBNo={PBID} in the Dij file. It is defined in more than one field, specifically in FNo={PBInfo.loc[PBInfo.PBID==PBID].FID.tolist()}."
                    )
                if len(PBInfo) < len(PBNo):
                    raise ValueError(f"Some of the PBNo refer to the same PB in the Dij file.")
    # get PBInfo if both FNo and PBNo are provided
    if FNo and PBNo:
        PBInfo = pd.DataFrame({"FID": FNo, "PBID": PBNo})
        for _, row in PBInfo.iterrows():
            if len(DijInfo.loc[(DijInfo.FID == row.FID) & (DijInfo.PBID == row.PBID)]) == 0:
                raise ValueError(f"Cannot find PBNo={row.PBID} for FNo={row.FID} in the Dij file.")

    # get new number of PBs to saved
    pencilBeamNo = len(PBInfo)

    # check the memory
    if raiseMemError:
        if psutil.virtual_memory().available < (size * 8 * pencilBeamNo):
            raise MemoryError(
                f"Requested to save {pencilBeamNo} into a vector image and it is expected to use {(size*4*pencilBeamNo/1024**3):.2f} GB of RAM but only {psutil.virtual_memory().available/1024**3:.2f} GB is available."
            )

    with open(dijFileName, "rb") as dijFile_h:
        [sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, pencilBeamNo] = struct.unpack("<3i6f1i", dijFile_h.read(40))

        arr = []
        FIDs = []
        PBIDs = []
        for pencilBeam_idx in range(pencilBeamNo):
            [PBTag, voxelsNo] = struct.unpack("2i", dijFile_h.read(8))
            PBID = PBTag % 1000000
            FID = int(PBTag / 1000000)
            if len(PBInfo.loc[(PBInfo.FID == FID) & (PBInfo.PBID == PBID)]) == 1:
                FIDs.append(FID)
                PBIDs.append(PBID)

                voxelIndices = np.frombuffer(dijFile_h.read(voxelsNo * 4), dtype="uint32", count=voxelsNo)
                voxelValues = np.frombuffer(dijFile_h.read(voxelsNo * 4), dtype="float32", count=voxelsNo)

                arrVec = np.zeros(size, dtype="float32")
                arrVec[voxelIndices] = voxelValues  # replace values in array
                arr.append(np.reshape(arrVec, shape, order="F"))

            elif len(PBInfo.loc[(PBInfo.FID == FID) & (PBInfo.PBID == PBID)]) > 1:
                raise ValueError(f"Ambiguous definition of the PBNo={PBID} and FNo={FID}")

            else:
                dijFile_h.seek(voxelsNo * 8, 1)

    arr = np.stack(arr)
    arr = np.moveaxis(arr, list(range(arr.ndim)), list(range(arr.ndim))[::-1])
    img = sitk.GetImageFromArray(arr, isVector=True)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Number of PBs: ", len(PBInfo))
        print("# Number of fields: ", len(PBInfo.FID.unique()))
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    if returnOrder:
        order = pd.DataFrame({"PBID": PBIDs, "FID": FIDs})
        return img, order
    else:
        return img


def getDijFREDSumImage(dijFileName, FNo=None, PBNo=None, weight=None, displayInfo=False):
    """Read FRED Dij image to sum SimpleITK image object.

    The function reads a Dij (influence matrix) file produced by
    the FRED Monte Carlo to an instance of a SimpleITK image object by summing
    the requested pencil beams with weights if requested. By default,
    all the pencil beams saved to the Dij influence matrix are read with the unitary weights
    for all pencil beams, but the user can ask for selected pencil beams (the field will
    be read automatically) or for all pencil beams for a given field or list of fields.
    The weights can be provided as a single value for all pencil beams, for each pencil
    beam separately or constant for each field. See the Notes for more details.

    Parameters
    ----------
    dijFileName : path
        Path to FRED Dij file to read.
    FNo : scalar or array_like, optional
        The number of field IDs. (def. None)
    PBNo : scalar or array_like, optional
        The number of pencil beams IDs.  (def. None)
    weight : scalar or array_like, optional
        The weights of pencil beams.  (def. None)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image.

    See Also
    --------
        getDijFREDInfo : get information from a Dij influence matrix produced by FRED Monte Carlo.
        getDijFREDPoint : get a vector of interpolated values in a point from a Dij influence matrix produced by FRED Monte Carlo.
        getDijFREDVectorImage : get a vector image from a Dij influence matrix produced by FRED Monte Carlo.

    Notes
    -----
    It is possible to request various combinations of pencil beams, fields and weights. For instance:

        - FNo=None, PBNo=2, weight=None: will get just the pencil beam no 2 with unitary weight and the field ID will be calculated automatically,
        - FNo=None, PBNo=[20,30,40], weight=None: will get just the pencil beams no 20, 30 and 40 with unitary weights and the field IDs will be calculated automatically,
        - FNo=None, PBNo=[20,30,40], weight=[100,200,300]: will get just the pencil beams no 20, 30 and 40 with weights 100, 200 and 300, respectively and the field IDs will be calculated automatically,
        - FNo=1, PBNo=None, weight=1E9: will get all the pencil beams from field 1 and all the pencil beams will be weighted by 1E9,
        - FNo=[3,1], PBNo=None, weight=[1E9, 1E5]: will get all the pencil beams from fields 1 and 3 and the pencil beams will be weighted by 1E9 and 1E5 for the respective fields,
        - FNo=None, PBNo=None, weight=[1E9, 2E9, ..., 3E9]: will get all the pencil beams saved in the Dij influence matrix and each pencil beam will be weighted by a separate number (the length of `weight` must be the same as the number of the pencil beams in the Dij influence matrix).

    The last example is the most common. The Dij influence matrix is usually calculated for a single primary for each pencil beam.
    In such a case, the weights represent the number of particles to be delivered, calculated from a treatment plan. It has been tested that
    the 3D dose distribution is consistent concerning the field-of-reference and values with the dose distribution calculated directly
    by FRED Monte Carlo.

    The binary influence matrix file format had changed from FRED 3.69.3 version. The function has been aligned to this format 
    but will not work for the previous format. Use FREDtools v. 0.7.6 to read the old binary influence matrix file format or 
    contact the FREDtools developers.
    """
    import struct
    import SimpleITK as sitk
    import numpy as np
    import fredtools as ft
    from collections.abc import Iterable
    import pandas as pd

    # get number of PBs and FoR saved to Dij
    with open(dijFileName, "rb") as dijFile_h:
        [sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, pencilBeamNo] = struct.unpack("<3i6f1i", dijFile_h.read(40))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=4) * 10  # [mm]
        offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=4) * 10  # [mm]
        origin = offset + spacing / 2  # [mm]

    # get PBIDs and FIDs, add unit weight
    DijInfo = ft.getDijFREDInfo(dijFileName, displayInfo=False)
    DijInfo["weight"] = 1

    # check if FNo, PBNo and weight are None, if they are iterable and correct for iterable if needed
    if FNo and not isinstance(FNo, Iterable):
        FNo = [FNo]
    if PBNo and not isinstance(PBNo, Iterable):
        PBNo = [PBNo]
    if weight and not isinstance(weight, Iterable):
        weight = [weight]
    if FNo and PBNo and not len(FNo) == len(PBNo):
        raise ValueError("The parameters 'PBNo' and 'FNo' must be both scalars or iterables of the same length.")

    # use all FIDs and PBIDs if FNo and PBNo are None
    if not PBNo and not FNo:
        PBInfo = DijInfo.copy()
    # get PBInfo if PBNo=None and FNo is provided (all PBs for given fields)
    if not PBNo and FNo:
        PBInfo = DijInfo.loc[DijInfo.FID.isin(FNo)].copy()
        if len(PBInfo) == 0:
            raise ValueError(f"Cannot find any of the FNo={FNo} in the Dij file.")
        for FID in FNo:
            if FID not in PBInfo.FID.tolist():
                raise ValueError(f"Cannot find the FNo={FID} in the Dij file.")
    # get PBInfo if FNo=None and PBNo is provided (selected PBs without specifying the field)
    if not FNo and PBNo:
        PBInfo = DijInfo.loc[DijInfo.PBID.isin(PBNo)].copy()
        if len(PBInfo) == 0:
            raise ValueError(f"Cannot find any of the PBNo={PBNo} in the Dij file.")
        for PBID in PBNo:
            if PBID not in PBInfo.PBID.tolist():
                raise ValueError(f"Cannot find the PBNo={PBID} in the Dij file.")
            else:
                if len(PBInfo.loc[PBInfo.PBID == PBID]) > 1:
                    raise ValueError(
                        f"Ambiguous definition of the PBNo={PBID} in the Dij file. It is defined in more than one field, specifically in FNo={PBInfo.loc[PBInfo.PBID==PBID].FID.tolist()}."
                    )
                if len(PBInfo) < len(PBNo):
                    raise ValueError(f"Some of the PBNo refer to the same PB in the Dij file.")
    # get PBInfo if both FNo and PBNo are provided
    if FNo and PBNo:
        PBInfo = pd.DataFrame({"FID": FNo, "PBID": PBNo})
        PBInfo["weight"] = 1
        for _, row in PBInfo.iterrows():
            if len(DijInfo.loc[(DijInfo.FID == row.FID) & (DijInfo.PBID == row.PBID)]) == 0:
                raise ValueError(f"Cannot find PBNo={row.PBID} for FNo={row.FID} in the Dij file.")
    # fill weights if provided
    if weight:
        # treat weight as a constant weights for all PBs and fields (the same for all requested)
        if len(weight) == 1:
            PBInfo = PBInfo.assign(weight=weight[0])

        if len(weight) > 1:
            # treat weight as weigths for each indivitual PB if both FNo and PBNo are not provided (weight for each PB)
            if not PBNo and not FNo:
                if not len(weight) == len(PBInfo):
                    raise ValueError(f"The parameter 'weight' must be a scalar or iterable of the length equal to the number of all PB saved to Dij file, i.e. {len(PBInfo)} pencil beams.")
                PBInfo.weight = weight
            # treat weight as field weights if PBNo=None and FNo is provided (same weight for a given field)
            if not PBNo and FNo:
                if not len(weight) == len(FNo):
                    raise ValueError("The parameter 'weight' and 'FNo' must be both scalars or iterables of the same length.")
                for i, FID in enumerate(FNo):
                    PBInfo.loc[PBInfo.FID == FID, "weight"] = weight[i]
            # treat weight as PBs weights if FNo=None and PBNo is provided
            if not FNo and PBNo:
                if not len(weight) == len(PBNo):
                    raise ValueError("The parameter 'weight' and 'PBNo' must be both scalars or iterables of the same length.")
                for i, PBID in enumerate(PBNo):
                    PBInfo.loc[PBInfo.PBID == PBID, "weight"] = weight[i]
            # treat weight as weigths for each indivitual PB if both FNo and PBNo are provided
            if FNo and PBNo:
                if not len(weight) == len(PBNo):
                    raise ValueError("The parameter 'weight', 'PBNo' and 'FNo' must be all scalars or iterables of the same length.")
                for i, (FID, PBID) in enumerate(zip(FNo, PBNo)):
                    PBInfo.loc[(PBInfo.FID == FID) & (PBInfo.PBID == PBID), "weight"] = weight[i]

    # get new number of PBs to saved
    pencilBeamNo = len(PBInfo)

    # create empty vector
    arrVec = np.zeros(size, dtype="float64")

    with open(dijFileName, "rb") as dijFile_h:
        [sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, pencilBeamNo] = struct.unpack("<3i6f1i", dijFile_h.read(40))

        for pencilBeam_idx in range(pencilBeamNo):
            [PBTag, voxelsNo] = struct.unpack("2i", dijFile_h.read(8))
            PBID = PBTag % 1000000
            FID = int(PBTag / 1000000)
            if len(PBInfo.loc[(PBInfo.FID == FID) & (PBInfo.PBID == PBID)]) == 1:
                voxelIndices = np.frombuffer(dijFile_h.read(voxelsNo * 4), dtype="uint32", count=voxelsNo)
                voxelValues = np.frombuffer(dijFile_h.read(voxelsNo * 4), dtype="float32", count=voxelsNo)

                weight = PBInfo.loc[(PBInfo.FID == FID) & (PBInfo.PBID == PBID)].weight.values[0]
                arrVec[voxelIndices] += voxelValues * weight  # add values to array

            elif len(PBInfo.loc[(PBInfo.FID == FID) & (PBInfo.PBID == PBID)]) > 1:
                raise ValueError(f"Ambiguous definition of the PBNo={PBID} and FNo={FID}")

            else:
                dijFile_h.seek(voxelsNo * 8, 1)

    arr = np.reshape(arrVec, shape, order="F")
    arr = arr.astype("float32")
    arr = np.moveaxis(arr, list(range(arr.ndim)), list(range(arr.ndim))[::-1])
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Number of PBs: ", len(PBInfo))
        print("# Number of fields: ", len(PBInfo.FID.unique()))
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return img


def getDijFREDPoint(dijFileName, point, interpolation="linear", raiseMemError=True, displayInfo=False):
    """Get vector of interpolated values in a point from FRED Dij.

    The function reads a Dij (influence matrix) file produced by the FRED Monte Carlo
    and interpolates the signal value for a given point or list of points for each single
    pencil beam.

    Parameters
    ----------
    dijFileName : path
        Path to FRED Dij file to read.
    point : Nx3 array_like
        3-element iterable or an N-element iterable of 3-element iterables.
    interpolation : {'linear', 'nearest', 'cubic'}, optional
       Determine the interpolation method. (def. 'linear')
    raiseMemError : bool, optional
        Raise an error if the expected size of the calculated array is larger than the available RAM space. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    NxP numpy array
        A Numpy array of shape NxP where N is the number of points and P is the number of pencil beams.

    See Also
    --------
        getDijFREDInfo : get information from a Dij influence matrix produced by FRED Monte Carlo.
        getDijFREDVectorImage : get a vector image from a Dij influence matrix produced by FRED Monte Carlo.
        getDijFREDSumImage : get FRED Dij image to a sum SimpleITK image object.

    Notes
    -----
    1. The function exploits the scipy RegularGridInterpolator [1]_ to interpolate the point value.

    2. The function calculates the values in a given point or list of points for each pencil beam saved
    in the Dij influence matrix. In particular, it can be used to get the values for each pencil beam for
    each voxel by supplying the position of the center of each voxel (interpolation "nearest" can be set
    to speed up the calculations). The user should be aware of the memory usage in such cases.

    3. It is often that some of the vectors (when calculating for multiple points) are all filled with zero.
    It means that no pencil beam, saved in the Dij influence matrix delivered the dose to that voxels. It is
    recommended filtering the output of the function for such cases and recording the zero-signal voxels.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

    Notes
    -----
    The binary influence matrix file format had changed from FRED 3.69.3 version. The function has been aligned to this format 
    but will not work for the previous format. Use FREDtools v. 0.7.6 to read the old binary influence matrix file format or 
    contact the FREDtools developers.
    """
    import struct
    import fredtools as ft
    import numpy as np
    import psutil
    import SimpleITK as sitk
    from scipy.interpolate import RegularGridInterpolator

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

    # get number of PBs saved to Dij and check the memory
    with open(dijFileName, "rb") as dijFile_h:
        [sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, pencilBeamNo] = struct.unpack("<3i6f1i", dijFile_h.read(40))
    if raiseMemError:
        if psutil.virtual_memory().available < (pencilBeamNo * pointsNo * 4):
            raise MemoryError(
                f"Requested to calculate signal in {pointsNo} points for {pencilBeamNo} pencil beams which is expected to use {(pencilBeamNo*pointsNo*4/1024**3):.2f} GB of RAM but only {psutil.virtual_memory().available/1024**3:.2f} GB is available."
            )

    with open(dijFileName, "rb") as dijFile_h:
        # get FoR of the Dij image and pencil beam number
        [sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, pencilBeamNo] = struct.unpack("<3i6f1i", dijFile_h.read(40))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=7) * 10
        offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=7) * 10
        origin = offset + spacing / 2

        # create empty basic image with FoR defined in Dij
        imgBase = sitk.GetImageFromArray(np.zeros(shape[::-1], dtype="uint8"))
        imgBase.SetOrigin(origin)
        imgBase.SetSpacing(spacing)

        # generate scipy RegularGridInterpolator
        voxelCentres = ft.getVoxelCentres(imgBase)
        arrVec = np.zeros(size, dtype="float32")
        rgi = RegularGridInterpolator(voxelCentres, np.reshape(arrVec, shape, order="F"), method=interpolation, bounds_error=False, fill_value=None)

        # interpolate point value for BP
        pointValues = []
        for pencilBeam_idx in range(pencilBeamNo):
            [PBTag, voxelsNo] = struct.unpack("2i", dijFile_h.read(8))
            voxelIndices = np.frombuffer(dijFile_h.read(voxelsNo * 4), dtype="uint32", count=voxelsNo)
            voxelValues = np.frombuffer(dijFile_h.read(voxelsNo * 4), dtype="float32", count=voxelsNo)

            arrVec = np.zeros(size, dtype="float32")
            arrVec[voxelIndices] = voxelValues  # replace values in array
            rgi.values = np.reshape(arrVec, shape, order="F")
            pointValues.append(rgi(points).astype("float32"))

    pointValues = np.stack(pointValues).T

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Number of points: ", pointsNo)
        print(f"# Number of no-signal points: {(pointValues.sum(axis=1)==0).sum()} ({((pointValues.sum(axis=1)==0).sum()/pointsNo)*100:.2f}%)")
        print("# Number of PBs: ", pencilBeamNo)
        print(f"# Memory used: {(pencilBeamNo*pointsNo*4/1024**3):.2f} GB")
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return pointValues


def getDijFREDInfo(dijFileName, displayInfo=False):
    """Read basic information from FRED Dij

    The function reads a Dij (influence matrix) file produced by the FRED Monte Carlo
    and gets basic information about the pencil beams and fields saved.

    Parameters
    ----------
    dijFileName : path
        Path to FRED Dij file to read.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    DataFrame
        Pandas DataFrame with pencil beams and field numbers.

    See Also
    --------
        getDijFREDPoint : get a vector of interpolated values in a point from a Dij influence matrix produced by FRED Monte Carlo.
        getDijFREDVectorImage : get a vector image from a Dij influence matrix produced by FRED Monte Carlo.
        getDijFREDSumImage : get FRED Dij image to a sum SimpleITK image object.

    Notes
    -----
    The binary influence matrix file format had changed from FRED 3.69.3 version. The function has been aligned to this format 
    but will not work for the previous format. Use FREDtools v. 0.7.6 to read the old binary influence matrix file format or 
    contact the FREDtools developers.
    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk
    import pandas as pd
    import struct

    with open(dijFileName, "rb") as dijFile_h:
        # get FoR of the Dij image and pencil beam number
        [sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, pencilBeamNo] = struct.unpack("<3i6f1i", dijFile_h.read(40))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=7) * 10
        offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=7) * 10
        origin = offset + spacing / 2

        # create empty basic image with FoR defined in Dij
        imgBase = sitk.GetImageFromArray(np.zeros(shape[::-1], dtype="float32"))
        imgBase.SetOrigin(origin)
        imgBase.SetSpacing(spacing)

        PBIDs = []
        FIDs = []
        voxelsNos = []
        for pencilBeam_idx in range(pencilBeamNo):
            [PBTag, voxelsNo] = struct.unpack("2i", dijFile_h.read(8))
            PBIDs.append(PBTag % 1000000)
            FIDs.append(int(PBTag / 1000000))
            voxelsNos.append(voxelsNo)

            dijFile_h.seek(voxelsNo * 8, 1)

    DijInfo = pd.DataFrame({"PBID": PBIDs, "FID": FIDs, "voxelsNo": voxelsNos})

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Number of PBs: ", DijInfo.PBID.size)
        print("# Number of fields: ", DijInfo.FID.unique().size)
        print(f"# Number of voxels (min/max/mean): {DijInfo.voxelsNo.min()}/{DijInfo.voxelsNo.max()}/{DijInfo.voxelsNo.mean():.0f}")
        print(f"# Percent of voxels (min/max/mean): {DijInfo.voxelsNo.min()/size*100:.2f}%/{DijInfo.voxelsNo.max()/size*100:.2f}%/{DijInfo.voxelsNo.mean()/size*100:.2f}%")
        print("# FoR of the image:")
        ft.ft_imgAnalyse._displayImageInfo(imgBase)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return DijInfo
