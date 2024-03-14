def convertCTtoWER(img, HU, WER, displayInfo=False):
    """Convert CT map to WER map.

    The function converts a 3D Computed Tomography (CT) map with Houndsfield 
    Unit (HU) values, defined as a SimpleITK image object, to an image with 
    Water-Equivalent Ratio values (WER). The two parameters, `HU` and `WER`, 
    define the HU to WER conversion, whereas the missing HU values are interpolated 
    linearly. 

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image with HU values.
    HU : array_like
        An iterable with HU values. It must be of the same size as WER.
    WER : array_like
        An iterable with WER values. It must be of the same size as HU.        
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object with WER values.

    See Also
    --------
        calcWETfromWER : calculate WET image from WER image for point-like source.
    """
    import fredtools as ft
    import numpy as np
    from scipy.interpolate import make_interp_spline
    import SimpleITK as sitk

    # validate img
    ft._isSITK(img, raiseError=True)

    # validate HU and WET vectors
    if not len(HU) == len(WER):
        raise ValueError(f"The length of 'HU' must be equal to the length of 'WER'.")

    # get CT statistics
    statCT = ft.getStatistics(img)

    # validate HU table against img
    if statCT.GetMinimum() < np.array(HU).min():
        raise ValueError(f"Minimum HU value of the img is {statCT.GetMinimum()} but HU table starts from value {np.array(HU).min()}.")
    if statCT.GetMaximum() > np.array(HU).max():
        raise ValueError(f"Maximum HU value of the img is {statCT.GetMaximum()} but HU table ends with value {np.array(HU).max()}.")

    # prepare HU to WER conversion for each HU value
    HU = np.array(HU)
    WER = np.array(WER)
    HU2WERinterp = make_interp_spline(HU, WER,  k=1)  # linear interpolation
    HU = np.arange(HU.min(), HU.max() + 1, 1)
    WER = HU2WERinterp(HU)

    # recalculate CT to pixel WET (WER)
    imgWER = sitk.GetArrayFromImage(img)
    imgWER = np.vectorize(dict(zip(HU, WER)).__getitem__)(imgWER)
    imgWER = sitk.GetImageFromArray(imgWER)
    imgWER.CopyInformation(img)
    imgWER = sitk.Cast(imgWER, sitk.sitkFloat64)
    ft._copyImgMetaData(img, imgWER)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(imgWER)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgWER


def calcWETfromWER(imgWER, imgMask, SAD, CPUNo="auto", displayInfo=False):
    """Calculate WET image from WER image for point-like source.

    The function calculates Water-Equivalent Thickness (WET) for each voxel of 
    an image defined as a SimpleITK image object containing Water-Equivalent 
    Ratio values (WER), inside a mask, defined as a SimpleITK image object describing
    a mask. The WET values are calculated starting from a virtual source, located 
    at [X,Y]=[0,0] position and with Z position defined with a two-element `SAD`, 
    describing the source point in X and Y. Partucullarly, te WET is calculated for 
    a virtual source, where rays are deflected in X and Y directions in different 
    distances from the isocenter.

    Parameters
    ----------
    imgWER : SimpleITK Image
        An object of a SimpleITK image with WER values.
    imgMask : SimpleITK Image
        An object of a SimpleITK image describing a mask.
    SAD : 2-element array_like
        Z coordinates of the virtual point source for deflection 
        in X and Y directions, respectively.
    CPUNo : {'auto', 'none'}, scalar or None, optional
        Define whether multiprocessing should be used and how many cores should
        be exploited (def. 'auto'). Can be None, then no multiprocessing will be used,
        a string 'auto', then the number of cores will be determined by os.cpu_count(),
        or a scalar defining the number of CPU cores to be used (def. 'auto').        
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An instance of a SimpleITK image object with WET values.

    See Also
    --------
        convertCTtoWER : convert CT map to WER map.
    """
    from multiprocessing import Pool
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk

    # check input images
    ft._isSITK3D(imgWER, raiseError=True)
    ft._isSITK3D(imgMask, raiseError=True)
    ft._isSITK_mask(imgMask, raiseError=True)
    if not ft.compareImgFoR(imgWER, imgMask):
        raise AttributeError("Both 'imgWER' and 'imgMask' must have the same FoR.")

    # check SAD
    if isinstance(SAD, str) and SAD.lower() in ["parallel", "par"]:
        raise ValueError("The parallel beam is has not been yet implemented.")
    if len(SAD) != 2:
        raise AttributeError("The 'SAD' parameter must be a 2-element iterable.")

    # get the number of CPUs to be used for computation
    CPUNo = ft.getCPUNo(CPUNo)

    def linePlaneIntersectionPoint(rayPosition, rayTarget, planesPosition, planeNormal):
        u = rayTarget - rayPosition
        dot = np.dot(planeNormal, u)
        if np.abs(dot) > 0:
            fac = np.sum((planesPosition - rayPosition) * planeNormal, axis=1) / dot
            fac = np.expand_dims(fac, 1)
            return rayPosition + (u * fac)
        return None

    def calcWETRay(rayPosition, rayTarget, imgWER, imgVoxelEdges):

        # calculate ray position at the volume entrance
        volumeEntrance = imgWER.GetOrigin()[2]-imgWER.GetSpacing()[2]/2  # min Z coordinate of the volume extent
        rayEntrance = linePlaneIntersectionPoint(rayPosition, rayTarget, np.array([[0, 0, volumeEntrance]]), np.array([0, 0, 1.]))[0]

        # filter valid voxel edges that can be potentially within the ray
        rayVoxelEdges = []
        rayVoxelEdges.append(imgVoxelEdges[0][np.where((imgVoxelEdges[0] >= np.array([rayEntrance[0], rayTarget[0]]).min()) & (imgVoxelEdges[0] <= np.array([rayEntrance[0], rayTarget[0]]).max()))[0]])
        rayVoxelEdges.append(imgVoxelEdges[1][np.where((imgVoxelEdges[1] >= np.array([rayEntrance[1], rayTarget[1]]).min()) & (imgVoxelEdges[1] <= np.array([rayEntrance[1], rayTarget[1]]).max()))[0]])
        rayVoxelEdges.append(imgVoxelEdges[2][np.where((imgVoxelEdges[2] >= np.array([rayEntrance[2], rayTarget[2]]).min()) & (imgVoxelEdges[2] <= np.array([rayEntrance[2], rayTarget[2]]).max()))[0]])

        # calculate voxel crossings for all axes
        planesNormal = np.identity(3)  # plane normal versors
        rayCrossPoints = []
        for i in range(3):
            if rayVoxelEdges[i].size > 0:
                planesPosition = np.zeros((len(rayVoxelEdges[i]), 3))
                planesPosition[:, i] = rayVoxelEdges[i]
                rayCrossPoints.append(linePlaneIntersectionPoint(rayPosition, rayTarget, planesPosition, planesNormal[i]))
        rayCrossPoints = np.concatenate(rayCrossPoints)

        # add target point to list of crossing points
        rayCrossPoints = np.append(rayCrossPoints, np.expand_dims(rayTarget, 0), axis=0)

        # sort crossing points by Z
        rayCrossPoints = rayCrossPoints[rayCrossPoints[:, 2].argsort()]

        # calculate ray crossing lengths
        rayCrossLengths = np.sqrt(np.sum((rayCrossPoints[1:]-rayCrossPoints[0:-1])**2, axis=1))

        # calculate the mean positions between crossings (it will be inside a given voxel)
        rayMeanPoints = (rayCrossPoints[0:-1]+rayCrossPoints[1:])/2

        # calculate voxels' indices (transform point to index)
        rayVoxelsIndices = ft.transformPhysicalPointToIndex(imgWER, rayMeanPoints)
        rayVoxelsValues = np.array(list(map(imgWER.GetPixel, rayVoxelsIndices)))

        # calculate WET
        WET = np.sum(rayCrossLengths*rayVoxelsValues)

        return WET

    # get all voxel positions inside the mask
    raysTargetIdx = np.array(np.where(ft.arr(imgMask).T)).T
    raysTarget = np.array(ft.transformIndexToPhysicalPoint(imgMask, raysTargetIdx))

    # get voxel edges
    imgVoxelEdges = [np.array(item) for item in ft.getVoxelEdges(imgWER)]

    # calculate ray position at the downstream magnet
    """It is assumed that the upstream magnet is diverging the beam in X direction and the downstream in Y direction"""
    raysPosition = np.zeros((raysTarget.shape[0], 3), dtype=np.float64)
    raysPosition[:, 0] = (SAD[0] - SAD[1]) * raysTarget[:, 0] / (raysTarget[:, 2] + SAD[0])
    raysPosition[:, 2] = -SAD[1]

    # generate an empty image with the same FoR as the imgWER
    imgWET = sitk.GetImageFromArray(np.zeros(imgWER.GetSize()).T)
    imgWET.CopyInformation(imgWER)

    global calcWETRayPool  # make the function global to use it in multiprocessing

    def calcWETRayPool(ray):
        # imgWER and imgVoxelEdges are be shared
        return calcWETRay(ray[0], ray[1], imgWER, imgVoxelEdges)

    # run multiprocessing calculation
    with Pool(CPUNo) as pool:
        WETs = pool.map(calcWETRayPool, zip(raysPosition,  raysTarget))

    del (calcWETRayPool)  # remove the global function

    # assign WET values to given voxel
    for rayTargetIdx, WET in zip(raysTargetIdx, WETs):
        imgWET[rayTargetIdx.tolist()] = WET

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(imgWET)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgWET


def generateIsoLayers(minRange, maxRange, beamParams):
    """Calculate iso-WET layers and corresponding energies.

    The function calculates iso Water-Equivalent Thickness (WET) layers between
    minimum and maximum range based on predefined parameters of the beam. 

    Parameters
    ----------
    minRange : scalar
        Minimum range to calculate layers.
    maxRange : scalar
        Maximum range to calculate layers.
    beamParams : pandas.DataFrame
        Parameters of the beam, i.e. dependance of the 
        beam range and width with nominal energies. Must include 
        at least columns: "nomEnergy", "rangeProx" and "rangeDist".

    Returns
    -------
    pandas.DataFrame
        An instance of pandas.DataFrame object describing the iso WET layers.
    """
    from scipy.interpolate import make_interp_spline
    import pandas as pd

    beamParams = beamParams.reset_index()

    # validate if required columns exist in beamParams
    requiredColumns = {"nomEnergy", "rangeProx", "rangeDist"}
    if not requiredColumns.issubset(beamParams.columns):
        raise ValueError(f"Missing columns or wrong column names in 'beamParams'.")

    # validate nomRanges
    if beamParams.rangeProx.min() > minRange:
        raise ValueError(f"Minimum value in the nomRange is {beamParams.rangeProx.min()} mm, but it was requested to generate iso ranges from range {minRange} mm.")
    if beamParams.rangeDist.max() < maxRange:
        raise ValueError(f"Maximum value in the nomRange is {beamParams.rangeDist.max()} mm, but it was requested to generate iso ranges up to range {maxRange} mm.")

    rangeDist2rangeProx = make_interp_spline(beamParams.rangeDist, beamParams.rangeProx, k=3)
    rangeDist2nomEnergy = make_interp_spline(beamParams.rangeDist, beamParams.nomEnergy, k=3)

    # distribute iso layers
    """
    Calculate consecutive layers between the distal and proximal range, where for each new 
    layer, the distal range is the proximal range for the previous one.
    """
    layersInfo = pd.DataFrame({"rangeProx": [float(rangeDist2rangeProx(maxRange))], "rangeDist": [maxRange]})
    for idx in range(1, 100):
        layersInfo = pd.concat([layersInfo, pd.DataFrame({"rangeProx": [float(rangeDist2rangeProx(layersInfo.iloc[idx-1].rangeProx))], "rangeDist": [layersInfo.iloc[idx-1].rangeProx]})], ignore_index=True)
        if layersInfo.iloc[idx].rangeProx < minRange:
            break

    # interpolate nominal energy for calculated distal ranges
    layersInfo["nomEnergy"] = rangeDist2nomEnergy(layersInfo.rangeDist)

    # interpolate any other additional beam parameters given in beamParams
    for additionalColumn in set(beamParams.columns).difference(requiredColumns):
        interp = make_interp_spline(beamParams.nomEnergy, beamParams[additionalColumn], k=3)
        layersInfo[additionalColumn] = interp(layersInfo.nomEnergy)

    # set index name
    layersInfo.index.name = "layerNo"

    return layersInfo


def calcContours(imgMask, level=0.5, displayInfo=False):
    """Calculate contours from 2D binary mask.

    The function calculates list of contours from a 2D image defined as
    a SimpleITK image object describing a binary or floating mask, along 
    a level value.


    Parameters
    ----------
    imgMask : SimpleITK Image
        Object of a SimpleITK 2D image describing a binary mask.
    level: scalar, optional
        Value along which to find contours in the image. (def. 0.5)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    list of shapely.polygon
        A list of polygons defined as instances of shapely.polygon objects.

    Notes
    -----
    The imgMask must be a 2D image describing a floating or binary mask. 
    If an image is, for instance, a 3D image describing a slice, it must 
    be squeezed to 2D first. It can be done, for instance by slicing the image, 
    e.g. imgMask[:,:,0] for image where the third axis is single sized. 
    """
    import fredtools as ft
    from skimage import measure
    import shapely as sph
    # check if the image is a binary mask describing a slice
    ft._isSITK_mask(imgMask, raiseError=True)
    ft._isSITK2D(imgMask, raiseError=True)

    # get contours
    contours = measure.find_contours(ft.arr(imgMask).T, positive_orientation="low", level=level)

    # convert contours to polygons
    contoursPolygon = []
    for contour in contours:
        contoursPolygon.append(sph.geometry.Polygon(contour))

    # convert polygons in px to image FoR ([mm] by default)
    for idx, contourPolygon in enumerate(contoursPolygon):
        polygonCoordinatesPx = sph.get_coordinates(contourPolygon)
        polygonCoordinatesRW = ft.transformContinuousIndexToPhysicalPoint(imgMask, polygonCoordinatesPx)
        contoursPolygon[idx] = sph.set_coordinates(contourPolygon, polygonCoordinatesRW)

    return contoursPolygon


def convertRayTargetToIsoPlane(rayTarget, SAD):
    """Calculate beam positon in the isocentre plane.

    The function calculates the beam positions in the isocentre plane, 
    based on the target position and distance to the virtual point (SAD). 


    Parameters
    ----------
    rayTarget : 3xN array_like
        Target positions in the format of 3xN iterable.
    SAD : 2-element array_like
        Z coordinates of the virtual point source for deflection 
        in X and Y directions, respectively.

    Returns
    -------
    3xN numpy array
        A 3xN array with the ray position in the isocentre plane.

    Notes
    -----
    The function assumes that the beam goes along +Z direction.
    """
    import numpy as np

    if rayTarget.ndim == 1:
        rayTarget = np.expand_dims(rayTarget, 0)

    # calculate the ray position at the downstream magnet
    """It is assumed that the upstream magnet is diverging the beam in X direction and the downstream in Y direction"""
    rayPosition = np.zeros((rayTarget.shape[0], 3), dtype=np.float64)
    rayPosition[:, 0] = (SAD[0] - SAD[1]) * rayTarget[:, 0] / (rayTarget[:, 2] + SAD[0])
    rayPosition[:, 2] = -SAD[1]

    posX = ((-rayPosition[:, 2] * (rayTarget[:, 0] - rayPosition[:, 0])) / (-rayPosition[:, 2] + rayTarget[:, 2])) + rayPosition[:, 0]
    posY = ((-rayPosition[:, 2] * (rayTarget[:, 1] - rayPosition[:, 1])) / (-rayPosition[:, 2] + rayTarget[:, 2])) + rayPosition[:, 1]
    posZ = np.zeros(len(rayPosition))
    pos = np.squeeze(np.vstack([posX, posY, posZ]).T)

    return pos
