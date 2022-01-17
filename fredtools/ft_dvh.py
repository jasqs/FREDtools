def _DVH_DoseLevelVolume(doseLevelArr):
    """Calculate number of voxels greater or equal to doseLevel"""
    import numpy as np

    return doseLevelArr[0], np.sum(doseLevelArr[1] >= doseLevelArr[0])


def getDVH(img, RSfileName, structName, dosePrescribed, doseLevelStep=0.01, resampleImg=None, method="centreInside", algorithm="smparallel", CPUNo="auto", displayInfo=False):
    """Calculate DVH for structure.

    The function calculates a dose-volume histogram (DVH) for voxels inside
    a structure named `structName` and defined in structure dicom file.
    The image can be resampled before mapping to increase the resolution and
    the routine supports multiple CPU computation. The routine exploits and returns
    dicompylercore.dvh.DVH class to hold the DVH. Read more about the dicompyler-core
    DVH module on https://dicompyler-core.readthedocs.io/en/latest/index.html.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK 3D image.
    RSfileName : path
        String path to RS dicom file.
    structName : string
        Name of the structure to calculate the DVH in.
    dosePrescribed : scalar
        Target prescription dose.
    doseLevelStep : scalar, optional
        Size of dose bins. (def. 0.01)
    resampleImg : scalar, array_like or None, optional
        Define if and how to resample the image while mapping the structure.
        Can be a scalar, then the same number will be used for each axis,
        3-element iterable defining the voxel size for each axis, or `None` meaning
        no interpolation. (def. None)
    method: {'centreInside', 'allInside'}, optional
        Method of calculation (def. 'centreInside'):

            -  'centreInside' : map the voxels which are all inside the contour.
            -  'allInside' : map only the centre of the voxels.

    algorithm: {'smparallel', 'matplotlib'}, optional
        Algorithm of calculation (def. 'smparallel'):

            -  'smparallel' : use matplotlib to calculate the voxels inside contours.
            -  'matplotlib' : use multiprocessing sm algorithm to calculate the voxels inside contours.

    CPUNo : {'auto', 'none'}, scalar or None
        Define if the multiprocessing should be used and how many cores should
        be exploited (def. 'auto'). Can be None, then no multiprocessing will be used,
        a string 'auto', then the number of cores will be determined by os.cpu_count(),
        or a scalar defining the number of CPU cored to be used (def. 'auto').
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dicompylercore DVH
        Instance of a dicompylercore.dvh.DVH class holding the DVH.

    See Also
    --------
        dicompylercore: more information about the dicompylercore.dvh.DVH.
        resampleImg: resample image.
        mapStructToImg: map structure to image (refer for more information about mapping algorithms).

    Notes
    -----
    Although, the 'allInside' method seems have more sense, the method 'centreInside' shows
    more consistent results to clinical Treatment Planning System Varian Eclipse 15.6. Nevertheless,
    both methods should, in theory, converge to the same results while the voxel size of the input 'img'
    decreasses. The comparison has been made for different structures of small/large volumes, with holes
    and with detached contours, with the image voxels size down to [0.2, 0.2, 0.2] mm. Consider, that
    decreasing the input image voxel size (for instance by the resampleImg function), its shape
    increases and the memory footprint increases.
    """
    from dicompylercore import dvh
    from multiprocessing import Pool
    import fredtools as ft
    import numpy as np

    ft._isSITK3D(img, raiseError=True)
    if not ft.ft_imgIO.dicom_io._isDicomRS(RSfileName):
        raise ValueError(f"The file {RSfileName} is not a proper dicom describing structures.")

    # get structure info
    structList = ft.getRSInfo(RSfileName)
    if structName not in list(structList.ROIName):
        raise ValueError(f"Cannot find the structure '{structName}' in {RSfileName} dicom file with structures.")
    else:
        struct = structList.loc[structList.ROIName == structName]

    # get number of CPUs to be used for computation
    CPUNo = ft._getCPUNo(CPUNo)

    if resampleImg:
        # check if rescaleImg is in proper format
        if np.isscalar(resampleImg):
            resampleImg = [resampleImg] * img.GetDimension()
        else:
            resampleImg = list(resampleImg)

        if not len(resampleImg) == img.GetDimension():
            raise ValueError(f"Shape of 'spacing' is {resampleImg} but must match the dimension of 'img' {img.GetDimension()} or be a scalar.")

        # resample croped image
        img = ft.resampleImg(img=img, spacing=resampleImg)

    # map structure to resampled img
    imgROI = ft.mapStructToImg(img=img, RSfileName=RSfileName, structName=structName, CPUNo=CPUNo, method=method, algorithm=algorithm)
    # print(ft.arr(imgROI).sum() * np.prod(np.array(imgROI.GetSpacing())) / 1e3)

    # remove values outside ROI
    img = ft.setValueMask(img, imgROI, value=np.nan)
    # crop img to ROI
    img = ft.cropImgToMask(img, imgROI, displayInfo=False)
    # get numpy array from image
    arr = ft.arr(img)

    doseBinsStart = np.arange(0, np.nanmax(arr) + doseLevelStep, doseLevelStep)
    doseLevelArr = list(zip(doseBinsStart, [arr] * len(doseBinsStart)))
    if not CPUNo:
        dvhData = [_DVH_DoseLevelVolume(doseLevelArrSingle) for doseLevelArrSingle in doseLevelArr]
    else:
        with Pool(CPUNo) as p:
            dvhData = p.map(_DVH_DoseLevelVolume, doseLevelArr)

    dvhData = np.array(dvhData, dtype="float")
    # sort by doseLevel
    dvhData = dvhData[np.argsort(dvhData[:, 0])]
    # get doseBins and volumes
    doseBins = dvhData[:, 0]
    volume = dvhData[:, 1]
    # recalculate number of voxels to volume
    volume *= np.prod(np.array(img.GetSpacing()))
    # add ending dose level
    doseBins = np.append(doseBins, doseBins[-1] + doseLevelStep)

    # generate DVH
    dvhROI = dvh.DVH(volume / 1e3, doseBins, rx_dose=dosePrescribed, name=structName, color=struct.ROIColor.values[0])

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Structure name: '{dvhROI.name}'")
        print(f"# Prescribed dose: {dosePrescribed:.3f} Gy")
        print(f"# Volume: {dvhROI.volume:.3f} {dvhROI.volume_units}")
        print(f"# Dose max/min: {dvhROI.max:.3f}/{dvhROI.min:.3f} {dvhROI.dose_units}")
        print(f"# Dose mean: {dvhROI.mean:.3f} {dvhROI.dose_units}")
        print(f"# Absolute HI (D02-D98): {dvhROI.statistic('D02').value-dvhROI.statistic('D98').value:.3f} {dvhROI.dose_units}")
        print(f"# D98: {dvhROI.statistic('D98').value:.3f} {dvhROI.dose_units}")
        print(f"# D50: {dvhROI.statistic('D50').value:.3f} {dvhROI.dose_units}")
        print(f"# D02: {dvhROI.statistic('D02').value:.3f} {dvhROI.dose_units}")
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return dvhROI
