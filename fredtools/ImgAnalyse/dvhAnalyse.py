def getDVHMask(img, imgMask, dosePrescribed, doseLevelStep=0.01, displayInfo=False):
    """Calculate DVH for the mask.

    The function calculates a dose-volume histogram (DVH) for voxels inside
    a mask with the same field of reference (FoR). The mask must defined as
    a SimpleITK image object describing a binary or floating mask.
    The routine exploits and returns dicompylercore.dvh.DVH class to hold the DVH.
    Read more about the dicompyler-core DVH module
    on https://dicompyler-core.readthedocs.io/en/latest/index.html.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK 3D image.
    imgMask : SimpleITK Image
        Object of a SimpleITK 3D image describing the binary of floating mask.
    dosePrescribed : scalar
        Target prescription dose.
    doseLevelStep : scalar, optional
        Size of dose bins. (def. 0.01)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dicompylercore DVH
        An instance of a dicompylercore.dvh.DVH class holding the DVH.

    See Also
    --------
        dicompylercore : more information about the dicompylercore.dvh.DVH.
        resampleImg : resample image.
        mapStructToImg : map structure to image (refer for more information about mapping algorithms).
        getDVHStruct : calculate DVH for structure.
    """
    from dicompylercore import dvh
    from multiprocessing import Pool
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk

    ft._isSITK3D(img, raiseError=True)
    ft._isSITK3D(imgMask, raiseError=True)
    ft._isSITK_mask(imgMask, raiseError=True)

    # check FoR matching of img and mask
    if not ft.compareImgFoR(img, imgMask, displayInfo=False):
        raise AttributeError("Both input images, 'img' and 'imgMask' must have the same FoR.")

    # convert images to vectors
    arrImg = sitk.GetArrayViewFromImage(img)
    arrMask = sitk.GetArrayViewFromImage(imgMask)

    # remove all voxels not within mask
    arrMaskValid = arrMask > 0
    arrImg = arrImg[arrMaskValid]
    arrMask = arrMask[arrMaskValid]

    # calculate DVH
    doseBins = np.arange(0, arrImg.max() + doseLevelStep, doseLevelStep)
    volume, doseBins = np.histogram(arrImg, doseBins, weights=arrMask.astype("float"))

    # calculate volume in real units [mm3]
    volume *= np.prod(img.GetSpacing())

    # get color and name
    maskColor = imgMask.GetMetaData("ROIColor") if "ROIColor" in imgMask.GetMetaDataKeys() else [0, 0, 255]
    maskName = imgMask.GetMetaData("ROIName") if "ROIName" in imgMask.GetMetaDataKeys() else "unknown"

    # generate DVH
    dvhMask = dvh.DVH(volume / 1e3, doseBins, rx_dose=dosePrescribed, name=maskName, color=maskColor, dvh_type="differential").cumulative

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Structure name: '{dvhMask.name}'")
        print(f"# Prescribed dose: {dosePrescribed:.3f} Gy")
        print(f"# Volume: {dvhMask.volume:.3f} {dvhMask.volume_units}")
        print(f"# Dose max/min: {dvhMask.max:.3f}/{dvhMask.min:.3f} {dvhMask.dose_units}")
        print(f"# Dose mean: {dvhMask.mean:.3f} {dvhMask.dose_units}")
        print(f"# Absolute HI (D02-D98): {dvhMask.statistic('D02').value-dvhMask.statistic('D98').value:.3f} {dvhMask.dose_units}")
        print(f"# D98: {dvhMask.statistic('D98').value:.3f} {dvhMask.dose_units}")
        print(f"# D50: {dvhMask.statistic('D50').value:.3f} {dvhMask.dose_units}")
        print(f"# D02: {dvhMask.statistic('D02').value:.3f} {dvhMask.dose_units}")
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return dvhMask


def getDVHStruct(img, RSfileName, structName, dosePrescribed, doseLevelStep=0.01, resampleImg=None, CPUNo="auto", displayInfo=False):
    """Calculate DVH for the structure.

    The function calculates a dose-volume histogram (DVH) for voxels inside
    a structure named `structName` and is defined in the structure dicom file.
    The image can be resampled before mapping to increase the resolution. The routine
    exploits and returns dicompylercore.dvh.DVH class to hold the DVH.
    Read more about the dicompyler-core DVH module
    on https://dicompyler-core.readthedocs.io/en/latest/index.html.

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
    CPUNo : {'auto', 'none'}, scalar or None
        Define if the multiprocessing should be used and how many cores should
        be exploited (def. 'auto'). Can be None, then no multiprocessing will be used,
        a string 'auto', then the number of cores will be determined by os.cpu_count(),
        or a scalar defining the number of CPU cores to be used (def. 'auto').
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dicompylercore DVH
        An instance of a dicompylercore.dvh.DVH class holding the DVH.

    See Also
    --------
        dicompylercore : more information about the dicompylercore.dvh.DVH.
        resampleImg : resample image.
        mapStructToImg : map structure to image (refer for more information about mapping algorithms).
        getDVHMask : calculate DVH for a mask.

    Notes
    -----
    The structure mapping is performed in 'floating' mode meaning that the fractional structure
    occupancy is assigned to each voxel of the image. Use `getDVHMask` to use a specially prepared
    mask for the DVH calculation.
    """
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

    # get the number of CPUs to be used for computation
    CPUNo = ft.getCPUNo(CPUNo)

    # resample image if requested
    if resampleImg:
        # check if rescaleImg is in the proper format
        if np.isscalar(resampleImg):
            resampleImg = [resampleImg] * img.GetDimension()
        else:
            resampleImg = list(resampleImg)

        if not len(resampleImg) == img.GetDimension():
            raise ValueError(f"Shape of 'spacing' is {resampleImg} but must match the dimension of 'img' {img.GetDimension()} or be a scalar.")

        # resample image
        img = ft.resampleImg(img=img, spacing=resampleImg)

    # map structure to resampled img generating a floating mask
    imgROI = ft.mapStructToImg(img=img, RSfileName=RSfileName, structName=structName, binaryMask=False, CPUNo=CPUNo, displayInfo=False)

    dvhROI = getDVHMask(img, imgROI, dosePrescribed=dosePrescribed, doseLevelStep=doseLevelStep)

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
