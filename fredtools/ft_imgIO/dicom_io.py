def _getDicomType_tags(dicomTags):
    """Check the type of the dicom tags.

    The function checks if the `dicomTags` is a dicom of a CT slice (CT),
    an 1D/2D/3D image (RD), a treatment plan (RN) or a structure set (RS).
    If the type could not be determined, 'Unknown' is returned.

    Parameters
    ----------
    dicomTags : dicom tags
        A dicom tag structure read by pydicom.read_file.

    Returns
    -------
    string
        A string that can be `CT`, `RD`, `RN`, `RS`, or `Unknown`.

    See Also
    --------
    _getDicomType_file : check the type a dicom file.
    """
    import pydicom as dicom
    import warnings

    if "SOPClassUID" in dicomTags:  # check if SOPClassUID exists
        if dicomTags.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2":  # CT Image Storage
            return "CT"
        if dicomTags.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3":  # Radiation Therapy Structure Set Storage
            return "RS"
        if dicomTags.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.8":  # Radiation Therapy Ion Plan Storage
            return "RN"
        if dicomTags.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.2":  # Radiation Therapy Dose Storage
            return "RD"
        else:
            warnings.warn("Warning: no 'SOPClassUID'. Could not determine the dicom type.")
            return "Unknown"


def _getDicomType_file(dicomFile):
    """Check the type of the dicom file.

    The function checks if the `dicomFile` is a dicom of a CT slice (CT),
    an 1D/2D/3D image (RD), a treatment plan (RN) or a structure set (RS).
    If the type could not be determined, 'Unknown' is returned.

    Parameters
    ----------
    dicomFile : path
        A string of path to a dicom file

    Returns
    -------
    string
        A string that can be `CT`, `RD`, `RN`, `RS`, or `Unknown`.

    See Also
    --------
    _getDicomType_tags : check the type of dicom tags.
    """
    import pydicom as dicom
    import warnings

    try:
        dicomTags = dicom.read_file(dicomFile)
    except dicom.errors.InvalidDicomError:
        warnings.warn("Warning: could not read file {:s}".format(dicomFile))
        return "Unknown"
    return _getDicomType_tags(dicomTags)


def getDicomType(dicomVar):
    r"""Check the type of the dicom given as a path or tags.

    The function checks if the `dicomVar` is a dicom of a CT slice (CT),
    an 1D/2D/3D image (RD), a treatment plan (RN) or a structure set (RS).
    If the type could not be determined, 'Unknown' is returned.

    Parameters
    ----------
    dicomVar : path or tags
        A string of path to a dicom file or a dicom tag structure read
        by pydicom.read_file.

    Returns
    -------
    string
        A string that can be `CT`, `RD`, `RN`, `RS`, or `Unknown`.

    See Also
    --------
    _getDicomType_tags : check the type of dicom tags.
    _getDicomType_file : check the type a dicom file.
    sortDicoms : sort dicom files in a folder by type.
    """
    import pydicom as dicom

    if isinstance(dicomVar, str):
        return _getDicomType_file(dicomVar)
    elif isinstance(dicomVar, dicom.dataset.FileDataset):
        return _getDicomType_tags(dicomVar)


def sortDicoms(searchFolder, recursive=False, displayInfo=False):
    """Sort dicom file names found in the search folder for CT, RS, RN, RD and Unknown.

    The function sorts file names found in the `searchFolder`
    (and subfolders if requested) for:

        -  CT - dicom files of CT slices
        -  RS - dicom files of structures
        -  RN - dicom files of radiotherapy plan (also called RP)
        -  RD - dicom files of 1D/2D/3D image (for instance dose distribution)
        -  Unknown - files with \*.dcm extension that were not recognised.

    Parameters
    ----------
    searchFolder : path
        The path to be searched.
    recursive : bool, optional
        Search for files recursively (def. False).
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    dict
        Dictionary with the sorted file names.
    """
    import glob
    import os
    import fredtools as ft

    if recursive:
        dicomfileNames = glob.glob(os.path.join(searchFolder, "**/*.dcm"), recursive=True)
    else:
        dicomfileNames = glob.glob(os.path.join(searchFolder, "*.dcm"), recursive=False)

    CTfileNames = []
    RSfileNames = []
    RNfileNames = []
    RDfileNames = []
    UnknownfileNames = []
    for dicomfileName in dicomfileNames:
        if getDicomType(dicomfileName) == "CT":  # CT Image Storage
            CTfileNames.append(dicomfileName)
        if getDicomType(dicomfileName) == "RS":  # Radiation Therapy Structure Set Storage
            RSfileNames.append(dicomfileName)
        if getDicomType(dicomfileName) == "RN":  # Radiation Therapy Ion Plan Storage
            RNfileNames.append(dicomfileName)
        if getDicomType(dicomfileName) == "RD":  # Radiation Therapy Dose Storage
            RDfileNames.append(dicomfileName)
        if getDicomType(dicomfileName) == "Unknown":
            UnknownfileNames.append(dicomfileName)  # unrecognised dicoms
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Found dicoms: {:d} x CT, {:d} x RS, {:d} x RN, {:d} x RD, {:d} x unknown".format(len(CTfileNames), len(RSfileNames), len(RNfileNames), len(RDfileNames), len(UnknownfileNames)))
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return {"CTfileNames": CTfileNames, "RSfileNames": RSfileNames, "RNfileNames": RNfileNames, "RDfileNames": RDfileNames, "Unknown": UnknownfileNames}


def getRNInfo(fileName, displayInfo=False):
    """Get some information from the RN plan from RN dicom file.

    The function retrieves some usefull information from a RN dicom of a treatment plan.
    Following information are saved to a dictionary:

        -  *targetStructName* : name of the structure which the plan was prepared for (dose not work for all RN dicoms).
        -  *numberOfFractionsPlanned* : number of the fractions planned.
        -  *targetPrescriptionDose* : dose prescribed to the target.
        -  *targetPrescriptionDose_fieldDose* : dose prescribed to the target calculated as the sum of each field dose.
        -  *numberOfBeams* : number of the fields.
        -  *fieldsNumber* : order of the fields to be delivered.
        -  *fieldsName* : list of field name for each field.
        -  *fieldsGantryAngle* : list of gantry angle for each field.
        -  *fieldsCouchRotationAngle* : list of couch angle for each field.
        -  *fieldsIsoPosition* : list of the isocentre position (X,Y,Z) for each field.
        -  *fieldsNumberOfRangeShifters* : list of number of range shifters for each field.
        -  *fieldsRangeShifterID* : list of range shifter ID for each field.
        -  *fieldsDose* : list of prescribed dose for each field.
        -  *fieldsMU* : list of prescribed MUs for each field.

    Parameters
    ----------
    fileName : path
        Path to RN dicom file.
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    dict
        Dictionary with the RN treatment plan parameters.
    """
    import numpy as np
    import fredtools as ft
    import pydicom as dicom
    import warnings

    if not getDicomType(fileName) == "RN":
        raise TypeError("The file {:s} is not a RN dicom file.".format(fileName))

    try:
        dicomTags = dicom.read_file(fileName)
    except dicom.errors.InvalidDicomError:
        raise ValueError("Warning: could not read file {:s}".format(fileName))

    # TargetPrescriptionDose
    if "TargetPrescriptionDose" in dicomTags.DoseReferenceSequence[0]:
        targetPrescriptionDose = dicomTags.DoseReferenceSequence[0].TargetPrescriptionDose.real
    else:
        targetPrescriptionDose = np.nan

    # numberOfFractionsPlanned
    if "FractionGroupSequence" in dicomTags:
        if "NumberOfFractionsPlanned" in dicomTags.FractionGroupSequence[0]:
            numberOfFractionsPlanned = dicomTags.FractionGroupSequence[0].NumberOfFractionsPlanned.real

    # targetStructName
    if [0x3267, 0x1000] in dicomTags.DoseReferenceSequence[0]:
        targetStructName = dicomTags.DoseReferenceSequence[0][0x3267, 0x1000].value.decode("utf-8")
    else:
        targetStructName = "unknown"

    fieldsNumber = []
    fieldsName = []
    fieldsGantryAngle = []
    fieldsCouchRotationAngle = []
    fieldsIsoPosition = []
    fieldsNumberOfRangeShifters = []
    fieldsRangeShifterID = []
    for IonBeamSequence in dicomTags.IonBeamSequence:
        if IonBeamSequence.TreatmentDeliveryType == "TREATMENT":
            fieldsNumber.append(IonBeamSequence.BeamNumber.real)
            fieldsName.append(IonBeamSequence.BeamName)
            fieldsGantryAngle.append(IonBeamSequence.IonControlPointSequence[0].GantryAngle.real)
            fieldsCouchRotationAngle.append(IonBeamSequence.IonControlPointSequence[0].PatientSupportAngle.real)
            fieldsIsoPosition.append(np.array(IonBeamSequence.IonControlPointSequence[0].IsocenterPosition))
            fieldsNumberOfRangeShifters.append(IonBeamSequence.NumberOfRangeShifters.real)
            if (IonBeamSequence.NumberOfRangeShifters.real != 0) and ("RangeShifterSequence" in IonBeamSequence):
                fieldsRangeShifterID.append(IonBeamSequence.RangeShifterSequence[0].RangeShifterID)
            else:
                fieldsRangeShifterID.append("")
    numberOfBeams = len(fieldsNumber)

    fieldsDose = []
    fieldsMU = []
    for fieldNumber in fieldsNumber:
        for ReferencedBeamSequence in dicomTags.FractionGroupSequence[0].ReferencedBeamSequence:
            if ReferencedBeamSequence.ReferencedBeamNumber == fieldNumber:
                fieldsDose.append(ReferencedBeamSequence.BeamDose.real)
                fieldsMU.append(ReferencedBeamSequence.BeamMeterset.real)
                continue
    targetPrescriptionDose_fieldDose = np.nansum(fieldsDose) * numberOfFractionsPlanned

    if np.round(targetPrescriptionDose, 3) != np.round(np.nansum(fieldsDose) * numberOfFractionsPlanned, 3):
        if np.nansum(fieldsDose) == 0:
            if displayInfo:
                warnings.warn("Warning: No 'BeamDose' in ReferencedBeamSequence for any field.")
        else:
            if displayInfo:
                warnings.warn("Warning: 'TargetPrescriptionDose' is different from sum of 'BeamDose' in 'FractionGroupSequence'.")

    planInfo = {
        "targetStructName": targetStructName,
        "numberOfFractionsPlanned": numberOfFractionsPlanned,
        "targetPrescriptionDose": targetPrescriptionDose,
        "targetPrescriptionDose_fieldDose": targetPrescriptionDose_fieldDose,
        "numberOfBeams": numberOfBeams,
        "fieldsNumber": fieldsNumber,
        "fieldsName": fieldsName,
        "fieldsGantryAngle": fieldsGantryAngle,
        "fieldsCouchRotationAngle": fieldsCouchRotationAngle,
        "fieldsIsoPosition": fieldsIsoPosition,
        "fieldsNumberOfRangeShifters": fieldsNumberOfRangeShifters,
        "fieldsRangeShifterID": fieldsRangeShifterID,
        "fieldsDose": fieldsDose,
        "fieldsMU": fieldsMU,
    }
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Target structure: '{:s}'".format(planInfo["targetStructName"]))
        print("# Number of fractions: {:d}".format(planInfo["numberOfFractionsPlanned"]))
        print("# Target prescription dose for all fractions: {:.3f} Gy RBE (from DoseReferenceSequence)".format(np.round(planInfo["targetPrescriptionDose"], 3)))
        print("# Target prescription dose for one fraction:  {:.3f} Gy RBE (from DoseReferenceSequence)".format(np.round(planInfo["targetPrescriptionDose"] / planInfo["numberOfFractionsPlanned"], 3)))
        print("# Target prescription dose for all fractions: {:.3f} Gy RBE (sum of BeamDose in FractionGroupSequence)".format(np.round(planInfo["targetPrescriptionDose_fieldDose"], 3)))
        print(
            "# Target prescription dose for one fraction:  {:.3f} Gy RBE (sum of BeamDose in FractionGroupSequence)".format(
                np.round(planInfo["targetPrescriptionDose_fieldDose"] / planInfo["numberOfFractionsPlanned"], 3)
            )
        )
        print("# Number of fields: {:d}".format(planInfo["numberOfBeams"]))
        print("# Order of fields: \t\t", list(planInfo["fieldsNumber"]))
        print("# Number of Range Shifters: \t", list(planInfo["fieldsRangeShifterID"]))
        print("# Fields Gantry Angle: \t\t", list(planInfo["fieldsGantryAngle"]))
        print("# Fields Couch Rotation Angle:\t", list(planInfo["fieldsCouchRotationAngle"]))
        print("# Field dose [Gy RBE]: \t\t", list(np.round(planInfo["fieldsDose"], 3)))
        print("# Field MU: \t\t\t", list(np.round(planInfo["fieldsMU"], 3)))
        print("# Isocentre Positions: \n", np.round(planInfo["fieldsIsoPosition"], 3))
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return planInfo


def getRSInfo(fileName, displayInfo=False):
    """Get some information from the RS structures from RS dicom file.

    The function retrieves some basic information about structures from a RS dicom file.

    Parameters
    ----------
    fileName : path
        Path to RS dicom file.
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    DataFrame
        Pandas DataFrame with structures and properties.
    """
    from pandas import DataFrame
    from dicompylercore import dicomparser
    import fredtools as ft

    rtss = dicomparser.DicomParser(fileName)
    structs = rtss.GetStructures()

    ROITable = DataFrame()

    for struct in structs:
        ROITable = ROITable.append(
            {
                "ID": structs[struct]["id"],
                "type": "unclasified" if not structs[struct]["type"] else structs[struct]["type"],
                "name": structs[struct]["name"],
                "color": structs[struct]["color"],
            },
            ignore_index=True,
        )

    ROITable = ROITable.set_index("ID")

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Found {:d} structures:".format(ROITable.shape[0]))
        print("#", ROITable.groupby("type")["type"].count())
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return ROITable


def getExternalName(fileName, displayInfo=False):
    """Get name of the EXTERNAM structure from RS dicom file.

    The function retrieves the name of the structure of type EXTERNAL from a RS dicom file.
    If more than one structure of type EXTERNAL exists in the RS dicom file, then the first one is returned.

    Parameters
    ----------
    fileName : path
        Path to RS dicom file.
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    string
        String with the name of the structure of type EXTERNAL.

    See Also
    --------
    getRSInfo : getting information about all structures on a RS dicom file.
    """
    import warnings
    import fredtools as ft

    ROIinfo = getRSInfo(fileName)
    externalName = ROIinfo.loc[ROIinfo.type == "EXTERNAL"].name.values

    if len(externalName) == 0:
        raise ValueError(f"No structure of type EXTERNAL is defined in {fileName}")
    elif len(externalName) > 1:
        warnings.warn(f"More than one structure of type EXTERNAL is defined in {fileName}. The first one was returned.")

    externalName = externalName[0]

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# ROI name of type EXTERNAL: '{externalName}'")
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return externalName


def getCT(fileNames, displayInfo=False):
    """Get image from dicom series.

    The function reads a series of dicom files and creates an instance
    of a SimpleITK object. The dicom files should come from the same Series
    and have the same frame of reference.

    Parameters
    ----------
    fileNames : array_like
        An iterable (list, tuple, etc.) of paths to dicoms.
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image of sitk.sitkInt16 (int16) type.

    See Also
    --------
    sortDicoms : get names of dicoms in a folder sorted by the dicom type.
    """
    import numpy as np
    import fredtools as ft
    import pydicom as dicom
    import warnings
    import SimpleITK as sitk

    # check if all files are CT type
    for fileName in fileNames:
        if not ft.getDicomType(fileName) == "CT":
            raise ValueError(f"File {fileName} is not a CT dicom.")

    # read dicoms' tags
    dicomSeries = []
    for fileName in fileNames:
        dicomSeries.append(dicom.read_file(fileName))

    # check if all dicoms have Frame of Reference UID tag
    for fileName, dicomSimple in zip(fileNames, dicomSeries):
        if "FrameOfReferenceUID" not in dicomSimple:
            raise TypeError(f"No 'FrameOfReferenceUID' tag in {fileName}.")

    # check if Frame of Reference UID is the same for all dicoms
    for fileName, dicomSimple in zip(fileNames, dicomSeries):
        if dicomSimple.FrameOfReferenceUID != dicomSeries[0].FrameOfReferenceUID:
            raise TypeError(f"Frame of Reference UID for dicom {fileName} is different than for {fileNames[0]}.")

    # check if all dicoms have Series Instance UID tag
    for fileName, dicomSimple in zip(fileNames, dicomSeries):
        if "SeriesInstanceUID" not in dicomSimple:
            raise TypeError(f"No 'SeriesInstanceUID' tag in {fileName}.")

    # check if Series Instance UID is the same for all dicoms
    for fileName, dicomSimple in zip(fileNames, dicomSeries):
        if dicomSimple.SeriesInstanceUID != dicomSeries[0].SeriesInstanceUID:
            raise TypeError(f"Series Instance UID for dicom {fileName} is different than for {fileNames[0]}.")

    # check if all dicoms have Slice Location tag
    sliceLosationPresent = []
    for fileName, dicomSimple in zip(fileNames, dicomSeries):
        sliceLosationPresent.append("SliceLocation" in dicomSimple)
    if not all(sliceLosationPresent):
        warnings.warn("Warning: All dicom files are of CT type but not all have 'SliceLocation' tag. The last element of 'ImagePositionPatient' tag will be used as the slice location.")

    # get slice location
    if not all(sliceLosationPresent):
        slicesLocation = list(map(lambda dicomSimple: float(dicomSimple.ImagePositionPatient[2]), dicomSeries))
    else:
        slicesLocation = list(map(lambda dicomSimple: float(dicomSimple.SliceLocation), dicomSeries))

    # sort fileNames according to the Slice Location tag
    slicesLocationSorted, fileNamesSorted = zip(*sorted(zip(slicesLocation, fileNames)))

    # check if slice location spacing is constant
    slicesLocationSpacingTolerance = 4  # decimal points (4 means 0.0001 tolerance)
    if np.unique(np.round(np.diff(np.array(slicesLocationSorted)), decimals=slicesLocationSpacingTolerance)).size != 1:
        raise ValueError(f"Slice location spacing is not constant with tolerance {10**-slicesLocationSpacingTolerance:.0E}")

    # read dicom series
    img = sitk.ReadImage(fileNamesSorted, outputPixelType=sitk.sitkInt16)
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return img


def getRD(fileNames, displayInfo=False):
    """Get image from dicom.

    The function reads a single dicom file or an iterable of dicom files
    and creates an instance or tuple of instances of a SimpleITK object.

    Parameters
    ----------
    fileNames : string or array_like
        A path or an iterable (list, tuple, etc.) of paths to dicoms.
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    SimpleITK Image or tuple
        Object or tuple of objects of a SimpleITK image.

    See Also
    --------
    sortDicoms : get names of dicoms in a folder sorted by the dicom type.
    """
    import SimpleITK as sitk
    import pydicom as dicom
    import fredtools as ft

    # if fileName is a single string then make it a single element list
    if isinstance(fileNames, str):
        fileNames = [fileNames]

    imgOut = []
    for fileName in fileNames:
        # read dicom tags
        dicomTag = dicom.read_file(fileName)

        # find dose scaling
        scaling = 1
        if "DoseGridScaling" in dicomTag:
            scaling = dicomTag.DoseGridScaling

        img = sitk.ReadImage(fileName, outputPixelType=sitk.sitkFloat64)

        # scale image values
        img *= scaling

        imgOut.append(img)

        if displayInfo:
            print(f"### {ft._currentFuncName()} ###")
            ft.ft_imgAnalyse._displayImageInfo(img)
            print("#" * len(f"### {ft._currentFuncName()} ###"))

    return imgOut[0] if len(imgOut) == 1 else tuple(imgOut)


def _getStructureContoursByName(RSfileName, structName):
    """Get structure contour and info.

    The function reads a dicom with RS structures and returns the contours
    as a list of numpy.array, as well as the contour info, such as: name,
    index, type and colour

    Parameters
    ----------
    RSfileName : string
        Path String to dicom file with structures (RS file).
    structName : string
        Name of the structure.

    Returns
    -------
    tuple
        A 2-element tuple where the first element is a list of Nx3
        numpy arrays describing X,Y,Z coordinates of the structure
        for each sequence, and the second element is a dictionary
        with the structure info.
    """
    import pydicom as dicom
    import numpy as np
    import fredtools as ft

    if not ft.getDicomType(RSfileName) == "RS":
        raise ValueError(f"The file {RSfileName} is not a proper dicom describing structures.")

    dicomRS = dicom.read_file(RSfileName)

    for StructureSetROISequence in dicomRS.StructureSetROISequence:
        if StructureSetROISequence.ROIName == structName:
            ROINumber = StructureSetROISequence.ROINumber
            break
        else:
            ROINumber = None
    ROIinfo = {"Number": int(StructureSetROISequence.ROINumber), "Name": StructureSetROISequence.ROIName, "GenerationAlgorithm": StructureSetROISequence.ROIGenerationAlgorithm}
    # raise error if no structName found
    if not ROINumber:
        raise ValueError(f"The structure named '{structName}' is not in the {RSfileName}.")

    # get ROI type
    for RTROIObservationsSequence in dicomRS.RTROIObservationsSequence:
        if RTROIObservationsSequence.ReferencedROINumber == ROINumber:
            ROIinfo["Type"] = RTROIObservationsSequence.RTROIInterpretedType

    for ROIContourSequence in dicomRS.ROIContourSequence:
        if ROIContourSequence.ReferencedROINumber == ROINumber:
            ROIinfo["Color"] = ROIContourSequence.ROIDisplayColor
            ContourSequence = ROIContourSequence.ContourSequence
    # get contour as a list of Nx3 numpy array for each contour
    StructureContours = [np.reshape(Contour.ContourData, [len(Contour.ContourData) // 3, 3]) for Contour in ContourSequence]

    return StructureContours, ROIinfo


# get CW (True) or CCW (False) direction for each contour
def _checkContourCWDirection(contour):
    import numpy as np

    """Check if the contour has CW (True) or CCW (False) direction"""
    result = 0.5 * np.array(np.dot(contour[:, 0], np.roll(contour[:, 1], 1)) - np.dot(contour[:, 1], np.roll(contour[:, 0], 1)))
    return result < 0
