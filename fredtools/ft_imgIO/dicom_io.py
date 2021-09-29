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
        -  Unknown - files with \*.dcm extension that were not recognized.

    Parameters
    ----------
    searchFolder : path
        The path to be searched.
    recursive : bool, optional
        Search for files recursively. (def. False)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

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
            UnknownfileNames.append(dicomfileName)  # unrecognized dicoms
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Found dicoms: {:d} x CT, {:d} x RS, {:d} x RN, {:d} x RD, {:d} x unknown".format(len(CTfileNames), len(RSfileNames), len(RNfileNames), len(RDfileNames), len(UnknownfileNames)))
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return {"CTfileNames": CTfileNames, "RSfileNames": RSfileNames, "RNfileNames": RNfileNames, "RDfileNames": RDfileNames, "Unknown": UnknownfileNames}


def _getIonBeamDatasetForFieldNumber(fileName, beamNumber):
    """Get IonBeamDataset for field number.

    The function reads a dicom with an RN plan and returns the IonBeamDataset from
    the IonBeamSequence for a given beam number.

    Parameters
    ----------
    fileName : path
        Path string to dicom file with plan (RN file).
    beamNumber : scalar, int
        The number of the beam to get the IonBeamDataset for.

    Returns
    -------
    IonBeamDataset
        Dataset describing the ion beam as an instance of
        a pydicom.dataset.Dataset.
    """
    import numpy as np
    import pydicom as dicom
    import fredtools as ft

    # check if dicom is RN
    if not ft.dicom_io.getDicomType(fileName) == "RN":
        raise TypeError("The file {:s} is not a RN dicom file.".format(fileName))

    # check if the beamNumber is an integer scalar
    if not np.isscalar(beamNumber) or not isinstance(beamNumber, int):
        raise TypeError(f"The value of 'beamNumber' must be a scalar integer.")

    # read dicom
    dicomTags = dicom.read_file(fileName)

    # check if IonBeamSequence exists
    if not "IonBeamSequence" in dicomTags:
        raise ValueError(f"Can not find 'IonBeamSequence' in the dicom.")

    for IonBeamDataset in dicomTags.IonBeamSequence:
        if int(IonBeamDataset.BeamNumber) == int(beamNumber):
            return IonBeamDataset
    return None


def _getReferencedBeamDatasetForFieldNumber(fileName, beamNumber):
    """Get ReferencedBeamDataset for field number.

    The function reads a dicom with an RN plan and returns the ReferencedBeamDataset from
    the ReferencedBeamSequence of the FractionGroupSequence[0] for a given beam number.

    Parameters
    ----------
    fileName : path
        Path string to dicom file with plan (RN file).
    beamNumber : scalar, int
        The number of the beam to get the IonBeamDataset for.

    Returns
    -------
    IonBeamDataset
        Dataset describing the referenced beam as an instance of
        a pydicom.dataset.Dataset.
    """
    import numpy as np
    import pydicom as dicom
    import fredtools as ft

    # check if dicom is RN
    if not ft.dicom_io.getDicomType(fileName) == "RN":
        raise TypeError("The file {:s} is not a RN dicom file.".format(fileName))

    # check if the beamNumber is an integer scalar
    if not np.isscalar(beamNumber) or not isinstance(beamNumber, int):
        raise TypeError(f"The value of 'beamNumber' must be a scalar integer.")

    # read dicom
    dicomTags = dicom.read_file(fileName)

    # check if FractionGroupSequence exists
    if not "FractionGroupSequence" in dicomTags:
        raise ValueError(f"Can not find 'FractionGroupSequence' in the dicom.")

    # check if ReferencedBeamSequence exists
    if not "ReferencedBeamSequence" in dicomTags.FractionGroupSequence[0]:
        raise ValueError(f"Can not find 'ReferencedBeamSequence' in the dicom.")

    for ReferencedBeamDataset in dicomTags.FractionGroupSequence[0].ReferencedBeamSequence:
        if int(ReferencedBeamDataset.ReferencedBeamNumber) == int(beamNumber):
            return ReferencedBeamDataset
    return None


def getRNInfo(fileName, displayInfo=False):
    """Get some information from the RN plan from RN dicom file.

    The function retrieves some usefull information from a RN dicom of a treatment plan.
    Following information are saved to a dictionary:

        -  *targetStructName* : name of the structure which the plan was prepared for (dose not work for all RN dicoms).
        -  *fractionsNo* : number of the fractions planned.
        -  *dosePrescribed* : dose prescribed to the target read from DoseReferenceSequence.
        -  *patientName* : name of the patient (usually empty string for anonymized DICOM)
        -  *patientBirthDate* : birth date of the patient (usually empty string for anonymized DICOM)
        -  *patientID* : ID of the patient (often empty string for anonymized DICOM)
        -  *planLabel* : name of the treatment plan.
        -  *planDate* : date of the plan creation.
        -  *planTime* : time of the plan creation.
        -  *treatmentMachineName* : name of the machine that the plan has been prepared for.
        -  *totalFieldsNumber* : total number of fields including setup, treatment and other fields.
        -  *treatmentFieldsNumber* : total number of treatment fields.
        -  *setupFieldsNumber* : total number of setup fields.
        -  *otherFieldsNumber* : total number of other fields.
        -  *manufacturer* : manufacturer of the treatment planning system.
        -  *softwareVersions* : version of the treatment planning system.
        -  *stationName* : name of the station on which the plan has been prepared.

    Parameters
    ----------
    fileName : path
        Path to RN dicom file.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dict
        Dictionary with the RN treatment plan parameters.

    See Also
    --------
    getRN : get parameters of the RN plan from RN dicom file.
    """
    import numpy as np
    import fredtools as ft
    import pydicom as dicom
    import warnings

    # check if dicom is RN
    if not ft.dicom_io.getDicomType(fileName) == "RN":
        raise TypeError("The file {:s} is not a RN dicom file.".format(fileName))

    # read dicom
    dicomTags = dicom.read_file(fileName)

    # prepare plan info
    planInfo = {}

    # get prescribed dose from TargetPrescriptionDose tag of DoseReferenceSequence
    if "TargetPrescriptionDose" in dicomTags.DoseReferenceSequence[0]:
        planInfo["dosePrescribed"] = dicomTags.DoseReferenceSequence[0].TargetPrescriptionDose.real
    else:
        planInfo["dosePrescribed"] = np.nan

    # get number of fractions from NumberOfFractionsPlanned tag of FractionGroupSequence
    if ("FractionGroupSequence" in dicomTags) and ("NumberOfFractionsPlanned" in dicomTags.FractionGroupSequence[0]):
        planInfo["fractionsNo"] = dicomTags.FractionGroupSequence[0].NumberOfFractionsPlanned.real
    else:
        planInfo["fractionsNo"] = np.nan

    # get target struct name from provate tag of DoseReferenceSequence
    if ("DoseReferenceSequence" in dicomTags) and ([0x3267, 0x1000] in dicomTags.DoseReferenceSequence[0]):
        planInfo["targetStructName"] = dicomTags.DoseReferenceSequence[0][0x3267, 0x1000].value.decode("utf-8")
    else:
        planInfo["targetStructName"] = "unknown"

    # get other plan info
    planInfo["planLabel"] = dicomTags.RTPlanLabel if "RTPlanLabel" in dicomTags else ""
    planInfo["planDate"] = dicomTags.RTPlanDate if "RTPlanDate" in dicomTags else ""
    planInfo["planTime"] = dicomTags.RTPlanTime if "RTPlanTime" in dicomTags else ""
    planInfo["patientName"] = str(dicomTags.PatientName) if "PatientName" in dicomTags else ""
    planInfo["patientBirthDate"] = dicomTags.PatientBirthDate if "PatientBirthDate" in dicomTags else ""
    planInfo["patientID"] = dicomTags.PatientID if "PatientID" in dicomTags else ""
    planInfo["manufacturer"] = dicomTags.Manufacturer if "Manufacturer" in dicomTags else ""
    planInfo["softwareVersions"] = dicomTags.SoftwareVersions if "SoftwareVersions" in dicomTags else ""
    planInfo["stationName"] = dicomTags.StationName if "StationName" in dicomTags else ""

    # check if IonBeamSequence exists
    if not "IonBeamSequence" in dicomTags:
        raise ValueError(f"Can not find 'IonBeamSequence' in the dicom.")

    # count fields' type and treatment machine name
    planInfo["totalFieldsNumber"] = int(dicomTags.FractionGroupSequence[0].NumberOfBeams)
    planInfo["treatmentFieldsNumber"] = 0
    planInfo["setupFieldsNumber"] = 0
    planInfo["otherFieldsNumber"] = 0
    for ifield in range(planInfo["totalFieldsNumber"]):
        if dicomTags.IonBeamSequence[ifield].TreatmentDeliveryType == "TREATMENT":
            planInfo["treatmentFieldsNumber"] += 1
            planInfo["treatmentMachineName"] = dicomTags.IonBeamSequence[ifield].TreatmentMachineName
        elif dicomTags.IonBeamSequence[ifield].TreatmentDeliveryType == "SETUP":
            planInfo["setupFieldsNumber"] += 1
        else:
            planInfo["otherFieldsNumber"] += 1

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Patient name:     '{:s}'".format(planInfo["patientName"]))
        print("# Plan label:       '{:s}'".format(planInfo["planLabel"]))
        print("# Plan date:        '{:s}'".format(planInfo["planDate"]))
        print("# Machine name:     '{:s}'".format(planInfo["treatmentMachineName"]))
        print("# Target structure: '{:s}'".format(planInfo["targetStructName"]))
        print("# Number of fractions: {:d}".format(planInfo["fractionsNo"]))
        print("# Dose pres. (all fractions):    {:.3f} Gy RBE".format(np.round(planInfo["dosePrescribed"], 3)))
        print("# Dose pres. (single fraction):  {:.3f} Gy RBE".format(np.round(planInfo["dosePrescribed"] / planInfo["fractionsNo"], 3)))
        print("# Number of treatment fields: {:d}".format(planInfo["treatmentFieldsNumber"]))
        print("# Number of setup fields:     {:d}".format(planInfo["setupFieldsNumber"]))
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return planInfo


def getRN(fileName, raiseWarning=True, displayInfo=False):
    """Get parameters of the RN plan from RN dicom file.

    The function reads an RN dicom file and collects general
    information of the plan, information of each treatment field
    and information about each scanning pencil beam. The general
    information about the plan is hold in a dictionary, is extended
    with comparison to the results of the getRNInfo function
    (cf. See Also section), and contains following keys:

        -  *targetStructName* : name of the structure which the plan was prepared for (dose not work for all RN dicoms).
        -  *fractionsNo* : number of the fractions planned.
        -  *dosePrescribed* : dose prescribed to the target read from DoseReferenceSequence.
        -  *patientName* : name of the patient (usually empty string for anonymized DICOM)
        -  *patientBirthDate* : birth date of the patient (usually empty string for anonymized DICOM)
        -  *patientID* : ID of the patient (often empty string for anonymized DICOM)
        -  *planLabel* : name of the treatment plan.
        -  *planDate* : date of the plan creation.
        -  *planTime* : time of the plan creation.
        -  *treatmentMachineName* : name of the machine that the plan has been prepared for.
        -  *totalFieldsNumber* : total number of fields including setup, treatment and other fields.
        -  *treatmentFieldsNumber* : total number of treatment fields.
        -  *setupFieldsNumber* : total number of setup fields.
        -  *otherFieldsNumber* : total number of other fields.
        -  *manufacturer* : manufacturer of the treatment planning system.
        -  *softwareVersions* : version of the treatment planning system.
        -  *stationName* : name of the station on which the plan has been prepared.
        -  *isocentrePos* : position of the plan isocentre for the first field (if it is not the same for all fields, then a warning is raised)

    Information about the fields is hold in a pandas DataFrame where
    the rows are in order of the fields delivery. It is assumed here that
    the order of the fields delivery is the same as the order of the fields
    in ReferencedBeamSequence of FractionGroupSequence.

    Information about each scanning pencil beam is hold in a pandas DataFrame.
    The scanning pencil beam information is collected in rows but does not
    necessary in a delivery order. Usually the delivery order of the spots
    for each energy layer is defined by the machine during beam delivery
    and this information can be only taken from the machine log files.

    Parameters
    ----------
    fileName : path
        Path to RN dicom file.
    raiseWarning : bool, optional
        Raise warnings. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dict
        Dictionary with the RN treatment plan parameters.
    fieldsInfo
        DataFrame with parameters for each field.
    fieldsSpotsInfo
        DataFrame with parameters of each scanning beam.

    See Also
    --------
    getRNInfo : get some information from the RN plan from RN dicom file.
    """
    import numpy as np
    import fredtools as ft
    import pandas as pd
    import pydicom as dicom
    import warnings

    # check if dicom is RN
    if not ft.dicom_io.getDicomType(fileName) == "RN":
        raise TypeError("The file {:s} is not a RN dicom file.".format(fileName))

    # read dicom
    dicomTags = dicom.read_file(fileName)

    # get general plan info
    planInfo = ft.getRNInfo(fileName, displayInfo=False)

    ### get fields information in order of delivery
    """It is assumed that the order of fields in ReferencedBeamSequence of FractionGroupSequence
    is the order of delivery."""
    # prepare fields info
    fieldsInfo = {}
    fieldsInfo["deliveryNo"] = []
    fieldsInfo["fieldNo"] = []
    fieldsInfo["fieldName"] = []
    fieldsInfo["gantryAngle"] = []
    fieldsInfo["couchAngle"] = []
    fieldsInfo["isocenterPos"] = []
    fieldsInfo["RSsID"] = []
    fieldsInfo["snoutPos"] = []
    fieldsInfo["dose"] = []
    fieldsInfo["MU"] = []
    fieldsInfo["cumMsW"] = []  # Final Cumulative Meterset Weight
    fieldsInfo["energyNo"] = []  #  Number of Control Points / 2
    fieldsInfo["minEnergy"] = []
    fieldsInfo["maxEnergy"] = []
    fieldsInfo["spotsNo"] = []
    fieldsInfo["nomRange"] = []  #  (0x300b, 0x1004)
    fieldsInfo["nomSOBPWidth"] = []  #  (0x300b, 0x100e)
    fieldsInfo["couchPitchAngle"] = []
    fieldsInfo["couchRollAngle"] = []
    fieldsInfo["supportID"] = []
    fieldsInfo["dosePosition"] = []
    fieldsInfo["magnetToIsoDist"] = []  #  Virtual Source-Axis Distances
    treatmentMachineNames = []
    for deliveryNo, ReferencedBeamDataset in enumerate(dicomTags.FractionGroupSequence[0].ReferencedBeamSequence, start=1):
        IonBeamDataset = _getIonBeamDatasetForFieldNumber(fileName, int(ReferencedBeamDataset.ReferencedBeamNumber))
        # Continue if couldn't find IonBeamDataset for the IonBeamDataset or the Treatment Delivery Type of the IonBeamDataset is not TREATMENT
        if not IonBeamDataset or not (IonBeamDataset.TreatmentDeliveryType == "TREATMENT"):
            continue
        fieldsInfo["deliveryNo"].append(deliveryNo)
        fieldsInfo["fieldNo"].append(IonBeamDataset.BeamNumber.real)
        fieldsInfo["fieldName"].append(IonBeamDataset.BeamName)
        fieldsInfo["gantryAngle"].append(IonBeamDataset.IonControlPointSequence[0].GantryAngle.real)
        fieldsInfo["couchAngle"].append(IonBeamDataset.IonControlPointSequence[0].PatientSupportAngle.real)
        fieldsInfo["couchPitchAngle"].append(IonBeamDataset.IonControlPointSequence[0].TableTopPitchAngle.real)
        fieldsInfo["couchRollAngle"].append(IonBeamDataset.IonControlPointSequence[0].TableTopRollAngle.real)
        fieldsInfo["isocenterPos"].append(np.array(IonBeamDataset.IonControlPointSequence[0].IsocenterPosition).tolist())
        if (IonBeamDataset.NumberOfRangeShifters.real != 0) and ("RangeShifterSequence" in IonBeamDataset):
            fieldsInfo["RSsID"].append(IonBeamDataset.RangeShifterSequence[0].RangeShifterID)
        else:
            fieldsInfo["RSsID"].append("")
        fieldsInfo["snoutPos"].append(IonBeamDataset.IonControlPointSequence[0].SnoutPosition.real)
        if "BeamDose" in ReferencedBeamDataset:
            fieldsInfo["dose"].append(ReferencedBeamDataset.BeamDose.real)
        else:
            fieldsInfo["dose"].append(np.nan)
        fieldsInfo["MU"].append(ReferencedBeamDataset.BeamMeterset.real)
        fieldsInfo["cumMsW"].append(IonBeamDataset.FinalCumulativeMetersetWeight.real)
        if "BeamDoseSpecificationPoint" in ReferencedBeamDataset:
            fieldsInfo["dosePosition"].append(np.array(ReferencedBeamDataset.BeamDoseSpecificationPoint).tolist())
        else:
            fieldsInfo["dosePosition"].append(np.nan)
        fieldsInfo["energyNo"].append(int(IonBeamDataset.NumberOfControlPoints.real / 2))
        spotsNo = 0
        energy = []
        for IonControlPointDataset in IonBeamDataset.IonControlPointSequence:
            spotsNo += IonControlPointDataset.NumberOfScanSpotPositions
            energy.append(IonControlPointDataset.NominalBeamEnergy.real)
        fieldsInfo["minEnergy"].append(np.min(energy))
        fieldsInfo["maxEnergy"].append(np.max(energy))
        fieldsInfo["spotsNo"].append(int(spotsNo / 2))

        fieldsInfo["magnetToIsoDist"].append(np.array(IonBeamDataset.VirtualSourceAxisDistances).tolist())
        if (0x300B, 0x1004) in IonBeamDataset:
            fieldsInfo["nomRange"].append(IonBeamDataset[(0x300B, 0x1004)].value)
        if (0x300B, 0x100E) in IonBeamDataset:
            fieldsInfo["nomSOBPWidth"].append(IonBeamDataset[(0x300B, 0x100E)].value)
        fieldsInfo["supportID"].append(IonBeamDataset.PatientSupportID)
        treatmentMachineNames.append(IonBeamDataset.TreatmentMachineName)

    # convert fields info to pandas dataframe
    fieldsInfo = pd.DataFrame(fieldsInfo)
    fieldsInfo.set_index("deliveryNo", inplace=True)

    # get treatment machine name as the machine name of the first field
    planInfo["treatmentMachineName"] = treatmentMachineNames[0]

    # get isocentre position of the first field
    planInfo["isocentrePos"] = fieldsInfo.isocenterPos.iloc[0]

    ### get spots parameters for each field
    fieldsSpotsInfo = []
    for deliveryNo, ReferencedBeamDataset in enumerate(dicomTags.FractionGroupSequence[0].ReferencedBeamSequence, start=1):
        IonBeamDataset = _getIonBeamDatasetForFieldNumber(fileName, int(ReferencedBeamDataset.ReferencedBeamNumber))
        # Continue if couldn't find IonBeamDataset for the IonBeamDataset or the Treatment Delivery Type of the IonBeamDataset is not TREATMENT
        if not IonBeamDataset or not (IonBeamDataset.TreatmentDeliveryType == "TREATMENT"):
            continue
        # check if RadiationType is 'PROTON'
        if not IonBeamDataset.RadiationType == "PROTON":
            raise TypeError(f"The type of TREATMENT field is not PROTON but {IonBeamDataset.RadiationType}.")
        # check if the scan mode is MODULATED and beam type is static
        if not IonBeamDataset.ScanMode == "MODULATED" and not IonBeamDataset.BeamType == "STATIC":
            raise TypeError(f"The scan mode of the field is not MODULATED and/or beam type is not STATIC.")

        # get spots parameters from IonControlPointSequence
        slicesInfo = []
        for sliceIdx, IonControlPointDataset in enumerate(IonBeamDataset.IonControlPointSequence):
            # skip if the sum of meterset weights is zero
            if np.sum(IonControlPointDataset.ScanSpotMetersetWeights) == 0:
                continue

            # number of spots for slice
            spotsNo = int(IonControlPointDataset.NumberOfScanSpotPositions)

            sliceInfo = {}
            sliceInfo["deliveryNo"] = [deliveryNo] * spotsNo
            sliceInfo["energyNo"] = [int(sliceIdx / 2) + 1] * spotsNo
            sliceInfo["nomEnergy"] = [IonControlPointDataset.NominalBeamEnergy.real] * spotsNo
            sliceInfo["spotMsW"] = IonControlPointDataset.ScanSpotMetersetWeights
            sliceInfo["spotPosX"] = IonControlPointDataset.ScanSpotPositionMap[0::2]
            sliceInfo["spotPosY"] = IonControlPointDataset.ScanSpotPositionMap[1::2]
            sliceInfo["spotTuneID"] = [float(IonControlPointDataset.ScanSpotTuneID)] * spotsNo
            sliceInfo["spotPaintingNo"] = [IonControlPointDataset.NumberOfPaintings] * spotsNo
            sliceInfo = pd.DataFrame(sliceInfo)
            sliceInfo["spotMU"] = sliceInfo.spotMsW / fieldsInfo.loc[deliveryNo].cumMsW * fieldsInfo.loc[deliveryNo].MU
            slicesInfo.append(sliceInfo)
        slicesInfo = pd.concat(slicesInfo)
        fieldsSpotsInfo.append(slicesInfo)

    fieldsSpotsInfo = pd.concat(fieldsSpotsInfo)
    fieldsSpotsInfo.index = range(1, fieldsSpotsInfo.shape[0] + 1)
    fieldsSpotsInfo.index.rename("spotNo", inplace=True)

    # make consistency check and raise warning if raiseWarning==True
    if raiseWarning:
        # check if the dose prescribed dose in DoseReferenceSequence is the same as the sum of BeamDose in FractionGroupSequence
        if not np.round(planInfo["dosePrescribed"], 3) == np.round(np.nansum(fieldsInfo.dose) * planInfo["fractionsNo"], 3):
            warnings.warn("Warning: 'TargetPrescriptionDose' is different from sum of 'BeamDose' in 'FractionGroupSequence'.")
        # check if treatment machine name is the same for all fields
        if not treatmentMachineNames.count(treatmentMachineNames[0]) == len(treatmentMachineNames):
            warnings.warn("Warning: The machine name is not the same for all fields.")
        # check if sum of meterset weights for spots is equal with the sum of the cumulative meterset weights for fields
        if not np.round(fieldsSpotsInfo.spotMsW.sum(), 3) == np.round(fieldsInfo.cumMsW.sum(), 3):
            warnings.warn("Warning: The sum of meterset weights for spots is not equal to the sum of cumulative metersetweights for fields.")
        # check if isocentre position is the same for all fields
        if not (np.array(fieldsInfo.isocenterPos.to_list())[0] == np.array(fieldsInfo.isocenterPos.to_list())).all():
            warnings.warn("Warning: Isocentre positions are not the same for all TREATMENT fields.")

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Patient name:     '{:s}'".format(planInfo["patientName"]))
        print("# Plan label:       '{:s}'".format(planInfo["planLabel"]))
        print("# Plan date:        '{:s}'".format(planInfo["planDate"]))
        print("# Machine name:     '{:s}'".format(planInfo["treatmentMachineName"]))
        print("# Target structure: '{:s}'".format(planInfo["targetStructName"]))
        print(
            "# Isocentre [mm]:   [{:.2f} {:.2f} {:.2f}]{:s}".format(
                *planInfo["isocentrePos"],
                " (the same for all fields)"
                if (np.array(fieldsInfo.isocenterPos.to_list())[0] == np.array(fieldsInfo.isocenterPos.to_list())).all()
                else " (first field here but various positions in fields)",
            )
        )
        print("# Number of fractions: {:d}".format(planInfo["fractionsNo"]))
        print("# Dose pres. (all fractions):    {:.3f} Gy RBE (from DoseReferenceSequence)".format(np.round(planInfo["dosePrescribed"], 3)))
        print("# Dose pres. (single fraction):  {:.3f} Gy RBE (from DoseReferenceSequence)".format(np.round(planInfo["dosePrescribed"] / planInfo["fractionsNo"], 3)))
        print("# Dose pres. (all fractions):    {:.3f} Gy RBE (sum of BeamDose in FractionGroupSequence)".format(fieldsInfo.dose.sum() * planInfo["fractionsNo"]))
        print("# Dose pres. (single fraction):  {:.3f} Gy RBE (sum of BeamDose in FractionGroupSequence)".format(fieldsInfo.dose.sum()))
        print("# Number of treatment fields: {:d}".format(planInfo["treatmentFieldsNumber"]))
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return planInfo, fieldsInfo, fieldsSpotsInfo


def getRSInfo(fileName, displayInfo=False):
    """Get some information from the RS structures from RS dicom file.

    The function retrieves some basic information about structures from a RS dicom file.

    Parameters
    ----------
    fileName : path
        Path to RS dicom file.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

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
                "ROIType": "unclasified" if not structs[struct]["type"] else structs[struct]["type"],
                "ROIName": structs[struct]["name"],
                "ROIColor": structs[struct]["color"],
            },
            ignore_index=True,
        )

    ROITable = ROITable.set_index("ID")

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Found {:d} structures:".format(ROITable.shape[0]))
        print("#", ROITable.groupby("ROIType")["ROIType"].count())
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return ROITable


def getExternalName(fileName, displayInfo=False):
    """Get name of the EXTERNAL structure from RS dicom file.

    The function retrieves the name of the structure of type EXTERNAL from a RS dicom file.
    If more than one structure of type EXTERNAL exists in the RS dicom file, then the first one is returned.

    Parameters
    ----------
    fileName : path
        Path to RS dicom file.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

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
    externalName = ROIinfo.loc[ROIinfo.ROIType == "EXTERNAL"].ROIName.values

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
        Displays a summary of the function results. (def. False)

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
        Displays a summary of the function results. (def. False)

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
        Path string to dicom file with structures (RS file).
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


def _checkContourCWDirection(contour):
    import numpy as np

    """Check if the contour has CW (True) or CCW (False) direction. 
    The CW (True) contour direction usually means that it is a filled polygon
    and the CCW (False) cuntour direction that it is a hole in the filled polygon."""
    result = 0.5 * np.array(np.dot(contour[:, 0], np.roll(contour[:, 1], 1)) - np.dot(contour[:, 1], np.roll(contour[:, 0], 1)))
    return result < 0


def getRDFileNameForFieldNumber(fileNames, fieldNumber, displayInfo=False):
    """Get file name of the RD dose dicom of for given field number.

    The function searches for the RD dose dicom file for a given
    field number (beam number) and returns its file name.

    Parameters
    ----------
    fileNames : array_like
        An iterable (list, tuple, etc.) of paths to RD dicoms.
    fieldNumber: scalar, int
        Field number to find the dicom for.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    Path string:
        Path to the RD dose file for the given beam number.
    """
    import pydicom as dicom
    import warnings
    import fredtools as ft

    for fileName in fileNames:
        if not ft.getDicomType(fileName) == "RD":
            warnings.warn(f"File {fileName} is not a RD dicom.")
            continue

        dicomRD = dicom.read_file(fileName)
        if fieldNumber == int(dicomRD.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber):
            break
        else:
            fileName = ""

    if not fileName:
        warnings.warn(f"Warning: could not find RD dose dicom file for the field number {fieldNumber}.")

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Path to the RD dose dicom with the field number {fieldNumber}:")
        print(f"# {fileName}")
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return fileName
