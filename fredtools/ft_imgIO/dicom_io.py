def getDicomTypeName(dicomVar):
    r"""Check the type of the dicom given as a path or tags.

    The function return the name of the SOP Class UID tag of a dicom
    file given as a file name or dicom tags. The description of the
    SOP Class UID names can be found in [1]_.

    Parameters
    ----------
    dicomVar : path or tags
        A string of path to a dicom file or a dicom tag structure read
        by pydicom.read_file.

    Returns
    -------
    string
        A string of the SOP Class UID name.

    See Also
    --------
    sortDicoms : sort dicom files in a folder by type.

    References
    ----------
    .. [1] `DICOM standard description - SOP Class UID <https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html>`_
    """
    import pydicom as dicom

    # if the input is a string then get the dicom tags
    if isinstance(dicomVar, str):
        dicomVar = dicom.read_file(dicomVar, specific_tags=["SOPClassUID"], stop_before_pixels=True)

    # check if the tags are dicom dataset
    if not isinstance(dicomVar, dicom.dataset.FileDataset):
        raise TypeError(f"The tags is not an instance of dicom.dataset.FileDataset.")

    # check if SOPClassUID exists in the tags
    if not "SOPClassUID" in dicomVar:
        raise ValueError("Cannot find tag 'SOPClassUID' in the dicom tags.")

    return dicomVar.SOPClassUID.name


def _isDicomCT(dicomVar, raiseError=False):
    r"""Check if the dicom is of CT type and raise error if requested."""
    import itk

    instanceBool = "CT Image Storage" in getDicomTypeName(dicomVar)

    if raiseError and not instanceBool:
        raise TypeError(f"The dicom is not a CT type but has SOP class UID name '{getDicomTypeName(dicomVar)}'.")
    return instanceBool


def _isDicomRS(dicomVar, raiseError=False):
    r"""Check if the dicom is of RS type and raise error if requested."""
    import itk

    instanceBool = "Structure Set Storage" in getDicomTypeName(dicomVar)

    if raiseError and not instanceBool:
        raise TypeError(f"The dicom is not a RS type but has SOP class UID name '{getDicomTypeName(dicomVar)}'.")
    return instanceBool


def _isDicomRN(dicomVar, raiseError=False):
    r"""Check if the dicom is of RN type and raise error if requested."""
    import itk

    instanceBool = "Plan Storage" in getDicomTypeName(dicomVar)  # ("RT Plan Storage" or "RT Ion Plan Storage")

    if raiseError and not instanceBool:
        raise TypeError(f"The dicom is not a RN type but has SOP class UID name '{getDicomTypeName(dicomVar)}'.")
    return instanceBool


def _isDicomRD(dicomVar, raiseError=False):
    r"""Check if the dicom is of RD type and raise error if requested."""
    import itk

    instanceBool = "Dose Storage" in getDicomTypeName(dicomVar)

    if raiseError and not instanceBool:
        raise TypeError(f"The dicom is not a RD type but has SOP class UID name '{getDicomTypeName(dicomVar)}'.")
    return instanceBool


def _getRNBeamSequence(dicomVar):
    r"""Get beam sequence ('IonBeamSequence' or 'BeamSequence') from dicom."""
    import pydicom as dicom

    # check if it is a RN dicom
    _isDicomRN(dicomVar)

    # if the input is a string then get the dicom tags
    if isinstance(dicomVar, str):
        dicomVar = dicom.read_file(dicomVar, stop_before_pixels=True)

    # check if the tags are dicom dataset
    if not isinstance(dicomVar, dicom.dataset.FileDataset):
        raise TypeError(f"The tags is not an instance of dicom.dataset.FileDataset.")

    # get name of the SOP Class UID
    dicomTypeName = getDicomTypeName(dicomVar)
    if dicomTypeName == "RT Ion Plan Storage":
        if not "IonBeamSequence" in dicomVar:
            raise ValueError(f"Can not find 'IonBeamSequence' in the dicom.")
        else:
            return dicomVar["IonBeamSequence"]
    elif dicomTypeName == "RT Plan Storage":
        if not "BeamSequence" in dicomVar:
            raise ValueError(f"Can not find 'BeamSequence' in the dicom.")
        else:
            return dicomVar["BeamSequence"]
    else:
        raise TypeError(f"Cannot recognise the dicom as 'RT Plan Storage' nor 'RT Ion Plan Storage'.")


def sortDicoms(searchFolder, recursive=False, displayInfo=False):
    """Sort dicom file names found in the search folder for CT, RS, RN, RD and Unknown.

    The function sorts file names found in the `searchFolder`
    (and subfolders if requested) for:

        -  CT - dicom files of CT Image Storage ("CT Image Storage", "Enhanced CT Image Storage" or "Legacy Converted Enhanced CT Image Storage")
        -  RS - dicom files of RT Structure Set Storage
        -  RN - dicom files of RT Plan Storage ("RT Plan Storage" or "RT Ion Plan Storage")
        -  RD - dicom files of 1D/2D/3D RT Dose Storage (for instance dose distribution)
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
        dicomFileNames = glob.glob(os.path.join(searchFolder, "**/*.dcm"), recursive=True)
    else:
        dicomFileNames = glob.glob(os.path.join(searchFolder, "*.dcm"), recursive=False)

    CTfileNames = []
    RSfileNames = []
    RNfileNames = []
    RDfileNames = []
    UnknownfileNames = []
    for dicomFileName in dicomFileNames:
        if _isDicomCT(dicomFileName):  # CT
            CTfileNames.append(dicomFileName)
        elif _isDicomRS(dicomFileName):  # RS
            RSfileNames.append(dicomFileName)
        elif _isDicomRN(dicomFileName):  # RN ("RT Plan Storage" or "RT Ion Plan Storage")
            RNfileNames.append(dicomFileName)
        elif _isDicomRD(dicomFileName):  # RD
            RDfileNames.append(dicomFileName)
        else:
            UnknownfileNames.append(dicomFileName)  # unrecognized dicoms
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Found dicoms: {:d} x CT, {:d} x RS, {:d} x RN, {:d} x RD, {:d} x unknown".format(len(CTfileNames), len(RSfileNames), len(RNfileNames), len(RDfileNames), len(UnknownfileNames)))
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    dicomTypes = {"CTfileNames": CTfileNames, "RSfileNames": RSfileNames, "RNfileNames": RNfileNames, "RDfileNames": RDfileNames, "Unknown": UnknownfileNames}
    for dicomType, dicomName in dicomTypes.items():
        if isinstance(dicomName, list) and len(dicomName) == 1:
            dicomTypes[dicomType] = dicomName[0]

    return dicomTypes


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
    _isDicomRN(fileName, raiseError=True)

    # check if the beamNumber is an integer scalar
    if not np.isscalar(beamNumber) or not isinstance(beamNumber, int):
        raise TypeError(f"The value of 'beamNumber' must be a scalar integer.")

    # read dicom
    dicomTags = dicom.read_file(fileName)

    # get beam sequence (BeamSequence or IonBeamSequence)
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
    _isDicomRN(fileName, raiseError=True)

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


def getRNMachineName(fileName, displayInfo=False):
    """Get the machine name defined in the RN plan.

    The function retrieves the machine name defined in the RN dicom file.

    Parameters
    ----------
    fileName : path
        Path to RN dicom file.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    string
        Treatment machine name.

    See Also
    --------
    getRNFields : get summary of parameters for each field defined in RN plan.
    getRNSpots : get summary of parameters for each spot defined in RN plan.
    getRNInfo : get some basic information from the RN plan.
    """
    import fredtools as ft
    import pydicom as dicom

    # check if dicom is RN
    ft.ft_imgIO.dicom_io._isDicomRN(fileName, raiseError=True)

    # read dicom
    dicomTags = dicom.read_file(fileName)

    # get beam sequence (BeamSequence or IonBeamSequence)
    beamSequence = _getRNBeamSequence(dicomTags)

    treatmentMachineName = []
    for beamDataset in beamSequence:
        # Continue if couldn't find beamDataset for the beamDataset or the Treatment Delivery Type of the beamDataset is not TREATMENT
        if not beamDataset or not (beamDataset.TreatmentDeliveryType == "TREATMENT"):
            continue

        if "TreatmentMachineName" in beamDataset:
            treatmentMachineName.append(beamDataset.TreatmentMachineName)
        else:
            continue

    # check if any value was found
    if len(set(treatmentMachineName)) == 0:
        raise ValueError(f"Could not find Treatment Machine Name. There is no 'TreatmentMachineName' tags in 'IonBeamSequence' or the Treatment Delivery Type is not 'TREATMENT' for any field.")

    # check if all values are the same
    if not len(set(treatmentMachineName)) == 1:
        raise ValueError(f"Not all 'TreatmentMachineName' tags in 'IonBeamSequence' are the same but are: {treatmentMachineName}.")

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Machine name: '{:s}'".format(treatmentMachineName[0]))
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return treatmentMachineName[0]


def getRNIsocenter(fileName, displayInfo=False):
    """Get the isocenter position defined in the RN plan.

    The function retrieves the isocenter position defined in the RN dicom file.
    The isocenter is defined for each field separately but usually it is the same
    for all fields. If it is not the same then a warning is raised and a geometrical
    center is returned.

    Parameters
    ----------
    fileName : path
        Path to RN dicom file.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    list
        3-elements list of XYZ isocenter coordinates

    See Also
    --------
    getRNFields : get summary of parameters for each field defined in RN plan.
    getRNSpots : get summary of parameters for each spot defined in RN plan.
    getRNInfo : get some basic information from the RN plan.
    """
    import fredtools as ft
    import pydicom as dicom
    import warnings
    import numpy as np

    # check if dicom is RN
    ft.ft_imgIO.dicom_io._isDicomRN(fileName, raiseError=True)

    # read dicom
    dicomTags = dicom.read_file(fileName)

    # get beam sequence (BeamSequence or IonBeamSequence)
    beamSequence = ft.ft_imgIO.dicom_io._getRNBeamSequence(dicomTags)

    isocenterPosition = []
    for beamDataset in beamSequence:
        # Continue if couldn't find beamDataset in the beamSequence or the Treatment Delivery Type of the beamDataset is not TREATMENT
        if not beamDataset or not (beamDataset.TreatmentDeliveryType == "TREATMENT"):
            continue
        if ("IonControlPointSequence" in beamDataset) and ("IsocenterPosition" in beamDataset.IonControlPointSequence[0]):
            isocenterPosition.append(beamDataset.IonControlPointSequence[0].IsocenterPosition)

    isocenterPosition = np.unique(isocenterPosition, axis=0)

    # check if any value was found
    if isocenterPosition.shape[0] == 0:
        raise ValueError(
            f"Could not find any isocenter position. There is no 'IsocenterPosition' tag in 'IonControlPointSequence[0]' of 'beamDataset' or the Treatment Delivery Type is not 'TREATMENT' for any field."
        )

    # check if all values are the same
    if not isocenterPosition.shape[0] == 1:
        warnings.warn(f"Not all isocenter positions are the same. Found isocenter positions:\n {isocenterPosition} \nThe geometrical centre will be returned.")
        isocenterPosition = np.mean(isocenterPosition, axis=0)
    else:
        isocenterPosition = isocenterPosition[0]

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Isocenter position [mm]: ", isocenterPosition)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return isocenterPosition.tolist()


def getRNSpots(fileName):
    """Get parameters of each spot defined in the RN file.

    The function retrieves information for each spot defined in the RN dicom file.
    All spots are listed in the results, including the spots with zero meterset weights.

    Parameters
    ----------
    fileName : path
        Path to RN dicom file.

    Returns
    -------
    pandas DataFrame
        DataFrame with the spots' parameters.

    See Also
    --------
    getRNFields : get summary of parameters for each field defined in RN plan.
    getRNInfo : get some basic information from the RN plan.
    """
    import pandas as pd
    import fredtools as ft
    import pydicom as dicom
    import numpy as np

    # check if dicom is RN
    _isDicomRN(fileName, raiseError=True)

    # check if dicom is "RT Ion Plan Storage"
    if not "RT Ion Plan Storage" == getDicomTypeName(fileName):
        raise TypeError(f"The dicom is not of 'RT Ion Plan Storage' type but SOP class UID name is '{getDicomTypeName(fileName)}'")

    # read dicom
    dicomTags = dicom.read_file(fileName)

    # get RN spots parameters in order of delivery
    spotsInfo = []
    for fieldDeliveryNo, IonBeamDataset in enumerate(dicomTags.IonBeamSequence, start=1):
        # continue if couldn't find IonBeamDataset or the Treatment Delivery Type of the IonBeamDataset is not TREATMENT
        if not IonBeamDataset or not (IonBeamDataset.TreatmentDeliveryType == "TREATMENT"):
            continue

        # get information for the field
        fieldNo = int(IonBeamDataset.BeamNumber)
        fieldName = IonBeamDataset.BeamName

        # get the field isocentre
        isocenterPos = np.array(IonBeamDataset.IonControlPointSequence[0].IsocenterPosition).tolist()

        # get the field RS IDs
        if (IonBeamDataset.NumberOfRangeShifters.real != 0) and ("RangeShifterSequence" in IonBeamDataset):
            RSIDs = IonBeamDataset.RangeShifterSequence[0].RangeShifterID
        else:
            RSIDs = ""

        # get the field magnets' distances
        if "VirtualSourceAxisDistances" in IonBeamDataset:
            fieldMagDist = IonBeamDataset.VirtualSourceAxisDistances
        else:
            fieldMagDist = np.nan

        # get ReferencedBeamDataset
        ReferencedBeamDataset = ft.ft_imgIO.dicom_io._getReferencedBeamDatasetForFieldNumber(fileName, fieldNo)

        # get fieldDose and field cumulative Meterset Weight
        fieldDose = ReferencedBeamDataset.BeamMeterset
        fieldCumMsW = IonBeamDataset.FinalCumulativeMetersetWeight

        # get spots parameters from IonControlPointSequence
        slicesInfo = []
        for sliceIdx, IonControlPointDataset in enumerate(IonBeamDataset.IonControlPointSequence):

            # number of spots for slice
            spotsNo = int(IonControlPointDataset.NumberOfScanSpotPositions)

            sliceInfo = {}
            sliceInfo["FDeliveryNo"] = [fieldDeliveryNo] * spotsNo
            sliceInfo["FNo"] = [fieldNo] * spotsNo
            sliceInfo["FName"] = [fieldName] * spotsNo
            sliceInfo["FGantryAngle"] = [IonControlPointDataset.GantryAngle.real] * spotsNo if "GantryAngle" in IonControlPointDataset else np.nan
            sliceInfo["FCouchAngle"] = [IonControlPointDataset.PatientSupportAngle.real] * spotsNo if "PatientSupportAngle" in IonControlPointDataset else np.nan
            sliceInfo["FCouchPitchAngle"] = [IonControlPointDataset.TableTopPitchAngle.real] * spotsNo if "TableTopPitchAngle" in IonControlPointDataset else np.nan
            sliceInfo["FCouchRollAngle"] = [IonControlPointDataset.TableTopRollAngle.real] * spotsNo if "TableTopRollAngle" in IonControlPointDataset else np.nan
            sliceInfo["FIsoPos"] = [isocenterPos] * spotsNo
            sliceInfo["FMagDist"] = [fieldMagDist] * spotsNo
            sliceInfo["FEnergyNo"] = [int(sliceIdx / 2) + 1] * spotsNo
            sliceInfo["PBRSID"] = [RSIDs] * spotsNo

            # get RS Settings for MEVION
            if "RangeShifterSettingsSequence" in IonControlPointDataset:
                sliceInfo["PBRSSetting"] = (
                    [IonControlPointDataset.RangeShifterSettingsSequence[0].RangeShifterSetting] * spotsNo
                    if "RangeShifterSetting" in IonControlPointDataset.RangeShifterSettingsSequence[0]
                    else np.nan
                )
            else:
                sliceInfo["PBRSSetting"] = np.nan

            sliceInfo["PBSnoutPos"] = [IonControlPointDataset.SnoutPosition.real] * spotsNo if "SnoutPosition" in IonControlPointDataset else np.nan
            sliceInfo["PBnomEnergy"] = [IonControlPointDataset.NominalBeamEnergy.real] * spotsNo if "NominalBeamEnergy" in IonControlPointDataset else np.nan
            sliceInfo["PBMsW"] = IonControlPointDataset.ScanSpotMetersetWeights
            sliceInfo["PBMU"] = np.array(IonControlPointDataset.ScanSpotMetersetWeights) / fieldCumMsW * fieldDose
            sliceInfo["PBPosX"] = IonControlPointDataset.ScanSpotPositionMap[0::2]
            sliceInfo["PBPosY"] = IonControlPointDataset.ScanSpotPositionMap[1::2]
            sliceInfo["PBTuneID"] = [IonControlPointDataset.ScanSpotTuneID] * spotsNo
            sliceInfo["PNPainting"] = [IonControlPointDataset.NumberOfPaintings] * spotsNo
            sliceInfo = pd.DataFrame(sliceInfo)
            slicesInfo.append(sliceInfo)
        slicesInfo = pd.concat(slicesInfo)
        slicesInfo["FSpotNo"] = range(1, slicesInfo.shape[0] + 1)
        spotsInfo.append(slicesInfo)
    spotsInfo = pd.concat(spotsInfo)

    # drop columns with all NaN values
    spotsInfo.dropna(axis="columns", how="all", inplace=True)

    # fill nan values with the lat valid value
    spotsInfo.fillna(method="ffill", inplace=True)

    # reset index
    spotsInfo.index = range(1, len(spotsInfo) + 1)

    return spotsInfo


def getRNFields(fileName, raiseWarning=True, displayInfo=False):
    """Get parameters of each field defined in the RN file.

    The function retrieves information for each field defined in the RN dicom file.
    A consistency check is performed to check correctness of the parameters written
    in the RN dicom file.

    Parameters
    ----------
    fileName : path
        Path to RN dicom file.
    raiseWarning : bool, optional
        Raise warnings if the consistency check fails. (def. False)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    pandas DataFrame
        DataFrame with the fields' parameters.

    See Also
    --------
    getRNSpots : get summary of parameters for each spot defined in RN plan.
    getRNInfo : get some basic information from the RN plan.
    """
    import pandas as pd
    import fredtools as ft
    import pydicom as dicom
    import numpy as np
    import warnings

    # check if dicom is RN
    _isDicomRN(fileName, raiseError=True)

    # check if dicom is "RT Ion Plan Storage"
    if not "RT Ion Plan Storage" == getDicomTypeName(fileName):
        raise TypeError(f"The dicom is not of 'RT Ion Plan Storage' type but SOP class UID name is '{getDicomTypeName(fileName)}'")

    # get spots info
    spotsInfo = getRNSpots(fileName)

    # drop all spots with the PBMsW (Meterset Weight) less or equal to zero
    spotsInfo = spotsInfo[spotsInfo.PBMsW > 0]

    # read dicom
    dicomTags = dicom.read_file(fileName)

    def uniqueValue(groupByDataSet):
        """Get the unique values for the grouped datasets. If no unique value then `var` is returned."""
        uniqueValue = np.unique(groupByDataSet)
        if len(uniqueValue) > 1:
            return "var"
        else:
            return uniqueValue[0]

    fieldsInfo = pd.DataFrame()
    fieldsInfo["FNo"] = spotsInfo.groupby("FDeliveryNo").FNo.apply(uniqueValue)
    fieldsInfo["FName"] = spotsInfo.groupby("FDeliveryNo").FName.apply(uniqueValue)
    fieldsInfo["FGantryAngle"] = spotsInfo.groupby("FDeliveryNo").FGantryAngle.apply(uniqueValue)
    fieldsInfo["FCouchAngle"] = spotsInfo.groupby("FDeliveryNo").FCouchAngle.apply(uniqueValue)
    fieldsInfo["FCouchPitchAngle"] = spotsInfo.groupby("FDeliveryNo").FCouchPitchAngle.apply(uniqueValue)
    fieldsInfo["FCouchRollAngle"] = spotsInfo.groupby("FDeliveryNo").FCouchRollAngle.apply(uniqueValue)
    fieldsInfo["FIsoPos"] = spotsInfo.groupby("FDeliveryNo").FIsoPos.apply(uniqueValue)
    fieldsInfo["FRSID"] = spotsInfo.groupby("FDeliveryNo").PBRSID.apply(uniqueValue)
    fieldsInfo["FSnoutPos"] = spotsInfo.groupby("FDeliveryNo").PBSnoutPos.apply(uniqueValue)
    fieldsInfo["FEnergyNo"] = spotsInfo.groupby("FDeliveryNo").FEnergyNo.nunique()
    fieldsInfo["FEnergyMin"] = spotsInfo.groupby("FDeliveryNo").PBnomEnergy.min()
    fieldsInfo["FEnergyMax"] = spotsInfo.groupby("FDeliveryNo").PBnomEnergy.max()
    fieldsInfo["FSpotNo"] = spotsInfo.groupby("FDeliveryNo").FSpotNo.count()

    fieldsInfo["FDosePos"] = np.nan
    fieldsInfo["FDosePos"] = fieldsInfo["FDosePos"].astype("object")
    for FDeliveryNo, fieldInfo in fieldsInfo.iterrows():
        ReferencedBeamDataset = ft.ft_imgIO.dicom_io._getReferencedBeamDatasetForFieldNumber(fileName, fieldInfo.FNo)

        fieldsInfo.loc[FDeliveryNo, "FDose"] = ReferencedBeamDataset.BeamDose.real if "BeamDose" in ReferencedBeamDataset else np.NaN
        fieldsInfo.at[FDeliveryNo, "FDosePos"] = np.array(ReferencedBeamDataset.BeamDoseSpecificationPoint).tolist() if "BeamDoseSpecificationPoint" in ReferencedBeamDataset else np.NaN
        fieldsInfo.loc[FDeliveryNo, "FMU"] = ReferencedBeamDataset.BeamMeterset.real if "BeamMeterset" in ReferencedBeamDataset else np.NaN

    fieldsInfo["FMagDist"] = np.nan
    fieldsInfo["FMagDist"] = fieldsInfo["FMagDist"].astype("object")
    for FDeliveryNo, fieldInfo in fieldsInfo.iterrows():
        IonBeamDataset = ft.ft_imgIO.dicom_io._getIonBeamDatasetForFieldNumber(fileName, fieldInfo.FNo)

        fieldsInfo.loc[FDeliveryNo, "FCumMsW"] = IonBeamDataset.FinalCumulativeMetersetWeight.real if "FinalCumulativeMetersetWeight" in IonBeamDataset else np.NaN  # Final Cumulative Meterset Weight
        fieldsInfo.loc[FDeliveryNo, "FnomRange"] = IonBeamDataset[(0x300B, 0x1004)].value if (0x300B, 0x1004) in IonBeamDataset else np.NaN
        fieldsInfo.loc[FDeliveryNo, "FnomSOBPWidth"] = IonBeamDataset[(0x300B, 0x100E)].value if (0x300B, 0x100E) in IonBeamDataset else np.NaN
        fieldsInfo.loc[FDeliveryNo, "FsupportID"] = IonBeamDataset.PatientSupportID if "PatientSupportID" in IonBeamDataset else np.NaN
        fieldsInfo.at[FDeliveryNo, "FMagDist"] = IonBeamDataset.VirtualSourceAxisDistances if "VirtualSourceAxisDistances" in IonBeamDataset else np.NaN  #  Virtual Source-Axis Distances

    # make consistency check and raise warning if raiseWarning==True
    if raiseWarning:
        # check if the 'FinalCumulativeMetersetWeight' defined in 'IonBeamDataset' for each field is similar to the sum of 'ScanSpotMetersetWeights' for each pencil beam in the field
        if any(np.abs(spotsInfo.groupby("FDeliveryNo").PBMsW.sum() / fieldsInfo.FCumMsW - 1) > 0.005):  # accuracy 0.5%
            warnings.warn(
                "Warning: At least for one field the 'FinalCumulativeMetersetWeight' defined in 'IonBeamDataset' is different from the sum of 'ScanSpotMetersetWeights' for each pencil beam.\n\tThe 'FinalCumulativeMetersetWeight' defined in 'IonBeamDataset' is in the output."
            )

    # drop columns with all NaN values
    fieldsInfo.dropna(axis="columns", how="all", inplace=True)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        if ft._checkJupyterMode():
            display(fieldsInfo)
        else:
            print(fieldsInfo)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return fieldsInfo


def getRNInfo(fileName, displayInfo=False):
    """Get some information from the RN plan.

    The function retrieves some usefull information from a RN dicom of a treatment plan.
    Following information are saved to a dictionary:

        -  *RNFileName* : absolute path to the RN file.
        -  *dosePrescribed* : dose prescribed to the target (see notes below).
        -  *fractionNo* : number of the fractions planned.
        -  *targetStructName* : name of the structure which the plan was prepared for (dose not work for all RN dicoms).
        -  *planLabel* : name of the treatment plan (can be empty for anonymized DICOM).
        -  *planDate* : date of the plan creation (can be empty for anonymized DICOM).
        -  *planTime* : time of the plan creation (can be empty for anonymized DICOM).
        -  *patientName* : name of the patient (can be empty for anonymized DICOM).
        -  *patientBirthDate* : birth date of the patient (can be empty for anonymized DICOM).
        -  *patientID* : ID of the patient (can be empty for anonymized DICOM).
        -  *manufacturer* : manufacturer of the treatment planning system.
        -  *softwareVersions* : version of the treatment planning system.
        -  *stationName* : name of the station on which the plan has been prepared.
        -  *machineName* : name of the treatment machine for which the plan was prepared.
        -  *totalFieldsNumber* : total number of fields including setup, treatment and other fields.
        -  *treatmentFieldsNumber* : total number of treatment fields.
        -  *setupFieldsNumber* : total number of setup fields.
        -  *otherFieldsNumber* : total number of other fields.

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
    getRNFields : get summary of parameters for each field defined in RN plan.
    getRNSpots : get summary of parameters for each spot defined in RN plan.

    Notes
    -----
    The prescribed dose, *dosePrescribed*, is read from *TargetPrescriptionDose* in *DoseReferenceSequence*,
    but if this is not available it is calculated as the sum of *BeamDose* in *FractionGroupSequence[0].ReferencedBeamSequence*
    multiplied by the number of fractions.
    """
    import numpy as np
    import fredtools as ft
    import pydicom as dicom
    import os

    # check if dicom is RN
    ft.ft_imgIO.dicom_io._isDicomRN(fileName, raiseError=True)

    # read dicom
    dicomTags = dicom.read_file(fileName)

    # prepare plan info
    planInfo = {}

    # get absolute path to RN file
    planInfo["RNFileName"] = os.path.abspath(fileName)

    # get number of fractions from NumberOfFractionsPlanned tag of FractionGroupSequence
    if ("FractionGroupSequence" in dicomTags) and ("NumberOfFractionsPlanned" in dicomTags.FractionGroupSequence[0]):
        planInfo["fractionNo"] = dicomTags.FractionGroupSequence[0].NumberOfFractionsPlanned.real
    else:
        planInfo["fractionNo"] = np.nan

    # get prescribed dose from TargetPrescriptionDose tag of DoseReferenceSequence
    if "TargetPrescriptionDose" in dicomTags.DoseReferenceSequence[0]:
        planInfo["dosePrescribed"] = dicomTags.DoseReferenceSequence[0].TargetPrescriptionDose.real
    else:
        planInfo["dosePrescribed"] = 0
        for referencedBeam in dicomTags.FractionGroupSequence[0].ReferencedBeamSequence:
            if "BeamDose" in referencedBeam:
                planInfo["dosePrescribed"] += referencedBeam.BeamDose
        if planInfo["dosePrescribed"] == 0:
            planInfo["dosePrescribed"] = np.nan
        else:
            planInfo["dosePrescribed"] *= planInfo["fractionNo"]

    # get target struct name from private tag of DoseReferenceSequence
    if ("DoseReferenceSequence" in dicomTags) and ([0x3267, 0x1000] in dicomTags.DoseReferenceSequence[0]):
        planInfo["targetStructName"] = dicomTags.DoseReferenceSequence[0][0x3267, 0x1000].value.decode("utf-8")
    else:
        planInfo["targetStructName"] = ""

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
    planInfo["machineName"] = ft.getRNMachineName(fileName)

    # get beam sequence (BeamSequence or IonBeamSequence)
    beamSequence = ft.ft_imgIO.dicom_io._getRNBeamSequence(dicomTags)

    # count fields' type and treatment machine name
    planInfo["totalFieldsNumber"] = int(dicomTags.FractionGroupSequence[0].NumberOfBeams)
    planInfo["treatmentFieldsNumber"] = 0
    planInfo["setupFieldsNumber"] = 0
    planInfo["otherFieldsNumber"] = 0
    for ifield in range(planInfo["totalFieldsNumber"]):
        if beamSequence[ifield].TreatmentDeliveryType == "TREATMENT":
            planInfo["treatmentFieldsNumber"] += 1
        elif beamSequence[ifield].TreatmentDeliveryType == "SETUP":
            planInfo["setupFieldsNumber"] += 1
        else:
            planInfo["otherFieldsNumber"] += 1

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Patient name:     '{:s}'".format(planInfo["patientName"].replace("^", " ")))
        print("# Plan label:       '{:s}'".format(planInfo["planLabel"]))
        print("# Plan date:        '{:s}'".format(planInfo["planDate"]))
        print("# Machine name:     '{:s}'".format(planInfo["machineName"]))
        print("# Target structure: '{:s}'".format(planInfo["targetStructName"]))
        print("# Number of fractions: {:d}".format(planInfo["fractionNo"]))
        print("# Dose pres. (all fractions):    {:.3f} Gy RBE".format(np.round(planInfo["dosePrescribed"], 3)))
        print("# Dose pres. (single fraction):  {:.3f} Gy RBE".format(np.round(planInfo["dosePrescribed"] / planInfo["fractionNo"], 3)))
        print("# Number of treatment fields: {:d}".format(planInfo["treatmentFieldsNumber"]))
        print("# Number of setup fields:     {:d}".format(planInfo["setupFieldsNumber"]))
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
        Displays a summary of the function results. (def. False)

    Returns
    -------
    DataFrame
        Pandas DataFrame with structures and properties.
    """
    import pandas as pd
    from dicompylercore import dicomparser
    import fredtools as ft
    import pydicom as dicom
    import numpy as np

    rtss = dicomparser.DicomParser(fileName)
    structs = rtss.GetStructures()
    dicomTags = dicom.read_file(fileName)

    ROITable = pd.DataFrame()

    for struct in structs:
        ROIinstance = pd.DataFrame(
            {
                "ID": structs[struct]["id"],
                "ROIType": "unclasified" if not structs[struct]["type"] else structs[struct]["type"],
                "ROIName": structs[struct]["name"],
                "ROIColor": [structs[struct]["color"]],
            }
        )

        ROITable = pd.concat([ROITable, ROIinstance], ignore_index=True)

    # get Physical property for ROI if defined
    ROITable["ROIPhysicalProperty"] = np.nan
    ROITable["ROIPhysicalPropertyValue"] = np.nan
    if "RTROIObservationsSequence" in dicomTags:
        for index, ROIinstance in ROITable.iterrows():
            # get RTROIObservationSequence for the ROIInstance
            for RTROIObservationSequence in dicomTags.RTROIObservationsSequence:
                if RTROIObservationSequence.ReferencedROINumber == ROIinstance.ID:
                    break

            if not "ROIPhysicalPropertiesSequence" in RTROIObservationSequence:
                continue
            else:
                ROIPhysicalProperties = RTROIObservationSequence.ROIPhysicalPropertiesSequence[0]

            if "ROIPhysicalProperty" in ROIPhysicalProperties:
                ROITable.loc[index, "ROIPhysicalProperty"] = ROIPhysicalProperties.ROIPhysicalProperty
            if "ROIPhysicalPropertyValue" in ROIPhysicalProperties:
                ROITable.loc[index, "ROIPhysicalPropertyValue"] = ROIPhysicalProperties.ROIPhysicalPropertyValue

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
        if not _isDicomCT(fileName):
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
    import warnings

    if not _isDicomRS(RSfileName):
        raise ValueError(f"The file {RSfileName} is not a proper dicom describing structures.")

    dicomRS = dicom.read_file(RSfileName)

    for StructureSetROISequence in dicomRS.StructureSetROISequence:
        if StructureSetROISequence.ROIName == structName:
            # get the ROInumber
            ROINumber = StructureSetROISequence.ROINumber
            # get ROIGenerationAlgorithm if available
            try:
                GenerationAlgorithm = StructureSetROISequence.ROIGenerationAlgorithm
            except:
                GenerationAlgorithm = ""
            break
        else:
            ROINumber = None

    # raise error if no structName found
    if ROINumber is None:
        raise ValueError(f"The structure named '{structName}' is not in the {RSfileName}.")

    ROIinfo = {"Number": int(ROINumber), "Name": structName, "GenerationAlgorithm": GenerationAlgorithm}

    # get ROI type
    for RTROIObservationsSequence in dicomRS.RTROIObservationsSequence:
        if RTROIObservationsSequence.ReferencedROINumber == ROINumber:
            ROIinfo["Type"] = RTROIObservationsSequence.RTROIInterpretedType

    for ROIContourSequence in dicomRS.ROIContourSequence:
        if ROIContourSequence.ReferencedROINumber == ROINumber:
            ROIinfo["Color"] = ROIContourSequence.ROIDisplayColor
            if "ContourSequence" in ROIContourSequence:
                ContourSequence = ROIContourSequence.ContourSequence
            else:
                warnings.warn("Warning: No 'ContourSequence' defined in 'ROIContourSequence' for the structure '{:s}'. An empty StructureContours will be returned.".format(ROIinfo["Name"]))
                ContourSequence=[]
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
        if not _isDicomRD(fileName):
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
