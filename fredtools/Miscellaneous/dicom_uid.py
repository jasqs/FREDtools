from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)

@overload
def getSOPInstanceUID(fileNames: Iterable[PathLike], displayInfo: bool = False) -> List[DicomUID]: ...

@overload
def getSOPInstanceUID(fileNames: PathLike, displayInfo: bool = False) -> DicomUID: ...


def getSOPInstanceUID(fileNames: PathLike | Iterable[PathLike], displayInfo: bool = False) -> DicomUID | List[DicomUID]:
    r"""Get the SOPInstanceUID from dicom files.

    The function reads the SOPInstanceUID tag from a dicom file or an iterable
    of dicom files of any type (CT, RS, RN, RD, etc.). The UIDs are returned
    as instances of pydicom.uid.UID, which is a subclass of str, therefore
    the UIDs can be compared directly with the '==' operator.

    Parameters
    ----------
    fileNames : path or iterable of paths
        A path or an iterable of paths to dicom files.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    UID or list of UIDs
        A single UID if a single path was given, or a list of UIDs (possibly
        empty) if an iterable of paths was given.

    Raises
    ------
    ValueError
        If the tag 'SOPInstanceUID' cannot be found in any of the dicom files.

    See Also
    --------
    getRNReferencedStructureSetUID : get the SOPInstanceUID of the structure set (RS) referenced in a plan (RN) dicom.
    getRSReferencedImageUIDs : get the SOPInstanceUIDs of the images (e.g. CT) referenced in a structure set (RS) dicom.
    sortDicoms : sort dicom files in a folder by type.
    """
    import pydicom as dicom

    # if fileNames is a single path then make it a single element list
    if singleFileName := isinstance(fileNames, PathLike):
        fileNames = [fileNames]
    else:
        fileNames = list(fileNames)

    SOPInstanceUIDs = []
    for fileName in fileNames:
        dicomTags = dicom.dcmread(fileName, specific_tags=["SOPInstanceUID"], stop_before_pixels=True)

        # check if SOPInstanceUID exists in the tags
        if "SOPInstanceUID" not in dicomTags:
            error = ValueError(f"Cannot find tag 'SOPInstanceUID' in the dicom file {fileName}.")
            _logger.error(error)
            raise error

        if not dicomTags.SOPInstanceUID.is_valid:
            _logger.warning(f"The SOPInstanceUID '{dicomTags.SOPInstanceUID}' read from the dicom file {fileName} is not a valid UID.")

        SOPInstanceUIDs.append(dicomTags.SOPInstanceUID)

    if displayInfo:
        _logger.info(f"Read SOPInstanceUID from {len(SOPInstanceUIDs)} dicom file{'' if len(SOPInstanceUIDs) == 1 else 's'}.")

    return SOPInstanceUIDs[0] if singleFileName else SOPInstanceUIDs


def getRNReferencedStructureSetUID(fileName: PathLike, displayInfo: bool = False) -> DicomUID:
    r"""Get the SOPInstanceUID of the structure set referenced in a plan dicom.

    The function reads the ReferencedSOPInstanceUID of the structure set (RS)
    dicom referenced in a dicom file with an RT plan (RN). The UID is returned
    as an instance of pydicom.uid.UID and should match the SOPInstanceUID of
    the structure set dicom that the plan was created for.

    Parameters
    ----------
    fileName : path
        Path to a dicom file with an RT plan (RN file).
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    UID
        The SOPInstanceUID of the referenced structure set.

    Raises
    ------
    TypeError
        If the dicom file is not of an RT plan (RN) type.
    ValueError
        If no 'ReferencedStructureSetSequence' item can be found in the dicom file.

    See Also
    --------
    getSOPInstanceUID : get the SOPInstanceUID from dicom files.
    getRSReferencedImageUIDs : get the SOPInstanceUIDs of the images (e.g. CT) referenced in a structure set (RS) dicom.
    sortDicoms : sort dicom files in a folder by type.
    """
    import pydicom as dicom
    import fredtools as ft

    # check if dicom is RN
    ft.ImgIO.dicom_io._isDicomRN(fileName, raiseError=True)

    dicomTags = dicom.dcmread(fileName, specific_tags=["ReferencedStructureSetSequence"], stop_before_pixels=True)

    # check if ReferencedStructureSetSequence exists in the tags and is not empty
    if "ReferencedStructureSetSequence" not in dicomTags or len(dicomTags.ReferencedStructureSetSequence) == 0:
        error = ValueError(f"Cannot find any 'ReferencedStructureSetSequence' item in the dicom file {fileName}.")
        _logger.error(error)
        raise error

    if len(dicomTags.ReferencedStructureSetSequence) > 1:
        _logger.warning(f"The dicom file {fileName} contains multiple ReferencedStructureSetSequence items. The first one was used.")

    ReferencedSOPInstanceUID = dicomTags.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID

    if displayInfo:
        _logger.info(f"SOPInstanceUID of the referenced structure set: '{ReferencedSOPInstanceUID}'")

    return ReferencedSOPInstanceUID


def getRSReferencedImageUIDs(fileName: PathLike, displayInfo: bool = False) -> List[DicomUID]:
    r"""Get the SOPInstanceUIDs of the images referenced in a structure set dicom.

    The function reads the ReferencedSOPInstanceUIDs of all the images
    (usually CT slices) referenced in the contour image sequences of a dicom
    file with a structure set (RS). The UIDs are returned as instances of
    pydicom.uid.UID and should match the SOPInstanceUIDs of the image dicoms
    that the structure set was created for.

    Parameters
    ----------
    fileName : path
        Path to a dicom file with a structure set (RS file).
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    list of UIDs
        A list (possibly empty) of the SOPInstanceUIDs of the referenced images.

    Raises
    ------
    TypeError
        If the dicom file is not of a structure set (RS) type.
    ValueError
        If the tag 'ReferencedFrameOfReferenceSequence' cannot be found in the dicom file.

    See Also
    --------
    getSOPInstanceUID : get the SOPInstanceUID from dicom files.
    getRNReferencedStructureSetUID : get the SOPInstanceUID of the structure set (RS) referenced in a plan (RN) dicom.
    sortDicoms : sort dicom files in a folder by type.
    """
    import pydicom as dicom
    import fredtools as ft

    # check if dicom is RS
    ft.ImgIO.dicom_io._isDicomRS(fileName, raiseError=True)

    dicomTags = dicom.dcmread(fileName, specific_tags=["ReferencedFrameOfReferenceSequence"], stop_before_pixels=True)

    # check if ReferencedFrameOfReferenceSequence exists in the tags
    if "ReferencedFrameOfReferenceSequence" not in dicomTags:
        error = ValueError(f"Cannot find tag 'ReferencedFrameOfReferenceSequence' in the dicom file {fileName}.")
        _logger.error(error)
        raise error

    # collect the referenced image UIDs from all frame of reference/study/series items
    ReferencedSOPInstanceUIDs = []
    referencedSeriesNo = 0
    for ReferencedFrameOfReference in dicomTags.ReferencedFrameOfReferenceSequence:
        if "RTReferencedStudySequence" not in ReferencedFrameOfReference:
            continue
        for RTReferencedStudy in ReferencedFrameOfReference.RTReferencedStudySequence:
            if "RTReferencedSeriesSequence" not in RTReferencedStudy:
                continue
            for RTReferencedSeries in RTReferencedStudy.RTReferencedSeriesSequence:
                referencedSeriesNo += 1
                if "ContourImageSequence" not in RTReferencedSeries:
                    continue
                for ContourImage in RTReferencedSeries.ContourImageSequence:
                    ReferencedSOPInstanceUIDs.append(ContourImage.ReferencedSOPInstanceUID)

    if referencedSeriesNo > 1:
        _logger.warning(f"The dicom file {fileName} references multiple image series. The referenced image UIDs of all the series were returned.")

    if len(ReferencedSOPInstanceUIDs) == 0:
        _logger.warning(f"No referenced image UIDs were found in the dicom file {fileName}.")

    if displayInfo:
        _logger.info(f"Found {len(ReferencedSOPInstanceUIDs)} referenced image{'' if len(ReferencedSOPInstanceUIDs) == 1 else 's'} in {referencedSeriesNo} referenced series.")

    return ReferencedSOPInstanceUIDs


def getRDReferencedPlanUID(fileName: PathLike, displayInfo: bool = False) -> DicomUID:
    r"""Get the SOPInstanceUID of the plan referenced in a dose dicom.

    The function reads the ReferencedSOPInstanceUID of the plan (RN) dicom
    referenced in a dicom file with a dose distribution (RD). The UID is
    returned as an instance of pydicom.uid.UID and should match the
    SOPInstanceUID of the plan dicom that the dose was calculated for.

    Parameters
    ----------
    fileName : path
        Path to a dicom file with a dose distribution (RD file).
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    UID
        The SOPInstanceUID of the referenced plan.

    Raises
    ------
    TypeError
        If the dicom file is not of a dose (RD) type.
    ValueError
        If no 'ReferencedRTPlanSequence' item can be found in the dicom file.

    See Also
    --------
    getSOPInstanceUID : get the SOPInstanceUID from dicom files.
    getRNReferencedStructureSetUID : get the SOPInstanceUID of the structure set (RS) referenced in a plan (RN) dicom.
    sortDicoms : sort dicom files in a folder by type.
    """
    import pydicom as dicom
    import fredtools as ft

    # check if dicom is RD
    ft.ImgIO.dicom_io._isDicomRD(fileName, raiseError=True)

    dicomTags = dicom.dcmread(fileName, specific_tags=["ReferencedRTPlanSequence"], stop_before_pixels=True)

    # check if ReferencedRTPlanSequence exists in the tags and is not empty
    if "ReferencedRTPlanSequence" not in dicomTags or len(dicomTags.ReferencedRTPlanSequence) == 0:
        error = ValueError(f"Cannot find any 'ReferencedRTPlanSequence' item in the dicom file {fileName}.")
        _logger.error(error)
        raise error

    if len(dicomTags.ReferencedRTPlanSequence) > 1:
        _logger.warning(f"The dicom file {fileName} contains multiple ReferencedRTPlanSequence items. The first one was used.")

    ReferencedSOPInstanceUID = dicomTags.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID

    if displayInfo:
        _logger.info(f"SOPInstanceUID of the referenced plan: '{ReferencedSOPInstanceUID}'")

    return ReferencedSOPInstanceUID


@overload
def getFrameOfReferenceUID(fileNames: Iterable[PathLike], displayInfo: bool = False) -> List[DicomUID]: ...

@overload
def getFrameOfReferenceUID(fileNames: PathLike, displayInfo: bool = False) -> DicomUID: ...


def getFrameOfReferenceUID(fileNames: PathLike | Iterable[PathLike], displayInfo: bool = False) -> DicomUID | List[DicomUID]:
    r"""Get the FrameOfReferenceUID from dicom files.

    The function reads the FrameOfReferenceUID tag from a dicom file or an
    iterable of dicom files of any type (CT, RS, RN, RD, etc.). For structure
    set (RS) dicoms without a top-level FrameOfReferenceUID, the UID is read
    from the first item of the ReferencedFrameOfReferenceSequence. The UIDs
    are returned as instances of pydicom.uid.UID and should be the same for
    all the dicoms describing the same treatment plan.

    Parameters
    ----------
    fileNames : path or iterable of paths
        A path or an iterable of paths to dicom files.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    UID or list of UIDs
        A single UID if a single path was given, or a list of UIDs (possibly
        empty) if an iterable of paths was given.

    Raises
    ------
    ValueError
        If no FrameOfReferenceUID can be found in any of the dicom files.

    See Also
    --------
    getSOPInstanceUID : get the SOPInstanceUID from dicom files.
    sortDicoms : sort dicom files in a folder by type.
    """
    import pydicom as dicom

    # if fileNames is a single path then make it a single element list
    if singleFileName := isinstance(fileNames, PathLike):
        fileNames = [fileNames]
    else:
        fileNames = list(fileNames)

    FrameOfReferenceUIDs = []
    for fileName in fileNames:
        dicomTags = dicom.dcmread(fileName, specific_tags=["FrameOfReferenceUID", "ReferencedFrameOfReferenceSequence"], stop_before_pixels=True)

        if "FrameOfReferenceUID" in dicomTags and dicomTags.FrameOfReferenceUID:
            FrameOfReferenceUIDs.append(dicomTags.FrameOfReferenceUID)
        elif "ReferencedFrameOfReferenceSequence" in dicomTags and len(dicomTags.ReferencedFrameOfReferenceSequence) > 0:
            if len(dicomTags.ReferencedFrameOfReferenceSequence) > 1:
                _logger.warning(f"The dicom file {fileName} contains multiple ReferencedFrameOfReferenceSequence items. The first one was used.")
            FrameOfReferenceUIDs.append(dicomTags.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID)
        else:
            error = ValueError(f"Cannot find any FrameOfReferenceUID in the dicom file {fileName}.")
            _logger.error(error)
            raise error

    if displayInfo:
        _logger.info(f"Read FrameOfReferenceUID from {len(FrameOfReferenceUIDs)} dicom file{'' if len(FrameOfReferenceUIDs) == 1 else 's'}.")

    return FrameOfReferenceUIDs[0] if singleFileName else FrameOfReferenceUIDs


def checkUID_RNtoRS(RNfileName: PathLike, RSfileName: PathLike) -> bool:
    r"""Check if the structure set referenced by a plan dicom matches a structure set dicom.

    The function validates that the given files are of the correct dicom type,
    then compares the ReferencedSOPInstanceUID of the structure set (RS) dicom
    referenced in the plan (RN) dicom, obtained with getRNReferencedStructureSetUID,
    with the SOPInstanceUID of the given RS dicom, obtained with getSOPInstanceUID.
    The comparison result is logged at the debug level only; no info or warning
    is logged by this function, so it is up to the calling code to log accordingly.

    Parameters
    ----------
    RNfileName : path
        Path to a dicom file with an RT plan (RN file).
    RSfileName : path
        Path to a dicom file with a structure set (RS file).

    Returns
    -------
    bool
        True if the RN dicom references the given RS dicom, False otherwise.

    Raises
    ------
    TypeError
        If RNfileName is not an RT plan dicom, or RSfileName is not a structure set dicom.
    ValueError
        If the RN dicom does not contain a ReferencedStructureSetSequence, or if
        the SOPInstanceUID tag cannot be found in the RS dicom.

    See Also
    --------
    checkUID_RStoCT : check if the images referenced by a structure set dicom match a set of CT dicoms.
    checkUID_RNtoRD : check if the plan referenced by dose dicoms matches a plan dicom.
    getRNReferencedStructureSetUID : get the SOPInstanceUID of the structure set referenced in a plan dicom.
    getSOPInstanceUID : get the SOPInstanceUID from dicom files.
    """
    import fredtools as ft

    # validate modality of both dicoms explicitly (getSOPInstanceUID below does not check modality on its own)
    ft.ImgIO.dicom_io._isDicomRN(RNfileName, raiseError=True)
    ft.ImgIO.dicom_io._isDicomRS(RSfileName, raiseError=True)

    referencedRSUID = getRNReferencedStructureSetUID(RNfileName)
    RSUID = getSOPInstanceUID(RSfileName)
    matching = referencedRSUID == RSUID
    _logger.debug(f"RN {RNfileName} references structure set UID '{referencedRSUID}'; RS {RSfileName} has UID '{RSUID}'. Matching: {matching}.")

    return matching


def checkUID_RStoCT(RSfileName: PathLike, CTfileNames: PathLike | Iterable[PathLike]) -> bool:
    r"""Check if the images referenced by a structure set dicom match a set of CT dicoms.

    The function validates that the given files are of the correct dicom type,
    then compares the SOPInstanceUIDs of the images referenced in the contour
    image sequences of the structure set (RS) dicom, obtained with
    getRSReferencedImageUIDs, with the SOPInstanceUIDs of the given CT dicoms,
    obtained with getSOPInstanceUID. The comparison is order-independent (both
    UID lists are sorted before comparing). The comparison result is logged at
    the debug level only, so it is up to the calling code to log the result
    accordingly. However, a warning is logged if the CT dicoms contain images
    not referenced in the RS dicom or if duplicated SOPInstanceUIDs are found
    among the CT dicoms.

    Parameters
    ----------
    RSfileName : path
        Path to a dicom file with a structure set (RS file).
    CTfileNames : path or iterable of paths
        A path or an iterable of paths to CT image dicom files.

    Returns
    -------
    bool
        True if the set of CT dicoms exactly matches the images referenced by
        the RS dicom, False otherwise.

    Raises
    ------
    TypeError
        If RSfileName is not a structure set dicom, or any of CTfileNames is
        not a CT image dicom.
    ValueError
        If the RS dicom does not contain a ReferencedFrameOfReferenceSequence,
        or if the SOPInstanceUID tag cannot be found in a CT dicom.

    See Also
    --------
    checkUID_RNtoRS : check if the structure set referenced by a plan dicom matches a structure set dicom.
    checkUID_RNtoRD : check if the plan referenced by dose dicoms matches a plan dicom.
    getRSReferencedImageUIDs : get the SOPInstanceUIDs of the images referenced in a structure set dicom.
    getSOPInstanceUID : get the SOPInstanceUID from dicom files.
    """
    import fredtools as ft

    # normalize CTfileNames to a list (sortDicoms squashes a single-file result to a bare string)
    CTfileNames = [CTfileNames] if isinstance(CTfileNames, PathLike) else list(CTfileNames)

    # validate modality of the RS dicom and every CT dicom explicitly
    ft.ImgIO.dicom_io._isDicomRS(RSfileName, raiseError=True)
    for CTfileName in CTfileNames:
        ft.ImgIO.dicom_io._isDicomCT(CTfileName, raiseError=True)

    referencedImageUIDs = sorted(getRSReferencedImageUIDs(RSfileName))
    CTUIDs = sorted(getSOPInstanceUID(CTfileNames))

    # warn about data anomalies which cannot be recognised from the comparison result alone
    if len(set(CTUIDs)) < len(CTUIDs):
        _logger.warning(f"The CT dicoms contain {len(CTUIDs) - len(set(CTUIDs))} duplicated SOPInstanceUIDs.")
    if extraCTUIDsNo := len(set(CTUIDs) - set(referencedImageUIDs)):
        _logger.warning(f"{extraCTUIDsNo} CT dicom{' is' if extraCTUIDsNo == 1 else 's are'} not referenced in the RS dicom {RSfileName}.")

    matching = referencedImageUIDs == CTUIDs
    _logger.debug(f"RS {RSfileName} references {len(referencedImageUIDs)} image UIDs; found {len(CTUIDs)} CT UIDs. Matching: {matching}.")

    return matching


def checkUID_RNtoRD(RNfileName: PathLike, RDfileNames: PathLike | Iterable[PathLike]) -> bool:
    r"""Check if the plan referenced by dose dicoms matches a plan dicom.

    The function validates that the given files are of the correct dicom type,
    then compares the ReferencedSOPInstanceUID of the plan (RN) dicom
    referenced in each dose (RD) dicom, obtained with getRDReferencedPlanUID,
    with the SOPInstanceUID of the given RN dicom, obtained with
    getSOPInstanceUID. The comparison result is logged at the debug level
    only; no info or warning is logged by this function, so it is up to the
    calling code to log accordingly.

    Parameters
    ----------
    RNfileName : path
        Path to a dicom file with an RT plan (RN file).
    RDfileNames : path or iterable of paths
        A path or an iterable of paths to dose (RD) dicom files.

    Returns
    -------
    bool
        True if every RD dicom references the given RN dicom, False otherwise.

    Raises
    ------
    TypeError
        If RNfileName is not an RT plan dicom, or any of RDfileNames is not
        a dose dicom.
    ValueError
        If no RD file names are provided, if an RD dicom does not contain
        a ReferencedRTPlanSequence, or if the SOPInstanceUID tag cannot be
        found in the RN dicom.

    See Also
    --------
    checkUID_RNtoRS : check if the structure set referenced by a plan dicom matches a structure set dicom.
    checkUID_RStoCT : check if the images referenced by a structure set dicom match a set of CT dicoms.
    getRDReferencedPlanUID : get the SOPInstanceUID of the plan referenced in a dose dicom.
    getSOPInstanceUID : get the SOPInstanceUID from dicom files.
    """
    import fredtools as ft

    # normalize RDfileNames to a list (sortDicoms squashes a single-file result to a bare string)
    RDfileNames = [RDfileNames] if isinstance(RDfileNames, PathLike) else list(RDfileNames)

    if len(RDfileNames) == 0:
        error = ValueError("No RD file names were provided.")
        _logger.error(error)
        raise error

    # validate modality of the RN dicom and every RD dicom explicitly
    ft.ImgIO.dicom_io._isDicomRN(RNfileName, raiseError=True)
    for RDfileName in RDfileNames:
        ft.ImgIO.dicom_io._isDicomRD(RDfileName, raiseError=True)

    RNUID = getSOPInstanceUID(RNfileName)
    matchingNo = sum(getRDReferencedPlanUID(RDfileName) == RNUID for RDfileName in RDfileNames)
    matching = matchingNo == len(RDfileNames)
    _logger.debug(f"{matchingNo} of {len(RDfileNames)} RD dicoms reference the RN {RNfileName} with UID '{RNUID}'. Matching: {matching}.")

    return matching
