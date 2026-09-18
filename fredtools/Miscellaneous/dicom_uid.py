from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def _isSingleDicomVar(dicomVars: PathLike | DicomDataset | Iterable[PathLike | DicomDataset]) -> bool:
    r"""Check if the variable describes a single dicom and not an iterable of dicoms.

    A single dicom is either a path to a dicom file or dicom tags already read
    by pydicom.dcmread. The explicit check for the dicom dataset is needed
    because a pydicom Dataset is iterable (it yields its data elements), so it
    would otherwise be mistaken for an iterable of dicoms.
    """
    return isinstance(dicomVars, PathLike) or isinstance(dicomVars, DicomDataset)


def _asDicomVarList(dicomVars: PathLike | DicomDataset | Iterable[PathLike | DicomDataset]) -> List[PathLike | DicomDataset]:
    r"""Normalise a single dicom or an iterable of dicoms to a list of dicoms."""
    if _isSingleDicomVar(dicomVars):
        return [cast(PathLike | DicomDataset, dicomVars)]

    return list(cast(Iterable[PathLike | DicomDataset], dicomVars))


def _getDicomTags(dicomVar: PathLike | DicomDataset, specificTags: Iterable[str] | None = None) -> DicomDataset:
    r"""Get the dicom tags of a dicom given as a path or as tags.

    If the dicom is given as dicom tags already read by pydicom.dcmread, then
    the tags are returned unchanged and no file is read. This makes it possible
    to read a dicom file only once and reuse the tags for all the consecutive
    queries.
    """
    import pydicom as dicom

    if isinstance(dicomVar, DicomDataset):
        return dicomVar

    return dicom.dcmread(dicomVar, specific_tags=list(specificTags) if specificTags else None, stop_before_pixels=True)


def _describeDicomVar(dicomVar: PathLike | DicomDataset) -> str:
    r"""Describe a dicom given as a path or as tags, to be used in logging messages."""
    return f"file {dicomVar}" if isinstance(dicomVar, PathLike) else "dataset"


@overload
def getSOPInstanceUID(dicomVars: Iterable[PathLike | DicomDataset], displayInfo: bool = False) -> List[DicomUID]: ...

@overload
def getSOPInstanceUID(dicomVars: PathLike | DicomDataset, displayInfo: bool = False) -> DicomUID: ...


def getSOPInstanceUID(dicomVars: PathLike | DicomDataset | Iterable[PathLike | DicomDataset], displayInfo: bool = False) -> DicomUID | List[DicomUID]:
    r"""Get the SOPInstanceUID from dicoms.

    The function reads the SOPInstanceUID tag from a dicom or an iterable
    of dicoms of any type (CT, RS, RN, RD, etc.), given as paths to dicom files
    or as dicom tags already read by pydicom.dcmread. The UIDs are returned
    as instances of pydicom.uid.UID, which is a subclass of str, therefore
    the UIDs can be compared directly with the '==' operator.

    Parameters
    ----------
    dicomVars : path, tags or iterable of paths or tags
        A path to a dicom file, a dicom tag structure read by pydicom.dcmread,
        or an iterable of those. No file is read for the dicoms given as tags.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    UID or list of UIDs
        A single UID if a single dicom was given, or a list of UIDs (possibly
        empty) if an iterable of dicoms was given.

    Raises
    ------
    ValueError
        If the tag 'SOPInstanceUID' cannot be found in any of the dicoms.

    See Also
    --------
    getRNReferencedStructureSetUID : get the SOPInstanceUID of the structure set (RS) referenced in a plan (RN) dicom.
    getRSReferencedImageUIDs : get the SOPInstanceUIDs of the images (e.g. CT) referenced in a structure set (RS) dicom.
    getDicomsInfo : read the identity and the references of all the dicoms in a folder in a single pass.
    sortDicoms : sort dicom files in a folder by type.
    """
    # if a single dicom is given then make it a single element list
    singleDicomVar = _isSingleDicomVar(dicomVars)
    dicomVarList = _asDicomVarList(dicomVars)

    SOPInstanceUIDs = []
    for dicomVar in dicomVarList:
        dicomTags = _getDicomTags(dicomVar, ["SOPInstanceUID"])

        # check if SOPInstanceUID exists in the tags
        if "SOPInstanceUID" not in dicomTags:
            error = ValueError(f"Cannot find tag 'SOPInstanceUID' in the dicom {_describeDicomVar(dicomVar)}.")
            _logger.error(error)
            raise error

        if not dicomTags.SOPInstanceUID.is_valid:
            _logger.warning(f"The SOPInstanceUID '{dicomTags.SOPInstanceUID}' read from the dicom {_describeDicomVar(dicomVar)} is not a valid UID.")

        SOPInstanceUIDs.append(dicomTags.SOPInstanceUID)

    if displayInfo:
        _logger.info(f"Read SOPInstanceUID from {len(SOPInstanceUIDs)} dicom{'' if len(SOPInstanceUIDs) == 1 else 's'}.")

    return SOPInstanceUIDs[0] if singleDicomVar else SOPInstanceUIDs


def getRNReferencedStructureSetUID(dicomVar: PathLike | DicomDataset, displayInfo: bool = False) -> DicomUID:
    r"""Get the SOPInstanceUID of the structure set referenced in a plan dicom.

    The function reads the ReferencedSOPInstanceUID of the structure set (RS)
    dicom referenced in a dicom with an RT plan (RN), given as a path to a dicom
    file or as dicom tags already read by pydicom.dcmread. The UID is returned
    as an instance of pydicom.uid.UID and should match the SOPInstanceUID of
    the structure set dicom that the plan was created for.

    Parameters
    ----------
    dicomVar : path or tags
        A path to a dicom file with an RT plan (RN file), or a dicom tag
        structure read by pydicom.dcmread. No file is read for tags.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    UID
        The SOPInstanceUID of the referenced structure set.

    Raises
    ------
    TypeError
        If the dicom is not of an RT plan (RN) type.
    ValueError
        If no 'ReferencedStructureSetSequence' item can be found in the dicom.

    See Also
    --------
    getSOPInstanceUID : get the SOPInstanceUID from dicoms.
    getRSReferencedImageUIDs : get the SOPInstanceUIDs of the images (e.g. CT) referenced in a structure set (RS) dicom.
    getDicomsInfo : read the identity and the references of all the dicoms in a folder in a single pass.
    sortDicoms : sort dicom files in a folder by type.
    """
    import fredtools as ft

    dicomTags = _getDicomTags(dicomVar, ["SOPClassUID", "ReferencedStructureSetSequence"])

    # check if dicom is RN
    ft.ImgIO.dicom_io._isDicomRN(dicomTags, raiseError=True)

    # check if ReferencedStructureSetSequence exists in the tags and is not empty
    if "ReferencedStructureSetSequence" not in dicomTags or len(dicomTags.ReferencedStructureSetSequence) == 0:
        error = ValueError(f"Cannot find any 'ReferencedStructureSetSequence' item in the dicom {_describeDicomVar(dicomVar)}.")
        _logger.error(error)
        raise error

    if len(dicomTags.ReferencedStructureSetSequence) > 1:
        _logger.warning(f"The dicom {_describeDicomVar(dicomVar)} contains multiple ReferencedStructureSetSequence items. The first one was used.")

    ReferencedSOPInstanceUID = dicomTags.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID

    if displayInfo:
        _logger.info(f"SOPInstanceUID of the referenced structure set: '{ReferencedSOPInstanceUID}'")

    return ReferencedSOPInstanceUID


def getRSReferencedImageUIDs(dicomVar: PathLike | DicomDataset, displayInfo: bool = False) -> List[DicomUID]:
    r"""Get the SOPInstanceUIDs of the images referenced in a structure set dicom.

    The function reads the ReferencedSOPInstanceUIDs of all the images
    (usually CT slices) referenced in the contour image sequences of a dicom
    with a structure set (RS), given as a path to a dicom file or as dicom tags
    already read by pydicom.dcmread. The UIDs are returned as instances of
    pydicom.uid.UID and should match the SOPInstanceUIDs of the image dicoms
    that the structure set was created for.

    Parameters
    ----------
    dicomVar : path or tags
        A path to a dicom file with a structure set (RS file), or a dicom tag
        structure read by pydicom.dcmread. No file is read for tags.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    list of UIDs
        A list (possibly empty) of the SOPInstanceUIDs of the referenced images.

    Raises
    ------
    TypeError
        If the dicom is not of a structure set (RS) type.
    ValueError
        If the tag 'ReferencedFrameOfReferenceSequence' cannot be found in the dicom.

    See Also
    --------
    getSOPInstanceUID : get the SOPInstanceUID from dicoms.
    getRNReferencedStructureSetUID : get the SOPInstanceUID of the structure set (RS) referenced in a plan (RN) dicom.
    getDicomsInfo : read the identity and the references of all the dicoms in a folder in a single pass.
    sortDicoms : sort dicom files in a folder by type.
    """
    import fredtools as ft

    dicomTags = _getDicomTags(dicomVar, ["SOPClassUID", "ReferencedFrameOfReferenceSequence"])

    # check if dicom is RS
    ft.ImgIO.dicom_io._isDicomRS(dicomTags, raiseError=True)

    # check if ReferencedFrameOfReferenceSequence exists in the tags
    if "ReferencedFrameOfReferenceSequence" not in dicomTags:
        error = ValueError(f"Cannot find tag 'ReferencedFrameOfReferenceSequence' in the dicom {_describeDicomVar(dicomVar)}.")
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
        _logger.warning(f"The dicom {_describeDicomVar(dicomVar)} references multiple image series. The referenced image UIDs of all the series were returned.")

    if len(ReferencedSOPInstanceUIDs) == 0:
        _logger.warning(f"No referenced image UIDs were found in the dicom {_describeDicomVar(dicomVar)}.")

    if displayInfo:
        _logger.info(f"Found {len(ReferencedSOPInstanceUIDs)} referenced image{'' if len(ReferencedSOPInstanceUIDs) == 1 else 's'} in {referencedSeriesNo} referenced series.")

    return ReferencedSOPInstanceUIDs


def getRDReferencedPlanUID(dicomVar: PathLike | DicomDataset, displayInfo: bool = False) -> DicomUID:
    r"""Get the SOPInstanceUID of the plan referenced in a dose dicom.

    The function reads the ReferencedSOPInstanceUID of the plan (RN) dicom
    referenced in a dicom with a dose distribution (RD), given as a path to
    a dicom file or as dicom tags already read by pydicom.dcmread. The UID is
    returned as an instance of pydicom.uid.UID and should match the
    SOPInstanceUID of the plan dicom that the dose was calculated for.

    Parameters
    ----------
    dicomVar : path or tags
        A path to a dicom file with a dose distribution (RD file), or a dicom
        tag structure read by pydicom.dcmread. No file is read for tags.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    UID
        The SOPInstanceUID of the referenced plan.

    Raises
    ------
    TypeError
        If the dicom is not of a dose (RD) type.
    ValueError
        If no 'ReferencedRTPlanSequence' item can be found in the dicom.

    See Also
    --------
    getSOPInstanceUID : get the SOPInstanceUID from dicoms.
    getRNReferencedStructureSetUID : get the SOPInstanceUID of the structure set (RS) referenced in a plan (RN) dicom.
    getDicomsInfo : read the identity and the references of all the dicoms in a folder in a single pass.
    sortDicoms : sort dicom files in a folder by type.
    """
    import fredtools as ft

    dicomTags = _getDicomTags(dicomVar, ["SOPClassUID", "ReferencedRTPlanSequence"])

    # check if dicom is RD
    ft.ImgIO.dicom_io._isDicomRD(dicomTags, raiseError=True)

    # check if ReferencedRTPlanSequence exists in the tags and is not empty
    if "ReferencedRTPlanSequence" not in dicomTags or len(dicomTags.ReferencedRTPlanSequence) == 0:
        error = ValueError(f"Cannot find any 'ReferencedRTPlanSequence' item in the dicom {_describeDicomVar(dicomVar)}.")
        _logger.error(error)
        raise error

    if len(dicomTags.ReferencedRTPlanSequence) > 1:
        _logger.warning(f"The dicom {_describeDicomVar(dicomVar)} contains multiple ReferencedRTPlanSequence items. The first one was used.")

    ReferencedSOPInstanceUID = dicomTags.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID

    if displayInfo:
        _logger.info(f"SOPInstanceUID of the referenced plan: '{ReferencedSOPInstanceUID}'")

    return ReferencedSOPInstanceUID


@overload
def getFrameOfReferenceUID(dicomVars: Iterable[PathLike | DicomDataset], displayInfo: bool = False) -> List[DicomUID]: ...

@overload
def getFrameOfReferenceUID(dicomVars: PathLike | DicomDataset, displayInfo: bool = False) -> DicomUID: ...


def getFrameOfReferenceUID(dicomVars: PathLike | DicomDataset | Iterable[PathLike | DicomDataset], displayInfo: bool = False) -> DicomUID | List[DicomUID]:
    r"""Get the FrameOfReferenceUID from dicoms.

    The function reads the FrameOfReferenceUID tag from a dicom or an
    iterable of dicoms of any type (CT, RS, RN, RD, etc.), given as paths to
    dicom files or as dicom tags already read by pydicom.dcmread. For structure
    set (RS) dicoms without a top-level FrameOfReferenceUID, the UID is read
    from the first item of the ReferencedFrameOfReferenceSequence. The UIDs
    are returned as instances of pydicom.uid.UID and should be the same for
    all the dicoms describing the same treatment plan.

    Parameters
    ----------
    dicomVars : path, tags or iterable of paths or tags
        A path to a dicom file, a dicom tag structure read by pydicom.dcmread,
        or an iterable of those. No file is read for the dicoms given as tags.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    UID or list of UIDs
        A single UID if a single dicom was given, or a list of UIDs (possibly
        empty) if an iterable of dicoms was given.

    Raises
    ------
    ValueError
        If no FrameOfReferenceUID can be found in any of the dicoms.

    See Also
    --------
    getSOPInstanceUID : get the SOPInstanceUID from dicoms.
    getDicomsInfo : read the identity and the references of all the dicoms in a folder in a single pass.
    sortDicoms : sort dicom files in a folder by type.
    """
    # if a single dicom is given then make it a single element list
    singleDicomVar = _isSingleDicomVar(dicomVars)
    dicomVarList = _asDicomVarList(dicomVars)

    FrameOfReferenceUIDs = []
    for dicomVar in dicomVarList:
        dicomTags = _getDicomTags(dicomVar, ["FrameOfReferenceUID", "ReferencedFrameOfReferenceSequence"])

        if "FrameOfReferenceUID" in dicomTags and dicomTags.FrameOfReferenceUID:
            FrameOfReferenceUIDs.append(dicomTags.FrameOfReferenceUID)
        elif "ReferencedFrameOfReferenceSequence" in dicomTags and len(dicomTags.ReferencedFrameOfReferenceSequence) > 0:
            if len(dicomTags.ReferencedFrameOfReferenceSequence) > 1:
                _logger.warning(f"The dicom {_describeDicomVar(dicomVar)} contains multiple ReferencedFrameOfReferenceSequence items. The first one was used.")
            FrameOfReferenceUIDs.append(dicomTags.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID)
        else:
            error = ValueError(f"Cannot find any FrameOfReferenceUID in the dicom {_describeDicomVar(dicomVar)}.")
            _logger.error(error)
            raise error

    if displayInfo:
        _logger.info(f"Read FrameOfReferenceUID from {len(FrameOfReferenceUIDs)} dicom{'' if len(FrameOfReferenceUIDs) == 1 else 's'}.")

    return FrameOfReferenceUIDs[0] if singleDicomVar else FrameOfReferenceUIDs


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

    # read each dicom only once and reuse the tags for the modality validation and the UID queries
    RNtags = _getDicomTags(RNfileName, ["SOPClassUID", "ReferencedStructureSetSequence"])
    RStags = _getDicomTags(RSfileName, ["SOPClassUID", "SOPInstanceUID"])

    # validate modality of both dicoms explicitly (getSOPInstanceUID below does not check modality on its own)
    ft.ImgIO.dicom_io._isDicomRN(RNtags, raiseError=True)
    ft.ImgIO.dicom_io._isDicomRS(RStags, raiseError=True)

    referencedRSUID = getRNReferencedStructureSetUID(RNtags)
    RSUID = getSOPInstanceUID(RStags)
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

    # read each dicom only once and reuse the tags for the modality validation and the UID queries
    RStags = _getDicomTags(RSfileName, ["SOPClassUID", "ReferencedFrameOfReferenceSequence"])
    CTtags = [_getDicomTags(CTfileName, ["SOPClassUID", "SOPInstanceUID"]) for CTfileName in CTfileNames]

    # validate modality of the RS dicom and every CT dicom explicitly
    ft.ImgIO.dicom_io._isDicomRS(RStags, raiseError=True)
    for CTtag in CTtags:
        ft.ImgIO.dicom_io._isDicomCT(CTtag, raiseError=True)

    referencedImageUIDs = sorted(getRSReferencedImageUIDs(RStags))
    CTUIDs = sorted(getSOPInstanceUID(CTtags))

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

    # read each dicom only once and reuse the tags for the modality validation and the UID queries
    RNtags = _getDicomTags(RNfileName, ["SOPClassUID", "SOPInstanceUID"])
    RDtags = [_getDicomTags(RDfileName, ["SOPClassUID", "ReferencedRTPlanSequence"]) for RDfileName in RDfileNames]

    # validate modality of the RN dicom and every RD dicom explicitly
    ft.ImgIO.dicom_io._isDicomRN(RNtags, raiseError=True)
    for RDtag in RDtags:
        ft.ImgIO.dicom_io._isDicomRD(RDtag, raiseError=True)

    RNUID = getSOPInstanceUID(RNtags)
    matchingNo = sum(getRDReferencedPlanUID(RDtag) == RNUID for RDtag in RDtags)
    matching = matchingNo == len(RDfileNames)
    _logger.debug(f"{matchingNo} of {len(RDfileNames)} RD dicoms reference the RN {RNfileName} with UID '{RNUID}'. Matching: {matching}.")

    return matching


# The dicom types recognised by getDicomsInfo, in the same order and with the same
# substring semantics as the cascade used by sortDicoms, so that both agree on
# how a SOP Class UID name is bucketed. The substring matching is intentional:
# "CT Image Storage" also matches "Enhanced CT Image Storage" and "Plan Storage"
# matches both "RT Plan Storage" and "RT Ion Plan Storage".
_DICOM_TYPE_PATTERNS = (("CT", "CT Image Storage"),
                        ("RS", "Structure Set Storage"),
                        ("RN", "Plan Storage"),
                        ("RD", "Dose Storage"),
                        ("PET", "Positron Emission Tomography Image Storage"))

# The tags read for each dicom type in the second pass of getDicomsInfo. Only
# the RS/RN/RD dicoms carry references to other dicoms, so only those are read
# a second time; the CT and PET dicoms are fully described by the first pass.
_DICOM_TYPE_REFERENCE_TAGS = {"RS": ["ReferencedFrameOfReferenceSequence", "StructureSetDate", "StructureSetTime", "SeriesInstanceUID", "StudyInstanceUID"],
                              "RN": ["ReferencedStructureSetSequence", "IonBeamSequence", "BeamSequence", "SeriesInstanceUID", "StudyInstanceUID"],
                              "RD": ["ReferencedRTPlanSequence", "DoseSummationType", "SeriesInstanceUID", "StudyInstanceUID"]}

_DICOMS_INFO_COLUMNS = ("fileName", "folderName", "SOPClassUID", "dicomTypeName", "dicomType", "SOPInstanceUID",
                        "FrameOfReferenceUID", "referencedSOPInstanceUIDs", "referencedSeriesNo", "beamNumbers",
                        "referencedBeamNumber", "doseSummationType", "structureSetDate", "structureSetTime",
                        "SeriesInstanceUID", "StudyInstanceUID", "error")


def _getDicomTypeFromTypeName(dicomTypeName: str) -> str:
    r"""Bucket a SOP Class UID name into the dicom type used by getDicomsInfo and by sortDicoms."""
    for dicomType, pattern in _DICOM_TYPE_PATTERNS:
        if pattern in dicomTypeName:
            return dicomType

    return "Unknown"


def _getDicomsInfoFileNames(searchFolder: PathLike | Iterable[PathLike], recursive: bool, pattern: str) -> List[str]:
    r"""Resolve the dicom file names to be indexed.

    A folder is searched with pathlib for the given pattern, a file is taken as
    it is, and an iterable may mix both. The file names are returned in the
    order they were found and are deliberately not sorted, so that the order
    of e.g. sortDicoms results is preserved.
    """
    from pathlib import Path

    searchFolders = [searchFolder] if isinstance(searchFolder, PathLike) else list(searchFolder)

    fileNames = []
    for searchFolderItem in searchFolders:
        searchFolderPath = Path(searchFolderItem)
        if searchFolderPath.is_dir():
            fileNames += [str(fileName) for fileName in (searchFolderPath.rglob(pattern) if recursive else searchFolderPath.glob(pattern))]
        else:
            fileNames.append(str(searchFolderPath))

    return fileNames


def _getDicomsInfoIdentity(fileName: str, readFrameOfReferenceUID: bool) -> dict:
    r"""Read the identity of a single dicom file: its type, its SOPInstanceUID and, optionally, its FrameOfReferenceUID.

    When the FrameOfReferenceUID is not requested, only the dicom file meta
    information is read, which is substantially faster than parsing the dataset.
    The DICOM standard (PS3.10 7.1) requires the MediaStorageSOPClassUID and
    MediaStorageSOPInstanceUID of the file meta information to be equal to the
    SOPClassUID and SOPInstanceUID of the encapsulated dataset.
    """
    import pydicom as dicom
    from pydicom.filereader import read_file_meta_info

    if readFrameOfReferenceUID:
        dicomTags = dicom.dcmread(fileName, specific_tags=["SOPClassUID", "SOPInstanceUID", "FrameOfReferenceUID"], stop_before_pixels=True)
        SOPClassUID = dicomTags.get("SOPClassUID", None)
        SOPInstanceUID = dicomTags.get("SOPInstanceUID", None)
        FrameOfReferenceUID = dicomTags.get("FrameOfReferenceUID", None)
    else:
        fileMeta = read_file_meta_info(fileName)
        SOPClassUID = fileMeta.get("MediaStorageSOPClassUID", None)
        SOPInstanceUID = fileMeta.get("MediaStorageSOPInstanceUID", None)
        FrameOfReferenceUID = None

    if not SOPClassUID:
        raise ValueError(f"Cannot find the SOP Class UID of the dicom file {fileName}.")

    dicomTypeName = SOPClassUID.name

    return {"SOPClassUID": str(SOPClassUID),
            "dicomTypeName": dicomTypeName,
            "dicomType": _getDicomTypeFromTypeName(dicomTypeName),
            "SOPInstanceUID": str(SOPInstanceUID) if SOPInstanceUID else None,
            "FrameOfReferenceUID": str(FrameOfReferenceUID) if FrameOfReferenceUID else None}


def _getDicomsInfoReferences(fileName: str, dicomType: str, infoRow: dict) -> None:
    r"""Read the references of a single RS/RN/RD dicom file and fill them into its info row in place."""
    dicomTags = _getDicomTags(fileName, _DICOM_TYPE_REFERENCE_TAGS[dicomType])

    infoRow["SeriesInstanceUID"] = str(dicomTags.SeriesInstanceUID) if "SeriesInstanceUID" in dicomTags else None
    infoRow["StudyInstanceUID"] = str(dicomTags.StudyInstanceUID) if "StudyInstanceUID" in dicomTags else None

    match dicomType:
        case "RS":
            referencedSOPInstanceUIDs = []
            referencedSeriesNo = 0
            for ReferencedFrameOfReference in dicomTags.get("ReferencedFrameOfReferenceSequence", []):
                for RTReferencedStudy in ReferencedFrameOfReference.get("RTReferencedStudySequence", []):
                    for RTReferencedSeries in RTReferencedStudy.get("RTReferencedSeriesSequence", []):
                        referencedSeriesNo += 1
                        for ContourImage in RTReferencedSeries.get("ContourImageSequence", []):
                            referencedSOPInstanceUIDs.append(str(ContourImage.ReferencedSOPInstanceUID))
            infoRow["referencedSOPInstanceUIDs"] = tuple(referencedSOPInstanceUIDs)
            infoRow["referencedSeriesNo"] = referencedSeriesNo
            # the values are read exactly as the plain tag values, so that a missing tag and
            # a present but empty tag can be told apart by the calling code
            infoRow["structureSetDate"] = dicomTags.get("StructureSetDate", None)
            infoRow["structureSetTime"] = dicomTags.get("StructureSetTime", None)
            # an RS dicom holds no top-level FrameOfReferenceUID, so it is taken from the referenced sequence
            if not infoRow["FrameOfReferenceUID"] and len(dicomTags.get("ReferencedFrameOfReferenceSequence", [])) > 0:
                infoRow["FrameOfReferenceUID"] = str(dicomTags.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID)

        case "RN":
            if len(dicomTags.get("ReferencedStructureSetSequence", [])) > 0:
                infoRow["referencedSOPInstanceUIDs"] = (str(dicomTags.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID),)
            beamSequence = dicomTags.get("IonBeamSequence", None) or dicomTags.get("BeamSequence", None) or []
            infoRow["beamNumbers"] = tuple(int(beamDataset.BeamNumber) for beamDataset in beamSequence if "BeamNumber" in beamDataset)

        case "RD":
            if len(dicomTags.get("ReferencedRTPlanSequence", [])) > 0:
                ReferencedRTPlan = dicomTags.ReferencedRTPlanSequence[0]
                infoRow["referencedSOPInstanceUIDs"] = (str(ReferencedRTPlan.ReferencedSOPInstanceUID),)
                try:
                    infoRow["referencedBeamNumber"] = int(ReferencedRTPlan.ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber)
                except (AttributeError, IndexError):
                    infoRow["referencedBeamNumber"] = None
            infoRow["doseSummationType"] = str(dicomTags.DoseSummationType) if "DoseSummationType" in dicomTags else None


def getDicomsInfo(searchFolder: PathLike | Iterable[PathLike], recursive: bool = True, pattern: str = "*.dcm",
                  readReferences: bool = True, readFrameOfReferenceUID: bool = True, displayInfo: bool = False) -> DataFrame:
    r"""Read the identity and the references of all the dicoms in a folder in a single pass.

    The function reads every dicom of a folder (or of an iterable of folders
    and files) only once and collects the tags which describe the identity of
    the dicom and its references to other dicoms. All the consecutive queries,
    in particular matching the dicoms to each other by their UIDs, can then be
    answered from the resulting index without reading any file again. This is
    the recommended way of working with a large number of dicoms, for which
    calling the checkUID_* functions for every pair of dicoms becomes expensive.

    The information is collected in at most two passes over each dicom. The first pass
    reads the type, the SOPInstanceUID and, if requested, the FrameOfReferenceUID
    of every dicom. The second pass is performed only for the structure set (RS),
    plan (RN) and dose (RD) dicoms, which are the only ones referencing other
    dicoms, and reads their references, beam numbers and related tags. Therefore
    the cost of indexing a folder dominated by image (e.g. CT) dicoms is
    essentially the cost of a single read of each image dicom.

    The function never raises for a dicom which cannot be read. Such a dicom is
    indexed with the 'dicomType' set to 'Unknown' and the description of the
    problem in the 'error' column.

    Parameters
    ----------
    searchFolder : path or iterable of paths
        A path to a folder to be searched for dicoms, a path to a single dicom
        file, or an iterable mixing both.
    recursive : bool, optional
        Determine if the folders should be searched recursively. (def. True)
    pattern : str, optional
        The pattern of the dicom file names to be searched for. (def. '*.dcm')
    readReferences : bool, optional
        Determine if the references of the RS/RN/RD dicoms should be read in
        the second pass. Set to False to only classify the dicoms. (def. True)
    readFrameOfReferenceUID : bool, optional
        Determine if the FrameOfReferenceUID should be read. Setting it to False
        is substantially faster, because only the dicom file meta information is
        read in the first pass, but the 'FrameOfReferenceUID' column is then
        filled only for the RS dicoms read in the second pass. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    DataFrame
        A pandas DataFrame with a single row for each dicom found and the
        columns: 'fileName', 'folderName', 'SOPClassUID', 'dicomTypeName',
        'dicomType', 'SOPInstanceUID', 'FrameOfReferenceUID',
        'referencedSOPInstanceUIDs', 'referencedSeriesNo', 'beamNumbers',
        'referencedBeamNumber', 'doseSummationType', 'structureSetDate',
        'structureSetTime', 'SeriesInstanceUID', 'StudyInstanceUID' and 'error'.
        The rows are given in the order the dicoms were found and are not sorted.

    See Also
    --------
    sortDicomsFromInfo : group the dicoms information by the dicom type.
    matchDicomsByUID : match the plan, structure set, image and dose dicoms by their UIDs.
    sortDicoms : sort dicom files in a folder by type.
    checkDicomsUID : check the UID matching of all the dicoms describing a single patient plan.

    Examples
    --------
    Index a patient folder once and find the structure set referenced by a plan
    without reading any dicom file again.

    >>> dicomsInfo = ft.getDicomsInfo('/path/to/patient')
    >>> RNrow = dicomsInfo[dicomsInfo.dicomType == 'RN'].iloc[0]
    >>> dicomsInfo[dicomsInfo.SOPInstanceUID == RNrow.referencedSOPInstanceUIDs[0]].fileName
    """
    import os
    import pandas as pd

    fileNames = _getDicomsInfoFileNames(searchFolder, recursive=recursive, pattern=pattern)

    infoRows = []
    for fileName in fileNames:
        infoRow: dict[str, Any] = {columnName: None for columnName in _DICOMS_INFO_COLUMNS}
        infoRow["fileName"] = fileName
        infoRow["folderName"] = os.path.dirname(fileName)
        infoRow["referencedSOPInstanceUIDs"] = ()

        try:
            infoRow.update(_getDicomsInfoIdentity(fileName, readFrameOfReferenceUID=readFrameOfReferenceUID))
        except Exception as error:  # a corrupted or non-dicom file must not break the reading of the whole folder
            infoRow["dicomTypeName"] = "Unknown"
            infoRow["dicomType"] = "Unknown"
            infoRow["error"] = repr(error)
            _logger.debug(f"Could not read the identity of the dicom file {fileName}: {error}")
            infoRows.append(infoRow)
            continue

        if readReferences and infoRow["dicomType"] in _DICOM_TYPE_REFERENCE_TAGS:
            try:
                _getDicomsInfoReferences(fileName, infoRow["dicomType"], infoRow)
            except Exception as error:  # the dicom is still indexed, only its references are unknown
                infoRow["error"] = repr(error)
                _logger.debug(f"Could not read the references of the dicom file {fileName}: {error}")

        infoRows.append(infoRow)

    dicomsInfo = pd.DataFrame(infoRows, columns=list(_DICOMS_INFO_COLUMNS))

    # A tag which is not defined for a given dicom type is missing rather than empty, and must be
    # told apart from an empty value. Pandas would infer a string or a float column and convert the
    # missing values to NaN, therefore the columns which can be missing are kept as objects holding
    # None, so that the calling code can always test them with 'is None'.
    for columnName in _DICOMS_INFO_COLUMNS:
        if columnName in ("fileName", "folderName", "referencedSOPInstanceUIDs"):
            continue
        dicomsInfo[columnName] = dicomsInfo[columnName].astype(object).where(dicomsInfo[columnName].notna(), None)

    if len(dicomsInfo) == 0:
        _logger.warning(f"No dicoms found in: {searchFolder}")

    if displayInfo:
        dicomTypeCounts = dicomsInfo.dicomType.value_counts()
        _logger.info(f"Indexed {len(dicomsInfo)} dicoms:" + "".join(f"\n\t{dicomTypeCount:d} x {dicomType}" for dicomType, dicomTypeCount in dicomTypeCounts.items()))

    return dicomsInfo


def sortDicomsFromInfo(dicomsInfo: DataFrame, collapseSingle: bool = True, displayInfo: bool = False) -> DottedDict:
    r"""Group the file names of the dicoms information by the dicom type.

    The function groups the file names of the dicoms information produced by getDicomsInfo
    by the dicom type, producing the same structure as sortDicoms but without
    reading any dicom file. It is used to implement sortDicoms and is useful
    whenever the dicoms information has already been read for other purposes.

    Parameters
    ----------
    dicomsInfo : DataFrame
        The dicoms information produced by getDicomsInfo.
    collapseSingle : bool, optional
        Determine if a group holding exactly one file name should be collapsed
        to that bare file name instead of a single element list. This is the
        behaviour of sortDicoms and is kept for compatibility. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dict
        A dictionary with the keys 'CTfileNames', 'RSfileNames', 'RNfileNames',
        'RDfileNames', 'PETfileNames' and 'Unknown'.

    See Also
    --------
    getDicomsInfo : read the identity and the references of all the dicoms in a folder in a single pass.
    sortDicoms : sort dicom files in a folder by type.
    """
    dicomFiles = {f"{dicomType}fileNames": dicomsInfo.fileName[dicomsInfo.dicomType == dicomType].tolist() for dicomType, _ in _DICOM_TYPE_PATTERNS}
    dicomFiles["Unknown"] = dicomsInfo.fileName[dicomsInfo.dicomType == "Unknown"].tolist()

    fileNamesNo = sum(len(fileNames) for fileNames in dicomFiles.values())

    if displayInfo:
        _logger.info(f"Found {fileNamesNo:d} dicoms:" + "".join(f"\n\t{len(fileNames):d} x {dicomType.replace('fileNames', '')}" for dicomType, fileNames in dicomFiles.items() if len(fileNames) > 0))

    if collapseSingle:
        for dicomType, fileNames in dicomFiles.items():
            if len(fileNames) == 1:
                dicomFiles[dicomType] = fileNames[0]

    return DottedDict(dicomFiles)


def matchDicomsByUID(dicomsInfo: DataFrame, groupImagesBy: Literal["folderName", "SeriesInstanceUID"] = "folderName", displayInfo: bool = False) -> DataFrame:
    r"""Match the plan, structure set, image and dose dicoms by their UIDs.

    The function matches every plan (RN) dicom of the dicoms information to the structure
    set (RS) dicom it references, to the image (e.g. CT) dicoms referenced by
    that structure set, and to all the dose (RD) dicoms referencing the plan.
    The matching is performed entirely on the given information, therefore no dicom file is
    read. The results of the UID checks are equivalent to the ones of
    checkDicomsUID, performed for every plan at once.

    The image dicoms are grouped, by default by the folder they are stored in,
    which is used as a proxy for a single image series export. A group matches
    the structure set only if the SOPInstanceUIDs of all its dicoms are exactly
    the ones referenced by the structure set.

    Parameters
    ----------
    dicomsInfo : DataFrame
        The dicoms information produced by getDicomsInfo. The FrameOfReferenceUID must
        have been read for the 'UIDFoR' check to be performed.
    groupImagesBy : {'folderName', 'SeriesInstanceUID'}, optional
        The column used to group the image dicoms into series. (def. 'folderName')
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    DataFrame
        A pandas DataFrame with a single row for each plan (RN) dicom of the
        index and the columns: 'RNfileName', 'RSfileName', 'CTfileNames',
        'RDfileNames', 'RSfolderName', 'CTfolderName', 'RDfolderNames',
        'UIDRNtoRS', 'UIDRStoCT', 'UIDRNtoRD', 'UIDFoR' and 'RDbeamNumbers'.
        The checks which could not be performed are set to None, i.e. 'UIDRStoCT'
        when no structure set was matched and 'UIDRNtoRD'/'RDbeamNumbers' when no
        dose dicom was matched. Note that the dose dicoms are matched to the plan
        by their UIDs, therefore 'UIDRNtoRD' is never False, unlike in
        checkDicomsUID where the dose dicoms are given explicitly.

    See Also
    --------
    getDicomsInfo : read the identity and the references of all the dicoms in a folder in a single pass.
    checkDicomsUID : check the UID matching of all the dicoms describing a single patient plan.
    """
    import pandas as pd

    rowByUID = {row.SOPInstanceUID: row for row in cast(Iterable[Any], dicomsInfo.itertuples()) if row.SOPInstanceUID}
    imageRows = dicomsInfo[dicomsInfo.dicomType.isin(["CT", "PET"])]
    imageGroups = {groupKey: groupRows for groupKey, groupRows in imageRows.groupby(getattr(imageRows, groupImagesBy), dropna=False)}
    RDrows = list(cast(Iterable[Any], dicomsInfo[dicomsInfo.dicomType == "RD"].itertuples()))

    matchRows = []
    for RNrow in cast(Iterable[Any], dicomsInfo[dicomsInfo.dicomType == "RN"].itertuples()):
        RSrow = rowByUID.get(RNrow.referencedSOPInstanceUIDs[0]) if RNrow.referencedSOPInstanceUIDs else None
        RSrow = RSrow if RSrow is not None and RSrow.dicomType == "RS" else None

        # find the image group holding exactly the images referenced by the structure set
        CTfileNames, CTfolderName, UIDRStoCT = [], None, None
        if RSrow is not None:
            referencedImageUIDs = sorted(RSrow.referencedSOPInstanceUIDs)
            UIDRStoCT = False
            for groupKey, groupRows in imageGroups.items():
                if sorted(groupRows.SOPInstanceUID.tolist()) == referencedImageUIDs and len(referencedImageUIDs) > 0:
                    CTfileNames = groupRows.fileName.tolist()
                    CTfolderName = groupKey
                    UIDRStoCT = True
                    break

        # find all the dose dicoms referencing the plan
        matchedRDrows = [RDrow for RDrow in RDrows if RDrow.referencedSOPInstanceUIDs and RDrow.referencedSOPInstanceUIDs[0] == RNrow.SOPInstanceUID]

        # check that the beam number referenced in every dose dicom is defined in the plan
        RDbeamNumbers = None
        if matchedRDrows:
            RDbeamNumbers = True
            for RDrow in matchedRDrows:
                if RDrow.referencedBeamNumber is None:
                    _logger.warning(f"Cannot find the referenced beam number in the RD dicom {RDrow.fileName} of DoseSummationType '{RDrow.doseSummationType}'. The RD dicom was skipped in the beam number check.")
                    continue
                if RDrow.referencedBeamNumber not in (RNrow.beamNumbers or ()):
                    RDbeamNumbers = False

        # check that all the matched dicoms share the same frame of reference
        matchedRows = [RNrow] + ([RSrow] if RSrow is not None else []) + matchedRDrows
        FoRUIDs = {row.FrameOfReferenceUID for row in matchedRows} | set(imageGroups[CTfolderName].FrameOfReferenceUID.tolist() if CTfolderName is not None else [])
        UIDFoR = None if None in FoRUIDs else len(FoRUIDs) == 1

        matchRows.append({"RNfileName": RNrow.fileName,
                          "RSfileName": RSrow.fileName if RSrow is not None else None,
                          "CTfileNames": CTfileNames,
                          "RDfileNames": [RDrow.fileName for RDrow in matchedRDrows],
                          "RSfolderName": RSrow.folderName if RSrow is not None else None,
                          "CTfolderName": CTfolderName,
                          "RDfolderNames": sorted({RDrow.folderName for RDrow in matchedRDrows}),
                          "UIDRNtoRS": RSrow is not None,
                          "UIDRStoCT": UIDRStoCT,
                          # the matched dose dicoms reference the plan by construction, so the check
                          # is only reported as performed when at least one of them was found
                          "UIDRNtoRD": True if matchedRDrows else None,
                          "UIDFoR": UIDFoR,
                          "RDbeamNumbers": RDbeamNumbers})

    matchResults = pd.DataFrame(matchRows, columns=["RNfileName", "RSfileName", "CTfileNames", "RDfileNames", "RSfolderName", "CTfolderName",
                                                    "RDfolderNames", "UIDRNtoRS", "UIDRStoCT", "UIDRNtoRD", "UIDFoR", "RDbeamNumbers"])

    if displayInfo:
        _logger.info(f"Matched {len(matchResults)} plan dicom{'' if len(matchResults) == 1 else 's'}, "
                     f"of which {int(matchResults.UIDRNtoRS.sum()) if len(matchResults) else 0} reference a structure set present in the given information.")

    return matchResults
