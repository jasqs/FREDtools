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
