from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def writeMHD(img: SITKImage, filePath: PathLike, singleFile: bool = True, overwrite: bool = True, useCompression: bool = False, compressionLevel: int = 5, displayInfo: bool = False) -> None:
    """Write image to MetaImage format.

    The function writes a SimpleITK image object to the MetaImage file. The function
    extends the functionality of SimpleITK.WriteImage() to write a single file
    MetaImage instead of standard two-files MHD+RAW. It is recommended to use .mhd
    extension when saving MetaImage.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    filePath : path
        Path to file to be saved.
    singleFile : bool, optional
        Determine if the MHD is a single file of two files MHD+RAW. (def. True)
    overwrite : bool, optional
        Overwrite the file if it exists, otherwise, raise an error. (def. True)
    useCompression : bool, optional
        Determine if a compression will be used when saving the file. (def. False)
    compressionLevel : unsigned int, optional
        Determine the compression level. For MHD files, the compression level
        above 10 does not have any effect. (def. 5)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    See Also
    --------
    SimpleITK.WriteImage : SimpleITK routine for writing files.
    """
    import os
    import re
    import fileinput
    import fredtools as ft
    import SimpleITK as sitk

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    if os.path.exists(filePath) and not overwrite:
        raise ValueError(f"Warning: {filePath} file already exists.")

    # remove empty metadata keys (empty metadata keys raise a warning of SimpleITK)
    for MetaDataKey in img.GetMetaDataKeys():
        if not img.GetMetaData(MetaDataKey):
            img.EraseMetaData(MetaDataKey)

    _logger.debug(f"Writing image to {os.path.abspath(filePath)}.")
    sitk.WriteImage(img, str(filePath), useCompression=useCompression, compressionLevel=compressionLevel)

    if singleFile:
        _logger.debug(f"Converting {filePath} to a single file MHD.")
        # get the original raw/zraw file name and change ElementDataFile to LOCAL
        rawFileName = None
        with fileinput.FileInput(filePath, inplace=True) as file:
            for line in file:
                rawFileName = re.findall(r"ElementDataFile\W+=\W+(.+)", line)
                print(re.sub(r"ElementDataFile.+", "ElementDataFile = LOCAL", line), end="")
        # read binary raw/zraw file and attach to the mhd file
        try:
            if not rawFileName:
                raise ValueError(f"Could not find `ElementDataFile` tag in {filePath}.")
            rawFileName = os.path.join(os.path.dirname(os.path.abspath(filePath)), rawFileName[0])
            with open(rawFileName, mode="rb") as file:
                rawFileContent = file.read()
            with open(filePath, "ab") as fout:
                fout.write(rawFileContent)
            # remove raw/zraw file
            os.remove(rawFileName)
        except IOError as error:
            _logger.error(error)
            raise error
        except ValueError as error:
            _logger.error(error)
            raise error

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(img))


@overload
def readMHD(fileNames: PathLike, displayInfo: bool = False) -> SITKImage: ...


@overload
def readMHD(fileNames: Sequence[PathLike], displayInfo: bool = False) -> tuple[SITKImage, ...]: ...


def readMHD(fileNames: Sequence[PathLike] | PathLike, displayInfo: bool = False) -> SITKImage | tuple[SITKImage, ...]:
    """Read MetaImage image to SimpleITK image object.

    The function reads a single MetaImage file or an iterable of MetaImage files
    and creates an instance or tuple of instances of a SimpleITK object.

    Parameters
    ----------
    fileNames : string or array_like
        A path or an iterable (list, tuple, etc.) of paths to MetaImage file.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image or tuple
        Object or tuple of objects of a SimpleITK image.


    See Also
    --------
    SimpleITK.ReadImage : SimpleITK routine for reading files.
    """
    import fredtools as ft
    import SimpleITK as sitk

    # if fileName is a single string then make it a single element list
    if isinstance(fileNames, PathLike):
        fileNames = [fileNames]

    img = []
    for fileName in fileNames:
        img.append(sitk.ReadImage(str(fileName), imageIO="MetaImageIO"))

    _logger.debug(f"Read {len(img)} {'file' if len(img)==1 else 'files'} from:\n\t" + "\n\t".join(map(str, fileNames)))

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(img[0]))

    return img[0] if len(img) == 1 else tuple(img)


def convertMHDtoSingleFile(fileName: PathLike, displayInfo: bool = False) -> None:
    """Convert two-files MetaImage to two-files.

    The function reads a MetaImage file (two- or single-file) and saves it as
    a single file MetaImage (MHD) with the same file name.

    Parameters
    ----------
    fileName : path
        Path to file to be converted.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    See Also
    --------
    readMHD : reading MetaImage file.
    writeMHD : writing MetaImage file.
    """
    import fredtools as ft
    _logger = ft.getLogger()

    img = ft.readMHD(fileName)

    ft.writeMHD(img, fileName, singleFile=True, overwrite=True)
    _logger.debug(f"Converted file {fileName} to a single file MHD")

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(img))


def convertMHDtoDoubleFiles(fileName: PathLike, displayInfo: bool = False) -> None:
    """Convert single file MetaImage to double- file.

    The function reads a MetaImage file (two- or single-file) and saves it as
    a two-file file MetaImage (mhd/raw) with the same file name.

    Parameters
    ----------
    fileName : path
        Path to file to be converted.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    See Also
    --------
    readMHD : reading MetaImage file.
    writeMHD : writing MetaImage file.
    """
    import fredtools as ft
    _logger = ft.getLogger()

    img = ft.readMHD(fileName)

    ft.writeMHD(img, fileName, singleFile=False, overwrite=True)

    _logger.debug(f"Converted file {fileName} to a double files MHD")
    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(img))
