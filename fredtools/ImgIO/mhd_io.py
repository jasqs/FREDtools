from fredtools._logger import loggerDecorator
from SimpleITK import Image as SITKImage
from typing import Iterable, Literal, Union


@loggerDecorator
def writeMHD(img: SITKImage, filePath: str, singleFile: bool = True, overwrite: bool = True, useCompression: bool = False, compressionLevel: int = 5, displayInfo: bool = False) -> None:
    """Write image to MetaImage format.

    The function writes a SimpleITK image object to the MetaImage file. The function
    extends the functionality of SimpleITK.WriteImage() to write a single file
    MetaImage instead of standard two-files MHD+RAW. It is recommended to use \*.mhd
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
    logger = ft._getLogger(__name__)

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    if os.path.exists(filePath) and not overwrite:
        raise ValueError("Warning: {:s} file already exists.".format(filePath))

    # remove empty metadata keys (empty metadata keys raise a warning of SimpleITK)
    for MetaDataKey in img.GetMetaDataKeys():
        if not img.GetMetaData(MetaDataKey):
            img.EraseMetaData(MetaDataKey)

    sitk.WriteImage(img, filePath, useCompression=useCompression, compressionLevel=compressionLevel)

    if singleFile:
        # get the original raw/zraw file name and change ElementDataFile to LOCAL
        with fileinput.FileInput(filePath, inplace=True) as file:
            for line in file:
                rawFileName = re.findall(r"ElementDataFile\W+=\W+(.+)", line)
                print(re.sub("ElementDataFile.+", "ElementDataFile = LOCAL", line), end="")
        rawFileName = os.path.join(os.path.dirname(os.path.abspath(filePath)), rawFileName[0])
        # read binary raw/zraw file and attach to the mhd file
        try:
            with open(rawFileName, mode="rb") as file:
                rawFileContent = file.read()
            with open(filePath, "ab") as fout:
                fout.write(rawFileContent)
            # remove raw/zraw file
            os.remove(rawFileName)
        except:
            logger.error(f"IO error: cannot save voxel map to file: {filePath}")

    logger.debug(f"Writing image to {os.path.abspath(filePath)}")
    logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(img, logger.getEffectiveLevel()))


@loggerDecorator
def readMHD(fileNames: str | Iterable[str], displayInfo: bool = False) -> SITKImage | tuple[SITKImage]:
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
    logger = ft._getLogger(__name__)

    # if fileName is a single string then make it a single element list
    if isinstance(fileNames, str):
        fileNames = [fileNames]

    img = []
    for fileName in fileNames:
        img.append(sitk.ReadImage(fileName, imageIO="MetaImageIO"))

    logger.debug(f"Read {len(img)} {'file' if len(img)==1 else 'files'} from:\n" + "\n".join(fileNames))

    logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(img[0], logger.getEffectiveLevel()))

    return img[0] if len(img) == 1 else tuple(img)


@loggerDecorator
def convertMHDtoSingleFile(fileName: str, displayInfo: bool = False) -> None:
    """Convert two-files MetaImage to two-files.

    The function reads a MetaImage file (two- or single-file) and saves it as
    a single file MetaImage (only \*.mhd) with the same file name.

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
    logger = ft._getLogger(__name__)

    img = ft.readMHD(fileName)
    if isinstance(img, tuple):
        img = img[0]

    ft.writeMHD(img, fileName, singleFile=True, overwrite=True)
    logger.debug(f"Converted file {fileName} to a single file MHD")

    logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(img, logger.getEffectiveLevel()))


@loggerDecorator
def convertMHDtoDoubleFiles(fileName: str, displayInfo: bool = False):
    """Convert single file MetaImage to double- file.

    The function reads a MetaImage file (two- or single-file) and saves it as
    a two-file file MetaImage (\*.mhd+\*.raw) with the same file name.

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
    logger = ft._getLogger(__name__)

    img = ft.readMHD(fileName)
    if isinstance(img, tuple):
        img = img[0]
    ft.writeMHD(img, fileName, singleFile=False, overwrite=True)

    logger.debug(f"Converted file {fileName} to a double files MHD")

    logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(img, logger.getEffectiveLevel()))
