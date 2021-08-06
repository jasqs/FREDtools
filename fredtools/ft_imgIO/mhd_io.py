def writeMHD(img, filePath, singleFile=True, overwrite=True, displayInfo=False):
    """Write image to MetaImage format.

    The function writes a SimpleITK image object to MetaImage file. The function
    extends the functionality of SimpleITK.WriteImage() to write also a single file
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
        Overwrite the file if it exists. Otherwise raise an error. (def. True)
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

    ft._isSITK(img, raiseError=True)

    if os.path.exists(filePath) and not overwrite:
        raise ValueError("Warning: {:s} file already exists.".format(filePath))

    sitk.WriteImage(img, filePath)

    if singleFile:
        # get the original raw file name and change ElementDataFile to LOCAL
        with fileinput.FileInput(filePath, inplace=True) as file:
            for line in file:
                rawFileName = re.findall(r"ElementDataFile\W+=\W+(.+)", line)
                print(re.sub("ElementDataFile.+", "ElementDataFile = LOCAL", line), end="")
        rawFileName = os.path.join(os.path.dirname(os.path.abspath(filePath)), rawFileName[0])
        # save binary data to mhd file
        try:
            fout = open(filePath, "ab")
            voxels = sitk.GetArrayFromImage(img)
            fout.write(voxels.tobytes())
            fout.close()
            # remove raw file
            os.remove(rawFileName)
        except:
            print("IO error: cannot save voxel map to file:", voxels)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))


def readMHD(filePath, displayInfo=False):
    """Read MetaImage image to SimpleITK image object.

    The function reads a MetaImage file to a SimpleITK image object.

    Parameters
    ----------
    filePath : path
        Path to MetaImage file to read.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image.

    See Also
    --------
    SimpleITK.ReadImage : SimpleITK routine for reading files.
    """
    import fredtools as ft
    import SimpleITK as sitk

    img = sitk.ReadImage(filePath, imageIO="MetaImageIO")
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return img


def convertMHDtoSingleFile(filePath, displayInfo=False):
    """Convert two-files MetaImage to two-files.

    The function reads a MetaImage file (two- or single-file) and saves it as
    a single file MetaImage (only \*.mhd) with the same file name.

    Parameters
    ----------
    filePath : path
        Path to file to be converted.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    See Also
    --------
    readMHD : reading MetaImage file.
    writeMHD : writing MetaImage file.
    """
    import fredtools as ft

    img = ft.readMHD(filePath)
    ft.writeMHD(img, filePath, singleFile=True, overwrite=True)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))


def convertMHDtoDoubleFiles(filePath, displayInfo=False):
    """Convert single file MetaImage to double- file.

    The function reads a MetaImage file (two- or single-file) and saves it as
    a two-file file MetaImage (\*.mhd+\*.raw) with the same file name.

    Parameters
    ----------
    filePath : path
        Path to file to be converted.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    See Also
    --------
    readMHD : reading MetaImage file.
    writeMHD : writing MetaImage file.
    """
    import fredtools as ft

    img = ft.readMHD(filePath)
    ft.writeMHD(img, filePath, singleFile=False, overwrite=True)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
