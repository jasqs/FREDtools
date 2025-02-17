from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def readOPG(fileName: PathLike, depth: float = 0, displayInfo: bool = False) -> SITKImage:
    """Read OPG files from OmniPro software.

    The function reads a single OPG file exported from OmniPro software (IBA)
    and creates an instance of a SimpleITK object.

    Parameters
    ----------
    fileName : string
        A path to OPG file.
    depth : scalar, optional
        A scalar defining the depth of a 3D image. Usually, it is the depth of the measurement. (def. 0)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object a SimpleITK image.
    """
    import fredtools as ft
    import SimpleITK as sitk
    import numpy as np
    import re

    data_factor_line = ft.getLineFromFile("^Data Factor:", fileName, kind="first")
    if data_factor_line is None:
        error = ImportError("Could not find the Data Factor line.")
        _logger.error(error)
        raise error
    DataFactor = float(re.findall(rf"^Data Factor:\s+({ft.re_number})", str(data_factor_line[1]))[0])

    data_unit_line = ft.getLineFromFile("^Data Unit:", fileName, kind="first")
    if data_unit_line is None:
        error = ImportError("Could not find the Data Unit line.")
        _logger.error(error)
        raise error
    DataUnit = re.findall(rf"^Data Unit:\s+(.+)", str(data_unit_line[1]))[0]

    length_unit_line = ft.getLineFromFile("^Length Unit:", fileName, kind="first")
    if length_unit_line is None:
        error = ImportError("Could not find the Length Unit line.")
        _logger.error(error)
        raise error
    LengthUnit = re.findall(rf"^Length Unit:\s+(.+)", str(length_unit_line[1]))[0]

    asciibodyStart = ft.getLineFromFile(r"^<asciibody>", fileName, kind="first")
    if asciibodyStart is None:
        error = ImportError("Could not find the start of the asciibody section.")
        _logger.error(error)
        raise error
    else:
        asciibodyStart = int(asciibodyStart[0])

    asciibodyEnd = ft.getLineFromFile(r"^</asciibody>", fileName, kind="first")
    if asciibodyEnd is None:
        error = ImportError("Could not find the end of the asciibody section.")
        _logger.error(error)
        raise error
    else:
        asciibodyEnd = int(asciibodyEnd[0])

    Xcoor_line = ft.getLineFromFile(rf"X\[{LengthUnit}\]", fileName, kind="first", startLine=int(asciibodyStart))
    if Xcoor_line is None:
        error = ImportError("Could not find the X coordinates line.")
        _logger.error(error)
        raise error
    Xcoor = Xcoor_line[1].replace(f"X[{LengthUnit}]", "").replace("\t", "")
    Xcoor = np.fromstring(Xcoor, sep=" ")

    with open(fileName, "r") as file:
        data = file.readlines()
    Ycoor_line = ft.getLineFromFile(rf"Y\[{LengthUnit}\]", fileName, kind="first", startLine=asciibodyStart)
    if Ycoor_line is None:
        error = ImportError("Could not find the Y coordinates line.")
        _logger.error(error)
        raise error
    data = data[Ycoor_line[0]: asciibodyEnd - 1]

    arr = []
    for dataLine in data:
        arr.append(np.fromstring(dataLine.replace("\t", ""), sep=" "))
    arr = np.stack(arr)
    Ycoor = arr[:, 0]
    arr = arr[:, 1:]
    arr = arr * DataFactor  # rescale data

    # rescale dose to [Gy]
    match DataUnit:
        case "Gy":
            arr = arr
        case "mGy":
            arr /= 1e3
        case "cGy":
            arr /= 1e2
        case "uGy":
            arr /= 1e6
        case _:
            error = ImportError(f"Could not recognise data unit '{DataUnit}'.")
            _logger.error(error)
            raise error

    # rescale length to [mm]
    match LengthUnit:
        case "mm":
            Xcoor = Xcoor
            Ycoor = Ycoor
        case "cm":
            Xcoor *= 10
            Ycoor *= 10
        case _:
            error = ImportError(f"Could not recognise length unit '{LengthUnit}'.")
            _logger.error(error)
            raise error

    arr = np.expand_dims(arr, 0)

    img = sitk.GetImageFromArray(arr)
    # img.SetSpacing([np.unique(np.diff(Xcoor).round(2))[0], np.unique(np.diff(Ycoor).round(2))[0], 0.1])
    img.SetSpacing([7.619354838709677, 7.619354838709677, 0.1])  # values set to constant distance of MatriXX PT
    img.SetOrigin([Xcoor[0], Ycoor[0], depth])

    if displayInfo:
        strLog = [f"Read OPG file: {fileName}",
                  f"Original Data Unit:   {DataUnit}",
                  f"Original Data Factor: {DataFactor}",
                  f"Original Length Unit: {LengthUnit}"]
        _logger.info("\n\t".join(strLog) + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(img))

    return img


def readOPD(fileName: PathLike, depth: Numberic = 0, returnImg=["Integral", "Sum"], raiseWarning: bool = True, displayInfo: bool = False) -> List[SITKImage]:
    """Read OPD files from OmniPro software.

    The function reads a single OPG file saved by OmniPro software (IBA)
    and creates an instance of a SimpleITK object. Only the files saved in
    video mode are handled now and the last saved integral is read.

    Parameters
    ----------
    fileName : string
        A path to OPD file.
    depth : scalar, optional
        A scalar defining the depth of a 3D image. Usually, it is the depth of the measurement. (def. 0)
    returnImg : string or iterable of strings
        A strung or an iterable of strings determining the type of image to be returned.
        Usually it might take "Snap", "Integral" and/or "Sum". (def. ["Integral", "Sum"])
    raiseWarning : bool, optional
        Raise warnings. (def. False)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object a SimpleITK image.

    Notes
    -----
    The implementation in python has been done based on the MATLAB
    implementation prepared by Dawid Krzempek.
    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk
    import re

    # open and read binary file
    with open(fileName, "rb") as f:
        data = f.read()

    imgsStartIdx = [m.start() for m in re.finditer(b"Integral|Snap|Sum", data)]

    imgs = []
    imgsType = []
    for imgStartIdx in imgsStartIdx:
        imgType = np.frombuffer(data, dtype="S4", count=1, offset=imgStartIdx)[0].decode()
        match imgType:
            case "Snap":
                imgType = np.frombuffer(data, dtype="S4", count=1, offset=imgStartIdx)[0].decode()
                imgNo = int(np.frombuffer(data, dtype="S4", count=1, offset=imgStartIdx + 5)[0])
            case "Inte":
                imgType = np.frombuffer(data, dtype="S8", count=1, offset=imgStartIdx)[0].decode()
                imgNo = int(np.frombuffer(data, dtype="S4", count=1, offset=imgStartIdx + 9)[0])
            case "Sum ":
                imgType = np.frombuffer(data, dtype="S3", count=1, offset=imgStartIdx)[0].decode()
                imgNo = int(np.frombuffer(data, dtype="S4", count=1, offset=imgStartIdx + 4)[0])

        # import image size
        imgSize = np.frombuffer(data, dtype=np.uint16, count=2, offset=imgStartIdx + 2603)
        # import image coordinates
        imgCornerVoxelsCentres = np.array(
            [
                np.frombuffer(data, dtype=np.float64, count=2, offset=imgStartIdx + 2617),
                np.frombuffer(data, dtype=np.float64, count=2, offset=imgStartIdx + 2635),
            ]
        )

        # calculate pixel size
        imgPixelSize = np.diff(imgCornerVoxelsCentres).squeeze() / (imgSize - 1)

        # import rescale factor
        imgRescaleFactor = np.frombuffer(data, dtype=np.float64, count=1, offset=imgStartIdx + 61)[0]

        # import relative dose distribution
        arr = np.frombuffer(data, dtype=np.uint32, count=1024, offset=imgStartIdx + 2673)
        arr = np.reshape(arr, imgSize)
        arr = arr.astype(float)

        # generate SimpleITK instance
        arr = arr * (imgRescaleFactor / 1000)  # rescale image
        arr = np.flipud(arr)
        arr = np.expand_dims(arr, 0)
        img = sitk.GetImageFromArray(arr)
        img.SetOrigin(imgCornerVoxelsCentres[:, 0].tolist() + [depth])
        img.SetSpacing(imgPixelSize.tolist() + [0.1])
        img.SetMetaData("OPDimageType", imgType)
        img.SetMetaData("OPDimageNo", str(imgNo))

        imgs.append(img)
        imgsType.append(imgType)

    # check integral and/or sum
    if ("Integral" in imgsType) or ("Sum" in imgsType):
        imgSum = []
        for img in imgs:
            if img.GetMetaData("OPDimageType") == "Snap":
                imgSum.append(sitk.GetArrayViewFromImage(img))
            elif img.GetMetaData("OPDimageType") in ["Integral", "Sum"]:
                imgSum = np.sum(np.array(imgSum), axis=0)
                if not np.all(np.isclose(imgSum, sitk.GetArrayViewFromImage(img), rtol=0, atol=1E-4)) and raiseWarning:
                    _logger.warning(f"Checking the Integral/Sum failed. The sum of Snap images preceding the Integral/Sum image no {img.GetMetaData('OPDimageNo')} is not equal to the Integral/Sum image.")
                imgSum = []
            else:
                imgSum = []

    if isinstance(returnImg, str):
        returnImg = [returnImg]
    returnImg = [x.lower() for x in returnImg]

    imgs = list(filter(lambda x: x.GetMetaData("OPDimageType").lower() in returnImg, imgs))

    if displayInfo:
        strLog = [f"Found {len(imgsType)} images of types: ",
                  f"\t Snap: {np.sum([x == 'Snap' for x in imgsType])}",
                  f"\t Integral: {np.sum([x == 'Integral' for x in imgsType])}",
                  f"\t Sum: {np.sum([x == 'Sum' for x in imgsType])}",
                  f"\t Other: {np.sum([x not in ['Snap', 'Integral', 'Sum'] for x in imgsType])}",
                  f"Returned image types: {', '.join(returnImg).title()}",
                  f"Information about the last image:"]
        _logger.info("\n\t".join(strLog) + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(imgs[-1]))

    return imgs
