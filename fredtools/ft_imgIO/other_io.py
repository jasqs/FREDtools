def readOPG(fileName, depth=0, displayInfo=True):
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

    DataFactor = float(re.findall(rf"^Data Factor:\s+({ft.re_number})", ft.getLineFromFile("^Data Factor:", fileName, kind="first")[1])[0])
    DataUnit = re.findall(rf"^Data Unit:\s+(.+)", ft.getLineFromFile("^Data Unit:", fileName, kind="first")[1])[0]
    LengthUnit = re.findall(rf"^Length Unit:\s+(.+)", ft.getLineFromFile("^Length Unit:", fileName, kind="first")[1])[0]

    asciibodyStart = ft.getLineFromFile(r"^<asciibody>", fileName, kind="first")[0]
    asciibodyEnd = ft.getLineFromFile(r"^</asciibody>", fileName, kind="first")[0]
    Xcoor = ft.getLineFromFile(rf"X\[{LengthUnit}\]", fileName, kind="first", startLine=asciibodyStart)[1].replace(f"X[{LengthUnit}]", "").replace("\t", "")
    Xcoor = np.fromstring(Xcoor, sep=" ")

    with open(fileName, "r") as file:
        data = file.readlines()
    data = data[ft.getLineFromFile(rf"Y\[{LengthUnit}\]", fileName, kind="first", startLine=asciibodyStart)[0] : asciibodyEnd - 1]
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
            raise AttributeError(f"Could not recognise data unit '{DataUnit}'.")

    # rescale length to [mm]
    match LengthUnit:
        case "mm":
            Xcoor = Xcoor
            Ycoor = Ycoor
        case "cm":
            Xcoor *= 10
            Ycoor *= 10
        case _:
            raise AttributeError(f"Could not recognise length unit '{LengthUnit}'.")

    arr = np.expand_dims(arr, 0)

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([np.unique(np.diff(Xcoor).round(2))[0], np.unique(np.diff(Ycoor).round(2))[0], 0.1])
    img.SetOrigin([Xcoor[0], Ycoor[0], depth])

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Original Data Unit:   {DataUnit}")
        print(f"# Original Data Factor: {DataFactor}")
        print(f"# Original Length Unit: {LengthUnit}")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return img
