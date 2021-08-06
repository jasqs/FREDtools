def writeMap3D(img, filePath, displayInfo=False):
    """Write image to Map3D file format.

    The function writes a SimpleITK image object to Map3D file. This format is
    an obsolete format used by the FRED Monte Carlo engine and the implementation
    of writing images in this format was maintained here for compatibility. It is
    recommended to use MetaImage file format (\*.mhd) for saving the images. It is
    recommended to use \*.m3d extension when saving Map3D.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.
    filePath : path
        Path to file to be saved.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    See Also
    --------
    writeMHD : Writing MetaImage files.
    """
    import numpy as np
    import struct
    import fredtools as ft

    ft._isSITK3D(img, raiseError=True)

    try:
        fout = open(filePath, "wb")
    except:
        raise ValueError(f"IO error: cannot open output file {filePath}")
    ncomp = 1
    endian = "L"
    vers = 1
    format = 1

    imgSize = img.GetSize()
    imgSpacing = img.GetSpacing()
    imgOrigin = img.GetOrigin()
    map3dheader = struct.pack(
        "<4s4i4s6fcbh2i4s",
        "MP3D".encode("utf-8"),
        imgSize[0],
        imgSize[1],
        imgSize[2],
        ncomp,
        _map3d_dtype2datatypeString(ft.arr(img).dtype).encode("utf-8"),
        imgSpacing[0] / 10,
        imgSpacing[1] / 10,
        imgSpacing[2] / 10,
        (imgOrigin[0] - imgSpacing[0] / 2) / 10,
        (imgOrigin[1] - imgSpacing[1] / 2) / 10,
        (imgOrigin[2] - imgSpacing[2] / 2) / 10,
        endian.encode("utf-8"),
        vers,
        format,
        0,
        0,
        "MP3D".encode("utf-8"),
    )

    fout.write(map3dheader)
    fout.write(ft.arr(img).tobytes(order="F"))  # write out as column-major
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))


def readMap3D(filePath, displayInfo=False):
    """Read image from Map3D file format.

    The function reads a Map3D file to a SimpleITK image object. The Map3D file
    format is an obsolete format used by the FRED Monte Carlo engine and the implementation
    of reading images from files in this format was maintained here for compatibility.

    Parameters
    ----------
    filePath : path
        Path to Map3D file to read.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image.

    See Also
    --------
    writeMHD : Writing MetaImage files.
    readMHD : Reading MetaImage files.
    """
    import numpy as np
    import struct
    import fredtools as ft
    import SimpleITK as sitk

    try:
        fin = open(filePath, "rb")
    except:
        raise ValueError(f"IO error: cannot open output file {filePath}")

    buffer = fin.read()
    map3dheader = struct.unpack("<4s4i4s6fcbh2i4s", buffer[0:64]) or die
    [magicbeg, nx, ny, nz, ncomp, datatype, hx, hy, hz, x0, y0, z0, endian, vers, format, usertag, reserved, magicend] = map3dheader
    if magicbeg.decode("utf-8") != "MP3D" or magicend.decode("utf-8") != "MP3D":
        print("file", filePath, "does not contain a valid map3d")
        return None
    if endian.decode("utf-8") != "L":
        print("endianess must be Little-endian L, instead found", endian)
        return None
    if vers not in [1]:
        print("map3d version not supported:", vers)
        return None
    if format not in [1, 10]:
        print("internal format not supported:", format)
        return None
    nn = np.array([nx, ny, nz])
    hs = np.around(np.array([hx, hy, hz]), decimals=8)
    xoff = np.around(np.array([x0, y0, z0]), decimals=8)
    N = np.prod(nn) * ncomp
    datatype = datatype.decode("utf-8")
    if format == 1:
        M = np.frombuffer(buffer[64:], dtype=_map3d_datatypeString2dtype(datatype), count=N)
    elif format == 10:
        M = np.zeros(N, dtype=_map3d_datatypeString2dtype(datatype))
        pos = 64
        [nvxl] = struct.unpack("i", buffer[pos : pos + 4])
        pos = pos + 4
        if nvxl > 0:
            Ivxl = np.frombuffer(buffer[pos:], dtype="uint32", count=nvxl)
            pos = pos + nvxl * Ivxl.itemsize
        refVal = np.frombuffer(buffer[pos:], dtype=_map3d_datatypeString2dtype(datatype), count=1)[0]
        pos = pos + M.itemsize
        M += refVal
        if nvxl > 0:
            Vals = np.frombuffer(buffer[pos:], dtype=_map3d_datatypeString2dtype(datatype), count=nvxl)
            pos = pos + nvxl * M.itemsize
            M[Ivxl] = Vals
    if M.size != N:
        print("Error: read ", M.size, "instead of", N)
        return [nn, hs, xoff, None]
    M = np.reshape(M, nn, order="F")
    img = sitk.GetImageFromArray(M.T)
    img.SetSpacing(hs * 10)
    img.SetOrigin((xoff + hs / 2) * 10)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return img


def _map3d_datatypeString2dtype(datatype):
    datatypeString = ["si8 ", "si16", "si32", "si64", "ui8 ", "ui16", "ui32", "ui64", "re32", "re64"]
    numpyDtypes = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"]
    for i, s in enumerate(datatypeString):
        if datatype == s:
            return numpyDtypes[i]
    raise TypeError(f"Error: datatype {datatype} unknown or not yet implemented")


def _map3d_dtype2datatypeString(dtype):
    datatypeString = ["si8 ", "si16", "si32", "si64", "ui8 ", "ui16", "ui32", "ui64", "re32", "re64"]
    numpyDtypes = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"]
    for i, nptype in enumerate(numpyDtypes):
        if dtype == nptype:
            return datatypeString[i]
    raise TypeError(f"Error: datatype {dtype} unknown or not yet implemented")
