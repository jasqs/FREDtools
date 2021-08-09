def readDijFRED(dijFileName, raiseMemError=True, displayInfo=False):
    """Read FRED Dij image to SimpleITK vector image object.

    The function reads a Dij (influence matrix) file produced
    by the FRED Monte Carlo to a SimpleITK vector image object.
    Each element of the vectors in voxels is a single pencil beam
    signal.

    Parameters
    ----------
    filePath : path
        Path to FRED Dij file to read.
    raiseMemError : bool, optional
        Raise an error if the expected size of the imported array is larger than the available RAM space. (def. True)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Vector Image
        Object of a SimpleITK vector image.
    """
    import struct
    import SimpleITK as sitk
    import numpy as np
    import fredtools as ft
    import psutil

    with open(dijFileName, "rb") as dijFile_h:
        [sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, pencilBeamNo] = struct.unpack("<3i6f1i", dijFile_h.read(40))
        shape = np.array([sizeX, sizeY, sizeZ])
        size = np.prod(shape)
        spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=7) * 10
        offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=7) * 10
        origin = offset + spacing / 2

        if raiseMemError:
            if psutil.virtual_memory().available < (size * 8 * pencilBeamNo):
                raise MemoryError(f"The Dij matrix is expected to use {(size*4*pencilBeamNo/1024**3):.2f} GB of RAM but only {psutil.virtual_memory().available/1024**3:.2f} GB is available.")

        imgs = []
        fieldIDs = []
        for bencilBeam_idx in range(pencilBeamNo):
            img = np.zeros(size, dtype="float32")
            [pencilBeamID, fieldID, voxelsNo] = struct.unpack("3i", dijFile_h.read(12))
            fieldIDs.append(fieldID)
            voxelIndices = np.frombuffer(dijFile_h.read(voxelsNo * 4), dtype="uint32", count=voxelsNo)
            vocelValues = np.frombuffer(dijFile_h.read(voxelsNo * 4), dtype="float32", count=voxelsNo)
            img[voxelIndices] += vocelValues
            img = np.reshape(img, shape, order="F")
            imgs.append(img)

    arr = np.stack(imgs)
    arr = np.moveaxis(arr, list(range(arr.ndim)), list(range(arr.ndim))[::-1])
    img = sitk.GetImageFromArray(arr, isVector=True)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Pencil beams: {pencilBeamNo}")
        print(f"# Fields: {np.unique(fieldIDs).size}")
        ft.ft_imgAnalyse._displayImageInfo(img)
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return img
