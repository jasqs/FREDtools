from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def getInmFREDVersion(fileName: PathLike) -> float:
    """Get the version of the FRED influence matrix file.

    The function reads the version of the influence matrix file produced by the FRED Monte Carlo.

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.

    Returns
    -------
    int
        Version of the FRED influence matrix file.
    """
    import struct

    with open(fileName, "rb") as file_h:
        InmFREDVersion, = struct.unpack("i", file_h.read(4))

    return InmFREDVersion/10


def _isInmFRED(fileName: PathLike, raiseError: bool = False) -> bool:
    """Check if the file is a proper FRED influence matrix file and raise error if requested."""
    import struct

    try:
        with open(fileName, "rb") as file_h:
            InmFREDVersion, = struct.unpack("i", file_h.read(4))
            InmFREDVersion = InmFREDVersion/10
        if InmFREDVersion == 0 or InmFREDVersion > 10:
            if raiseError:
                raise TypeError(f"The file is not a proper FRED influence matrix file.")
            else:
                return False
        else:
            return True
    except TypeError as error:
        if raiseError:
            _logger.error(f"Error reading the version of the Inm file '{fileName}': {error}")
            raise error
        else:
            return False
    except FileNotFoundError as error:
        if raiseError:
            _logger.error(f"Error reading the version of the Inm file '{fileName}': {error}")
            raise error
        else:
            return False


def getInmFREDInfo(fileName: PathLike, displayInfo: bool = False) -> DataFrame:
    """Read basic information from FRED influence matrix.

    The function reads an influence matrix file produced by the FRED Monte Carlo
    and gets the basic information about the pencil beams and fields saved.

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    DataFrame
        Pandas DataFrame with pencil beams and field numbers.

    See Also
    --------
        getInmFREDBaseImg : get base image defined in FRED influence matrix.
        getInmFREDSparse : get sparse matrices of point values from an influence matrix produced by FRED Monte Carlo.
    """
    import fredtools as ft
    import numpy as np
    import struct

    _isInmFRED(fileName, raiseError=True)

    # create an empty basic image with FoR defined in Inm
    imgBase = ft.getInmFREDBaseImg(fileName, dtype="uint8")

    # read the influence matrix file depending on the version
    InmFREDVersion = getInmFREDVersion(fileName)
    _logger.debug(f"Version of Inm file {fileName}: {InmFREDVersion}.")

    match InmFREDVersion:
        case 2.0 | 2.1:
            inmInfo = _getInmFREDInfoVersion2(fileName)
        case 3.0 | 3.1:
            inmInfo = _getInmFREDInfoVersion3(fileName)
        case _:
            error = NotImplementedError(f"Version {InmFREDVersion} of the Inm file is not supported.")
            _logger.error(error)
            raise error

    if displayInfo:
        with open(fileName, "rb") as file_h:
            match InmFREDVersion:
                case 2.0 | 3.0:
                    [_, _, _, _, _, _, _, _, _, _, componentNo, _] = struct.unpack("<4i6f2i", file_h.read(48))
                case 2.1 | 3.1:
                    [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, componentNo, _] = struct.unpack("<4i6f9f2i", file_h.read(84))
                case _:
                    error = NotImplementedError(f"Version {InmFREDVersion} of the Inm file is not supported.")
                    _logger.error(error)
                    raise error

        strLog = [f"Imn file version: {InmFREDVersion}",
                  f"Number of PBs: {inmInfo.PBID.size}",
                  f"Number of fields: {inmInfo.FID.unique().size}",
                  f"Number of components: {componentNo}",
                  f"Number of voxels (min/max/mean): {inmInfo.voxelsNo.min()}/{inmInfo.voxelsNo.max()}/{inmInfo.voxelsNo.mean():.0f}",
                  f"Percent of voxels (min/max/mean): {inmInfo.voxelsNo.min()/np.prod(imgBase.GetSize())*100:.2f}%/{inmInfo.voxelsNo.max()/np.prod(imgBase.GetSize())*100:.2f}%/{inmInfo.voxelsNo.mean()/np.prod(imgBase.GetSize())*100:.2f}%",
                  "FoR of the image:"]
        _logger.info("\n\t".join(strLog) + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(imgBase))

    return inmInfo


def _getInmFREDInfoVersion2(fileName: PathLike) -> DataFrame:
    """Get information from the FRED influence matrix file version 2"""
    import pandas as pd
    import struct

    InmFREDVersion = getInmFREDVersion(fileName)

    with open(fileName, "rb") as file_h:
        match InmFREDVersion:
            case 2.0:
                headerSize = 48
                [_, _, _, _, _, _, _, _, _, _, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(headerSize))
            case 2.1:
                headerSize = 84
                [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, componentNo, pencilBeamNo] = struct.unpack("<4i6f9f2i", file_h.read(headerSize))
            case _:
                error = NotImplementedError(f"Version {InmFREDVersion} of the Inm file is not supported.")
                _logger.error(error)
                raise error

    with open(fileName, "rb") as file_h:
        file_h.seek(headerSize, 1)  # skip header

        FIDs = []
        PBIDs = []
        # fileTargets =[]
        voxelsNos = []
        try:
            for _ in range(pencilBeamNo):
                [PBTag, voxelsNo] = struct.unpack("2i", file_h.read(8))

                # fileTargets.append(file_h.tell())
                FIDs.append(int(PBTag / 1000000))
                PBIDs.append(PBTag % 1000000)
                voxelsNos.append(voxelsNo)

                file_h.seek(voxelsNo * 4 * (componentNo+1), 1)  # jump to the next PB
        except:
            error = TypeError('Could not parse the whole structure of the influence matrix.')
            _logger.error(error)
            raise error

    inmInfo = pd.DataFrame({"FID": FIDs, "PBID": PBIDs, "voxelsNo": voxelsNos})
    inmInfo = inmInfo.astype({"FID": "uint32", "PBID": "uint32", "voxelsNo": "uint32"})
    inmInfo.index.set_names('PBIdx', inplace=True)

    return inmInfo


def _getInmFREDInfoVersion3(fileName: PathLike) -> DataFrame:
    """Get information from the FRED influence matrix file version 3"""
    import pandas as pd
    import struct
    import numpy as np

    InmFREDVersion = getInmFREDVersion(fileName)

    with open(fileName, "rb") as file_h:
        match InmFREDVersion:
            case 3.0:
                headerSize = 48
                [_, _, _, _, _, _, _, _, _, _, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(headerSize))
            case 3.1:
                headerSize = 84
                [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, componentNo, pencilBeamNo] = struct.unpack("<4i6f9f2i", file_h.read(headerSize))
            case _:
                error = NotImplementedError(f"Version {InmFREDVersion} of the Inm file is not supported.")
                _logger.error(error)
                raise error

    with open(fileName, "rb") as file_h:
        file_h.seek(headerSize, 1)  # skip header

        # get PBIdx to (FID, PBID) mapping array
        mapPBIdx = np.frombuffer(file_h.read(3 * pencilBeamNo * 4), dtype="uint32", count=pencilBeamNo*3)
        mapPBIdx = np.reshape(mapPBIdx, (pencilBeamNo, 3))

        # get components' size
        componentsDataSize = np.frombuffer(file_h.read(4 * componentNo), dtype="uint32", count=componentNo)

        pbIdx = np.frombuffer(file_h.read(componentsDataSize[0]*4), dtype="uint32", count=componentsDataSize[0])

    inmInfo = pd.DataFrame(mapPBIdx, columns=["PBIdx", "FID", "PBID"]).set_index("PBIdx")

    pbIdx, voxelsNos = np.unique(pbIdx, return_counts=True)
    voxelsNo = pd.DataFrame({"PBIdx": pbIdx, "voxelsNo": voxelsNos.astype("uint32")}).set_index("PBIdx")
    inmInfo = pd.merge(inmInfo, voxelsNo, left_index=True, right_index=True, how="left")
    inmInfo.fillna(0, inplace=True)
    inmInfo = inmInfo.astype({"FID": "uint32", "PBID": "uint32", "voxelsNo": "uint32"})

    return inmInfo


def getInmFREDBaseImg(fileName: PathLike, dtype: DTypeLike = float, displayInfo: bool = False) -> SITKImage:
    """Get base image defined in FRED influence matrix.

    The function reads the header of an influence matrix file produced by the FRED Monte Carlo
    and builds the basic image of a given type, defined as an instance of a SimpleITK image object,
    with the frame of reference defined in the influence matrix.

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.
    dtype : data-type, optional
        The desired data-type for the output image, e.g., `numpy.uint32` or `float32`. (def. numpy.float64)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK Image
        An object of a SimpleITK image.

    See Also
    --------
        getInmFREDInfo : get information from an influence matrix produced by FRED Monte Carlo.
        getInmFREDSparse : get sparse matrices of point values from an influence matrix produced by FRED Monte Carlo.
    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk
    import struct

    _isInmFRED(fileName, raiseError=True)

    InmFREDVersion = getInmFREDVersion(fileName)
    _logger.debug(f"Version of Inm file {fileName}: {InmFREDVersion}.")

    with open(fileName, "rb") as file_h:
        # get FoR of the Inm image and pencil beam number
        match InmFREDVersion:
            case 2.0 | 3.0:
                [InmFREDVersion, sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, offsetX, offsetY, offsetZ, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(48))
                shape = np.array([sizeX, sizeY, sizeZ])
                spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=4) * 10
                offset = np.around(np.array([offsetX, offsetY, offsetZ]), decimals=4) * 10
                origin = offset + spacing / 2

                # create empty basic image with FoR defined in Inm
                imgBase = sitk.GetImageFromArray(np.zeros(shape[::-1], dtype=dtype))
                imgBase.SetOrigin(origin)
                imgBase.SetSpacing(spacing)
            case 2.1 | 3.1:
                transformMatrix = np.zeros((3, 3)).flatten()
                [InmFREDVersion, sizeX, sizeY, sizeZ, spacingX, spacingY, spacingZ, originX, originY, originZ, *transformMatrix, componentNo, pencilBeamNo] = struct.unpack("<4i6f9f2i", file_h.read(84))
                shape = np.array([sizeX, sizeY, sizeZ])
                spacing = np.around(np.array([spacingX, spacingY, spacingZ]), decimals=4) * 10
                origin = np.around(np.array([originX, originY, originZ]), decimals=4) * 10

                # create empty basic image with FoR defined in Inm
                imgBase = sitk.GetImageFromArray(np.zeros(shape[::-1], dtype=dtype))
                imgBase.SetOrigin(origin)
                imgBase.SetSpacing(spacing)
                directionMatrix = np.reshape(transformMatrix, [imgBase.GetDimension()]*2).T.dot(np.eye(imgBase.GetDimension()))
                imgBase.SetDirection(directionMatrix.flatten())

            case _:
                error = NotImplementedError(f"Version {InmFREDVersion} of the Inm file is not supported.")
                _logger.error(error)
                raise error

    if displayInfo:
        strLog = [f"Inm file version: {InmFREDVersion}",
                  f"Number of PBs: {pencilBeamNo}",
                  f"Number of components: {componentNo}"]
        _logger.info("\n\t".join(strLog) + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(imgBase))

    return imgBase


def getInmFREDSparse(fileName: PathLike, points: Iterable[PointLike], interpreter: str = "numpy", displayInfo: bool = False) -> Sequence[SparseMatrixCSR]:
    """Get sparse matrices of point values from an influence matrix produced by FRED Monte Carlo.

    The function reads an influence matrix file produced by the FRED Monte Carlo and returns
    a list of sparse matrices of point values for each component at the requested points.

    Parameters
    ----------
    fileName : path
        Path to FRED influence matrix file to read.
    points : array_like
        An N-element iterable of 3 elements iterables with physical points.
    interpreter : {'numpy', 'cupy'}, optional
        The interpreter to be used for calculations. Use 'numpy' for CPU or 'cupy' for GPU implementation. (def. 'numpy')
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    list[scipy.sparse.csr_matrix] or list[cupy.sparse.csr_matrix]
        List of sparse matrices of point values for each component.
    """
    import numpy as np
    import cupy as cp
    from fredtools._helper import checkGPUcupy

    _isInmFRED(fileName, raiseError=True)

    # validate interpreter
    if interpreter.lower() not in ["numpy", "cupy"]:
        error = ValueError(f"Interpreter '{interpreter}' is not supported. Use 'numpy' for CPU or 'cupy' for GPU implementation.")
        _logger.error(error)
        raise error

    # validate cupy
    if interpreter.lower() == "cupy" and not checkGPUcupy():
        _logger.warning("Cupy is not available. The numpy interpreter will be used.")
        interpreter = "numpy"

    # validate points
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        error = TypeError("Parameter 'point' must be an N-element iterable of 3 elements iterables.")
        _logger.error(error)
        raise error
    pointsNo = points.shape[0]

    # create an empty basic image with FoR defined in Inm
    imgBase = getInmFREDBaseImg(fileName, dtype="uint8")

    # convert physical points to indices
    points = np.round((points-imgBase.GetOrigin())/imgBase.GetSpacing()).astype(int)  # equivalent to transformPhysicalPointToIndex but faster
    indices = np.ravel_multi_index(tuple(np.array(points).T), imgBase.GetSize(), order="F")

    # get pencil beams info
    imnInfo = getInmFREDInfo(fileName)

    # read the influence matrix file depending on the version
    InmFREDVersion = getInmFREDVersion(fileName)
    match InmFREDVersion:
        case 2.0 | 2.1:
            if interpreter == "cupy":
                _logger.warning("Cupy interpreter is not supported for version 2.0 of the Inm file. The numpy interpreter will be used to read the Imn file and then the result will be uploaded to GPU.")
            listInmSparse = _getInmFREDSparseVersion2(fileName, indices, imnInfo, imgBase)
            if interpreter == "cupy":
                listInmSparse = [cp.sparse.csr_matrix(InmSparse) for InmSparse in listInmSparse]

        case 3.0 | 3.1:
            listInmSparse = _getInmFREDSparseVersion3(fileName, indices, imnInfo, imgBase, interpreter=interpreter)
        case _:
            error = NotImplementedError(f"Version {InmFREDVersion} of the Inm file is not supported.")
            _logger.error(error)
            raise error

    if displayInfo:
        strLog = [f"Number of points: {pointsNo}",
                  f"Number of PBs: {imnInfo.shape[0]}",
                  f"Number of fields: {len(imnInfo.FID.unique())}",
                  f"Number of components: {len(listInmSparse)}",
                  f"Stored elements per component: {[InmSparse.nnz for InmSparse in listInmSparse]}"]
        _logger.info("\n\t" + "\n\t".join(strLog))

    return listInmSparse


def _getInmFREDSparseVersion2(fileName: PathLike, indices: ArrayLike, imnInfo: DataFrame, imgBase: SITKImage) -> Sequence[SparseMatrixCSR]:
    """Get sparse matrices of point values from the FRED influence matrix file version 2.0."""
    import struct
    import numpy as np
    from scipy import sparse

    InmFREDVersion = getInmFREDVersion(fileName)

    # get number of PBs and components
    with open(fileName, "rb") as file_h:
        match InmFREDVersion:
            case 2.0:
                headerSize = 48
                [_, _, _, _, _, _, _, _, _, _, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(headerSize))
            case 2.1:
                headerSize = 84
                [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, componentNo, pencilBeamNo] = struct.unpack("<4i6f9f2i", file_h.read(headerSize))
            case _:
                error = NotImplementedError(f"Version {InmFREDVersion} of the Inm file is not supported.")
                _logger.error(error)
                raise error

    with open(fileName, "rb") as file_h:
        file_h.seek(headerSize, 1)  # skip header
        # list of sparse matrices of point values for each component
        listInmSparse = [sparse.lil_array((pencilBeamNo, np.prod(imgBase.GetSize())), dtype=np.float32) for _ in range(componentNo)]

        for PBIdx, inmInfoRow in imnInfo.iterrows():
            # skip the header of PB with PBtag and number of voxels
            file_h.seek(8, 1)

            # get number of voxels to read
            voxelsNo = int(inmInfoRow.voxelsNo)
            # read voxel indices
            voxelIndices = np.frombuffer(file_h.read(voxelsNo * 4), dtype="uint32", count=voxelsNo)

            # generate mask of voxelIndices length with marked positions overlapping with requested indices
            voxelIndicesMask = np.isin(voxelIndices, indices, assume_unique=True)
            # get only voxelIndices that overly with the requested indices
            voxelIndices = voxelIndices[voxelIndicesMask]
            if voxelIndices.size == 0:
                continue

            # read voxel values
            voxelValues = np.frombuffer(file_h.read(voxelsNo * componentNo * 4), dtype="float32", count=voxelsNo*componentNo)

            for component in range(componentNo):
                listInmSparse[component][PBIdx, voxelIndices] = voxelValues[component::componentNo][voxelIndicesMask]

    return [sparse.csr_matrix(InmSparse) for InmSparse in listInmSparse]


def _getInmFREDSparseVersion3(fileName: PathLike, indices: ArrayLike, imnInfo: DataFrame, imgBase: SITKImage, interpreter: str = "numpy") -> Sequence[SparseMatrixCSR]:
    """Get sparse matrices of point values from the FRED influence matrix file version 3.0."""
    import struct
    import numpy as np
    from scipy import sparse
    import cupy as cp

    # determine interpreter
    match interpreter.lower():
        case "cupy":
            xp = cp
            indices = cp.asarray(indices)
        case "numpy":
            xp = np
        case _:
            error = ValueError(f"Interpreter '{interpreter}' is not supported.")
            _logger.error(error)
            raise error

    InmFREDVersion = getInmFREDVersion(fileName)

    with open(fileName, "rb") as file_h:
        match InmFREDVersion:
            case 3.0:
                headerSize = 48
                [_, _, _, _, _, _, _, _, _, _, componentNo, pencilBeamNo] = struct.unpack("<4i6f2i", file_h.read(headerSize))
                file_h.seek(3 * pencilBeamNo * 4, 1)  # skip PB mapping
                componentsDataSize = np.frombuffer(file_h.read(4 * componentNo), dtype="uint32", count=componentNo)
            case 3.1:
                headerSize = 84
                [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, componentNo, pencilBeamNo] = struct.unpack("<4i6f9f2i", file_h.read(headerSize))
                file_h.seek(3 * pencilBeamNo * 4, 1)  # skip PB mapping
                componentsDataSize = np.frombuffer(file_h.read(4 * componentNo), dtype="uint32", count=componentNo)
            case _:
                error = NotImplementedError(f"Version {InmFREDVersion} of the Inm file is not supported.")
                _logger.error(error)
                raise error

    size = np.prod(imgBase.GetSize())

    with open(fileName, "rb") as file_h:
        file_h.seek(headerSize + 3 * pencilBeamNo * 4 + 4 * componentNo)  # skip header

        listInmSparse = []
        for component in range(componentNo):
            componentDataSize = componentsDataSize[component]
            pbIdx = xp.frombuffer(file_h.read(componentDataSize*4), dtype="uint32", count=componentDataSize)
            voxelIdx = xp.frombuffer(file_h.read(componentDataSize*4), dtype="uint32", count=componentDataSize)
            voxelData = xp.frombuffer(file_h.read(componentDataSize*4), dtype="float32", count=componentDataSize)

            # filter sparse matrix to requested indices/points
            voxelIdxRequestedMask = xp.isin(voxelIdx, indices, assume_unique=True)
            pbIdx = pbIdx[voxelIdxRequestedMask]
            voxelIdx = voxelIdx[voxelIdxRequestedMask]
            voxelData = voxelData[voxelIdxRequestedMask]

            if interpreter == "numpy":
                inmPointSparse = sparse.csr_matrix((voxelData, (pbIdx, voxelIdx)), shape=(pencilBeamNo, size))
            elif interpreter == "cupy":
                inmPointSparse = cp.sparse.csr_matrix((voxelData, (pbIdx, voxelIdx)), shape=(pencilBeamNo, size))
            else:
                error = NotImplementedError(f"Interpreter '{interpreter}' is not supported.")
                _logger.error(error)
                raise error

            listInmSparse.append(inmPointSparse)

    return listInmSparse
