from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def inmSumVec(imnSparse: SparseMatrixCSR, weigths: Iterable[Numberic], displayInfo: bool = False) -> NDArray:

    import cupy as cp
    xp = cp.get_array_module(imnSparse)
    if imnSparse.shape is None:
        error = ValueError("The influence matrix must be a sparse matrix.")
        _logger.error(error)
        raise error

    weigthsArray = xp.asarray(weigths)
    if imnSparse.shape[0] != weigthsArray.shape[0]:
        error = ValueError("Number of weights must be equal to the number of pencil beams in the influence matrix.")
        _logger.error(error)
        raise error

    # sum up the influence matrix
    vecSum = xp.asarray(imnSparse.T.dot(weigthsArray))

    if displayInfo:
        strLog = [f"Summed {imnSparse.shape[0]} PBs.",
                  f"Number of voxels: {imnSparse.shape[1]}",
                  f"Sum of image: {vecSum.sum()}"]
        _logger.info("\n\t".join(strLog))

    return vecSum


def inmSumImg(imnSparse: SparseMatrixCSR, weigths: Iterable[Numberic], imgBase: SITKImage, displayInfo: bool = False) -> SITKImage:

    import fredtools as ft
    import SimpleITK as sitk
    import cupy as cp

    vecInmSum = inmSumVec(imnSparse, weigths)

    if cp.get_array_module(vecInmSum) is cp:
        vecInmSum = cp.asnumpy(vecInmSum)

    imgInmSum = sitk.GetImageFromArray(np.reshape(vecInmSum, imgBase.GetSize()[::-1]))
    imgInmSum.CopyInformation(imgBase)

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgInmSum))

    return imgInmSum
