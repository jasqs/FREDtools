from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def inmSumVec(inmSparse: SparseMatrixCSR, weigths: Iterable[Numberic], displayInfo: bool = False) -> NDArray:
    """Sum up the influence matrix to a vector.

    The function sums up the influence matrix for a given set of pencil beams
    and their weights. The influence matrix must be a sparse matrix. The function
    returns a summed influence matrix as a numpy array. The sparse matrix can be given
    as an instance of a scipy.sparse.csr_matrix or cupy.sparse.csr_matrix object. 
    In case of the cupy.sparse.csr_matrix object, the multiplication and summing 
    will be perfoemd on GPU.   

    Parameters
    ----------
    imnSparse : scipy.sparse.csr_matrix or cupy.sparse.csr_matrix
        Sparse matrix of the influence matrix.
    weigths : array-like
        Array of weights for each pencil beam.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    numpy.ndarray
        Summed influence matrix.

    See Also
    --------
        inmSumImg : sum up the influence matrix and create an image.
    """
    from fredtools._helper import checkGPUcupy
    if checkGPUcupy():
        import cupy as cp
        xp = cp.get_array_module(inmSparse)
    else:
        import numpy as np
        xp = np
    if inmSparse.shape is None:
        error = ValueError("The influence matrix must be a sparse matrix.")
        _logger.error(error)
        raise error

    weigthsArray = xp.asarray(weigths)
    if inmSparse.shape[0] != weigthsArray.shape[0]:
        error = ValueError("Number of weights must be equal to the number of pencil beams in the influence matrix.")
        _logger.error(error)
        raise error

    # sum up the influence matrix
    vecSum = xp.asarray(inmSparse.T.dot(weigthsArray))

    if displayInfo:
        strLog = [f"Summed {inmSparse.shape[0]} PBs.",
                  f"Number of voxels: {inmSparse.shape[1]}",
                  f"Sum of image: {vecSum.sum()}"]
        _logger.info("\n\t".join(strLog))

    return vecSum


def inmSumImg(inmSparse: SparseMatrixCSR, weigths: Iterable[Numberic], imgBase: SITKImage, displayInfo: bool = False) -> SITKImage:
    """Sum up the influence matrix and create an image.

    The function sums up the influence matrix for a given set of pencil beams
    and their weights. The influence matrix must be a sparse matrix. The function
    returns a summed influence image defined as an instance of a SimpleITK object. 
    The function is useful for calculating the sum of the influence matrix for a set of pencil beams.

    Parameters
    ----------
    imnSparse : scipy.sparse.csr_matrix or cupy.sparse.csr_matrix
        Sparse matrix of the influence matrix.
    weigths : array-like
        Array of weights for each pencil beam.
    imgBase : SimpleITK.Image
        Base image for the influence matrix.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK.Image
        Summed influence image.

    See Also
    --------
        inmSumVec : sum up the influence matrix to a vector.
    """
    import fredtools as ft
    import SimpleITK as sitk

    vecInmSum = inmSumVec(inmSparse, weigths)

    if ft._helper.checkGPUcupy():
        import cupy as cp
        if cp.get_array_module(vecInmSum) is cp:
            vecInmSum = cp.asnumpy(vecInmSum)

    imgInmSum = sitk.GetImageFromArray(np.reshape(vecInmSum, imgBase.GetSize()[::-1]))
    imgInmSum.CopyInformation(imgBase)

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgInmSum))

    return imgInmSum
