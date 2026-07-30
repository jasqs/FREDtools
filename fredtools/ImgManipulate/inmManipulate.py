from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def inmSumVec(inmSparse: SparseMatrixCSR, weights: Iterable[Numeric], displayInfo: bool = False) -> NDArray:
    """Sum up the influence matrix to a vector.

    The function sums up the influence matrix for a given set of pencil beams
    and their weights. The influence matrix must be a sparse matrix. The function
    returns a summed influence matrix as an array. The sparse matrix can be given
    as an instance of a scipy.sparse.csr_matrix or cupy.sparse.csr_matrix object.
    In case of the cupy.sparse.csr_matrix object, the multiplication and summing
    will be performed on GPU.

    Parameters
    ----------
    inmSparse : scipy.sparse.csr_matrix or cupy.sparse.csr_matrix
        Sparse matrix of the influence matrix.
    weights : array_like
        Array of weights for each pencil beam.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        Summed influence matrix. A cupy array is returned when `inmSparse`
        is a cupy sparse matrix, and a numpy array otherwise.

    Raises
    ------
    ValueError
        If `inmSparse` is not a sparse matrix, or if the number of weights
        is not equal to the number of pencil beams in the influence matrix.

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

    weightsArray = xp.asarray(weights)
    if inmSparse.shape[0] != weightsArray.shape[0]:
        error = ValueError("Number of weights must be equal to the number of pencil beams in the influence matrix.")
        _logger.error(error)
        raise error

    # sum up the influence matrix
    vecSum = xp.asarray(inmSparse.T.dot(weightsArray))

    if displayInfo:
        strLog = [f"Summed {inmSparse.shape[0]} PBs.",
                  f"Number of voxels: {inmSparse.shape[1]}",
                  f"Sum of image: {vecSum.sum()}"]
        _logger.info("\n\t".join(strLog))

    return vecSum


def inmSumImg(inmSparse: SparseMatrixCSR, weights: Iterable[Numeric], imgBase: SITKImage, displayInfo: bool = False) -> SITKImage:
    """Sum up the influence matrix and create an image.

    The function sums up the influence matrix for a given set of pencil beams
    and their weights. The influence matrix must be a sparse matrix and the number
    of its columns must be equal to the total number of voxels of `imgBase`,
    i.e. the product of the `imgBase` size in each direction. The function
    returns a summed influence image defined as an instance of a SimpleITK object
    that inherits the frame of reference of `imgBase`. If the summed influence
    vector is a cupy array (i.e. `inmSparse` is a cupy sparse matrix), it is
    converted to a numpy array before the image is built.
    The function is useful for calculating the sum of the influence matrix for a set of pencil beams.

    Parameters
    ----------
    inmSparse : scipy.sparse.csr_matrix or cupy.sparse.csr_matrix
        Sparse matrix of the influence matrix.
    weights : array_like
        Array of weights for each pencil beam.
    imgBase : SimpleITK.Image
        Base image for the influence matrix.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    SimpleITK.Image
        Summed influence image with the frame of reference of `imgBase`.

    Raises
    ------
    ValueError
        If `inmSparse` is not a sparse matrix, if the number of weights is
        not equal to the number of pencil beams in the influence matrix, or
        if the number of columns of `inmSparse` is not equal to the total
        number of voxels of `imgBase`.

    See Also
    --------
        inmSumVec : sum up the influence matrix to a vector.
    """
    import fredtools as ft
    import SimpleITK as sitk

    vecInmSum = inmSumVec(inmSparse, weights)

    if ft._helper.checkGPUcupy():
        import cupy as cp
        if cp.get_array_module(vecInmSum) is cp:
            vecInmSum = cp.asnumpy(vecInmSum)

    imgInmSum = sitk.GetImageFromArray(np.reshape(vecInmSum, imgBase.GetSize()[::-1]))
    imgInmSum.CopyInformation(imgBase)

    if displayInfo:
        _logger.info(ft.ImgAnalyse.imgInfo._displayImageInfo(imgInmSum))

    return imgInmSum
