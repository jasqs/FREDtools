from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def SITK2ITK(imgSITK: SITKImage) -> ITKImage:
    """Convert image from SimpleITK.Image object to ITK.Image object.

    The function converts a SimpleITK image object to an ITK image object,
    preserving the origin, spacing and direction. The pixel buffer is copied
    (via GetArrayFromImage/GetImageFromArray). Vector images
    (multi-component pixels) are handled.

    Parameters
    ----------
    imgSITK : SimpleITK Image
        Object of a SimpleITK image.

    Returns
    -------
    ITK Image
        Object of an ITK image.

    Raises
    ------
    TypeError
        If `imgSITK` is not an instance of a SimpleITK image object.

    See Also
    --------
    ITK2SITK : convert an ITK image to a SimpleITK image.
    """
    import numpy as np
    import itk
    import SimpleITK as sitk
    import fredtools as ft

    ft._imgTypeChecker.isSITK(imgSITK, raiseError=True)
    imgITK = itk.GetImageFromArray(sitk.GetArrayFromImage(imgSITK), is_vector=imgSITK.GetNumberOfComponentsPerPixel() > 1)
    imgITK.SetOrigin(imgSITK.GetOrigin())
    imgITK.SetSpacing(imgSITK.GetSpacing())
    imgITK.SetDirection(itk.GetMatrixFromArray(np.array(imgSITK.GetDirection()).reshape(imgSITK.GetDimension(), imgSITK.GetDimension())))
    return imgITK


def ITK2SITK(imgITK: ITKImage) -> SITKImage:
    """Convert image from ITK.Image object to SimpleITK.Image object.

    The function converts an ITK image object to a SimpleITK image object,
    preserving the origin, spacing and direction. The pixel buffer is copied
    (via GetArrayFromImage/GetImageFromArray). Vector images
    (multi-component pixels) are handled.

    Parameters
    ----------
    imgITK : ITK Image
        Object of an ITK image.

    Returns
    -------
    SimpleITK Image
        Object of a SimpleITK image.

    Raises
    ------
    TypeError
        If `imgITK` is not an instance of an ITK image object.

    See Also
    --------
    SITK2ITK : convert a SimpleITK image to an ITK image.
    """
    import SimpleITK as sitk
    import fredtools as ft
    import itk

    ft._imgTypeChecker.isITK(imgITK, raiseError=True)
    imgSITK = sitk.GetImageFromArray(itk.GetArrayFromImage(imgITK), isVector=imgITK.GetNumberOfComponentsPerPixel() > 1)
    imgSITK.SetOrigin(list(imgITK.GetOrigin()))
    imgSITK.SetSpacing(list(imgITK.GetSpacing()))
    imgSITK.SetDirection(itk.GetArrayFromMatrix(imgITK.GetDirection()).flatten())
    return imgSITK


def img2vec(img: SITKImage) -> NDArray:
    """Convert an image to a vector of voxel values.

    The function flattens a SimpleITK image to a 1D numpy array. The voxel
    array is first transposed from the numpy zyx order to the xyz order
    (swapaxes(0, -1)) and then flattened in the Fortran order
    (flatten(order='F')), so the element order matches the voxel indexing
    convention of numpy.ravel_multi_index(..., order='F') used by
    getInmFREDSparse.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image.

    Returns
    -------
    numpy ndarray
        1D numpy array of the voxel values.

    Raises
    ------
    TypeError
        If `img` is not an instance of a SimpleITK image object.

    See Also
    --------
    getInmFREDSparse : get sparse matrices of point values from an influence matrix produced by FRED Monte Carlo.
    """
    import SimpleITK as sitk
    import fredtools as ft

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    vec = np.swapaxes(sitk.GetArrayViewFromImage(img), 0, -1).flatten(order='F')

    return vec
