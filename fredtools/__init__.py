# from .ft_io_dicom import *
from .ft_imgIO import *

# from .ft_io_mhd import *
# from .ft_io_map3d import *

from .ft_imgAnalyse import *
from .ft_imgGetSubimg import *
from .ft_imgManipulate import *
from .ft_simTools import *

import itk
import SimpleITK as sitk

import sys

version_info = [0, 3, 0]
__version__ = ".".join(map(str, version_info))


def _currentFuncName(n=0):
    r"""Get name of the function where the currentFuncName() is called.
    currentFuncName(1) get the name of the caller.
    """
    return sys._getframe(n + 1).f_code.co_name


def _isITK2D(img, raiseError=False):
    r"""Check if input is a 2D itk.Image object and raise error if requested."""
    instanceBool = isinstance(img, itk.Image) & (img.ndim == 2)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 2D itk image")
    return instanceBool


def _isITK3D(img, raiseError=False):
    r"""Check if input is a 3D itk.Image object and raise error if requested."""
    instanceBool = isinstance(img, itk.Image) & (img.ndim == 3)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 3D itk image")
    return instanceBool


def _isITK4D(img, raiseError=False):
    r"""Check if input is a 4D itk.Image object and raise error if requested."""
    instanceBool = isinstance(img, itk.Image) & (img.ndim == 4)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 4D itk image")
    return instanceBool


def _isITK(img, raiseError=False):
    r"""Check if input is an itk.Image object and raise error if requested."""
    instanceBool = isinstance(img, itk.Image)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of an itk image.")
    return instanceBool


def _isSITK2D(img, raiseError=False):
    r"""Check if input is a 2D SimpleITK.Image object and raise error if requested."""
    instanceBool = isinstance(img, sitk.Image) & (img.GetDimension() == 2)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 2D SimpleITK image")
    return instanceBool


def _isSITK3D(img, raiseError=False):
    r"""Check if input is a 3D SimpleITK.Image object and raise error if requested."""
    instanceBool = isinstance(img, sitk.Image) & (img.GetDimension() == 3)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 3D SimpleITK image")
    return instanceBool


def _isSITK4D(img, raiseError=False):
    r"""Check if input is a 4D SimpleITK.Image object and raise error if requested."""
    instanceBool = isinstance(img, sitk.Image) & (img.GetDimension() == 4)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 4D SimpleITK image")
    return instanceBool


def _isSITK(img, raiseError=False):
    r"""Check if input is an SimpleITK.Image object and raise error if requested."""
    instanceBool = isinstance(img, sitk.Image)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of an SimpleITK image.")
    return instanceBool


def _isSITK_slice(img, raiseError=False):

    ft._isSITK(img, raiseError=True)
    instanceBool = img.GetSize().count(1) == (img.GetDimension() - 2)

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK image but not describing a slice. Size of 'img' is {img.GetSize()}")
    return instanceBool


def _isSITK_profile(img, raiseError=False):

    ft._isSITK(img, raiseError=True)
    instanceBool = img.GetSize().count(1) == (img.GetDimension() - 1)

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK image but not describing a profile. Size of 'img' is {img.GetSize()}")
    return instanceBool


def _isSITK_point(img, raiseError=False):

    ft._isSITK(img, raiseError=True)
    instanceBool = img.GetSize().count(1) == (img.GetDimension())

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimspleITK image but not describing a point. Size of 'img' is {img.GetSize()}")
    return instanceBool


def _isSITK_mask(img, raiseError=False):

    ft._isSITK(img, raiseError=True)
    stat = getStatistics(img)
    instanceBool = (stat.GetMaximum() == 1) and (stat.GetMinimum() == 0) and (img.GetPixelIDTypeAsString() == "8-bit unsigned integer")

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimspleITK image but not describing a simple mask. Mask image should contain only voxels with values 0 and 1.")
    return instanceBool


def SITK2ITK(imgSITK):
    r"""Convert image from SimpleITK.Image object to ITK.Image object."""
    ft._isSITK(imgSITK, raiseError=True)
    imgITK = itk.GetImageFromArray(sitk.GetArrayFromImage(imgSITK), is_vector=imgSITK.GetNumberOfComponentsPerPixel() > 1)
    imgITK.SetOrigin(imgSITK.GetOrigin())
    imgITK.SetSpacing(imgSITK.GetSpacing())
    imgITK.SetDirection(itk.GetMatrixFromArray(np.array(imgSITK.GetDirection()).reshape(imgSITK.GetDimension(), imgSITK.GetDimension())))
    return imgITK


def ITK2SITK(imgITK):
    r"""Convert image from ITK.Image object to SimpleITK.Image object."""
    ft._isITK(imgITK, raiseError=True)
    imgSITK = sitk.GetImageFromArray(itk.GetArrayFromImage(imgITK), isVector=imgITK.GetNumberOfComponentsPerPixel() > 1)
    imgSITK.SetOrigin(list(imgITK.GetOrigin()))
    imgSITK.SetSpacing(list(imgITK.GetSpacing()))
    imgSITK.SetDirection(itk.GetArrayFromMatrix(imgITK.GetDirection()).flatten())
    return imgSITK
