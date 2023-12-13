from .ft_imgIO import *
from .ft_imgAnalyse import *
from .ft_imgGetSubimg import *
from .ft_imgManipulate import *
from .ft_simTools import *
from .ft_dvh import *
from .ft_braggPeak import *
from .ft_spotAnalyse import *
from .ft_displayImg import *
from .ft_optimisation import *
from .ft_gammaIndex import *
from .ft_misc import *

import itk
import SimpleITK as sitk

import sys

version_info = [0, 7, 4]
__version__ = ".".join(map(str, version_info))


def _checkJupyterMode():
    """Check if the FREDtools was loaded from jupyter"""
    try:
        if get_ipython().config["IPKernelApp"]:
            return True
    except:
        return False


def _checkMatplotlibBackend():
    import matplotlib

    if "inline" in matplotlib.get_backend():
        return "inline"
    elif "ipympl" in matplotlib.get_backend():
        return "ipympl"
    else:
        return "unknown"


def _currentFuncName(n=0):
    r"""Get name of the function where the currentFuncName() is called.
    currentFuncName(1) get the name of the caller.
    """
    return sys._getframe(n + 1).f_code.co_name


def _isITK2D(img, raiseError=False):
    r"""Check if input is a 2D itk.Image object and raise error if requested."""
    import itk

    instanceBool = isinstance(img, itk.Image) & (img.ndim == 2)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 2D itk image")
    return instanceBool


def _isITK3D(img, raiseError=False):
    r"""Check if input is a 3D itk.Image object and raise error if requested."""
    import itk

    instanceBool = isinstance(img, itk.Image) & (img.ndim == 3)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 3D itk image")
    return instanceBool


def _isITK4D(img, raiseError=False):
    r"""Check if input is a 4D itk.Image object and raise error if requested."""
    import itk

    instanceBool = isinstance(img, itk.Image) & (img.ndim == 4)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 4D itk image")
    return instanceBool


def _isITK(img, raiseError=False):
    r"""Check if input is an itk.Image object and raise error if requested."""
    import itk

    instanceBool = isinstance(img, itk.Image)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of an itk image.")
    return instanceBool


def _isSITK2D(img, raiseError=False):
    r"""Check if input is a 2D SimpleITK.Image object and raise error if requested."""
    import SimpleITK as sitk

    instanceBool = isinstance(img, sitk.Image) & (img.GetDimension() == 2)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 2D SimpleITK image")
    return instanceBool


def _isSITK3D(img, raiseError=False):
    r"""Check if input is a 3D SimpleITK.Image object and raise error if requested."""
    import SimpleITK as sitk

    instanceBool = isinstance(img, sitk.Image) & (img.GetDimension() == 3)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 3D SimpleITK image")
    return instanceBool


def _isSITK4D(img, raiseError=False):
    r"""Check if input is a 4D SimpleITK.Image object and raise error if requested."""
    import SimpleITK as sitk

    instanceBool = isinstance(img, sitk.Image) & (img.GetDimension() == 4)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of a 4D SimpleITK image")
    return instanceBool


def _isSITK(img, raiseError=False):
    r"""Check if input is an SimpleITK.Image object and raise error if requested."""
    import SimpleITK as sitk

    instanceBool = isinstance(img, sitk.Image)
    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is not an instance of an SimpleITK image.")
    return instanceBool


def _isSITK_volume(img, raiseError=False):
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    instanceBool = img.GetSize().count(1) == (img.GetDimension() - 3)

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK image but does not describe a volume. Size of 'img' is {img.GetSize()}")
    return instanceBool


def _isSITK_timevolume(img, raiseError=False):
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    instanceBool = img.GetSize().count(1) == (img.GetDimension() - 4)

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK image but does not describe a time volume. Size of 'img' is {img.GetSize()}")
    return instanceBool


def _isSITK_slice(img, raiseError=False):
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    instanceBool = img.GetSize().count(1) == (img.GetDimension() - 2)

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK image but does not describe a slice. Size of 'img' is {img.GetSize()}")
    return instanceBool


def _isSITK_profile(img, raiseError=False):
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    instanceBool = img.GetSize().count(1) == (img.GetDimension() - 1)

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK image but does not describe a profile. Size of 'img' is {img.GetSize()}")
    return instanceBool


def _isSITK_point(img, raiseError=False):
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    instanceBool = img.GetSize().count(1) == (img.GetDimension())

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimspleITK image but does not describe a point. Size of 'img' is {img.GetSize()}")
    return instanceBool


def _isSITK_maskBinary(img, raiseError=False):
    import fredtools as ft
    import SimpleITK as sitk

    ft._isSITK(img, raiseError=True)
    stat = getStatistics(img)
    instanceBool = (stat.GetMaximum() in [0, 1]) and (stat.GetMinimum() in [0, 1]) and (img.GetPixelID() == sitk.sitkUInt8)

    if raiseError and not instanceBool:
        raise TypeError(
            f"The object '{type(img)}' is an instance of SimspleITK image but does not describe a binary mask. Binary mask image must be of type '8-bit unsigned integer' and contain only voxels with values 0 or 1."
        )
    return instanceBool


def _isSITK_maskFloating(img, raiseError=False):
    import fredtools as ft
    import SimpleITK as sitk

    ft._isSITK(img, raiseError=True)
    stat = getStatistics(img)
    instanceBool = (stat.GetMaximum() <= 1) and (stat.GetMinimum() >= 0) and ((img.GetPixelID() == sitk.sitkFloat64) or (img.GetPixelID() == sitk.sitkFloat32))

    if raiseError and not instanceBool:
        raise TypeError(
            f"The object '{type(img)}' is an instance of SimspleITK image but does not describe a floating mask. Floating mask image must be of type '32-bit float' or '64-bit float' and contain only voxels with values in range 0-1."
        )
    return instanceBool


def _isSITK_mask(img, raiseError=False):
    import fredtools as ft
    import SimpleITK as sitk

    ft._isSITK(img, raiseError=True)

    instanceBool = _isSITK_maskBinary(img) or _isSITK_maskFloating(img)

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimspleITK image but does not describe floating nor binary mask.")
    return instanceBool


def _getMaskType(img):
    import fredtools as ft
    import SimpleITK as sitk

    ft._isSITK(img, raiseError=True)
    ft._isSITK_mask(img, raiseError=True)

    if _isSITK_maskBinary(img):
        return "binary"
    elif _isSITK_maskFloating(img):
        return "floating"
    else:
        return "unknown"


def _isSITK_vector(img, raiseError=False):
    import fredtools as ft

    ft._isSITK(img, raiseError=True)
    instanceBool = "vector" in img.GetPixelIDTypeAsString()

    if raiseError and not instanceBool:
        raise TypeError(f"The object '{type(img)}' is an instance of SimspleITK image but not vector image.")
    return instanceBool


def _isSITK_transform(img, raiseError=False):
    try:
        return "Transform" in img.GetName()
    except:
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is not an instance of SimspleITK transform object.")


def _copyImgMetaData(imgSrc, imgDes):
    """Copy meta data to the image source to the image destination"""
    _isSITK(imgSrc, raiseError=True)
    _isSITK(imgDes, raiseError=True)
    for key in imgSrc.GetMetaDataKeys():
        imgDes.SetMetaData(key, imgSrc.GetMetaData(key))
    return imgDes


def SITK2ITK(imgSITK):
    r"""Convert image from SimpleITK.Image object to ITK.Image object."""
    import numpy as np
    import itk
    import SimpleITK as sitk
    import fredtools as ft

    ft._isSITK(imgSITK, raiseError=True)
    imgITK = itk.GetImageFromArray(sitk.GetArrayFromImage(imgSITK), is_vector=imgSITK.GetNumberOfComponentsPerPixel() > 1)
    imgITK.SetOrigin(imgSITK.GetOrigin())
    imgITK.SetSpacing(imgSITK.GetSpacing())
    imgITK.SetDirection(itk.GetMatrixFromArray(np.array(imgSITK.GetDirection()).reshape(imgSITK.GetDimension(), imgSITK.GetDimension())))
    return imgITK


def ITK2SITK(imgITK):
    r"""Convert image from ITK.Image object to SimpleITK.Image object."""
    import SimpleITK as sitk
    import fredtools as ft

    ft._isITK(imgITK, raiseError=True)
    imgSITK = sitk.GetImageFromArray(itk.GetArrayFromImage(imgITK), isVector=imgITK.GetNumberOfComponentsPerPixel() > 1)
    imgSITK.SetOrigin(list(imgITK.GetOrigin()))
    imgSITK.SetSpacing(list(imgITK.GetSpacing()))
    imgSITK.SetDirection(itk.GetArrayFromMatrix(imgITK.GetDirection()).flatten())
    return imgSITK
