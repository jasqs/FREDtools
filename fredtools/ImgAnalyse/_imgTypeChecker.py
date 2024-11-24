from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)

#### ITK Image checkers ####


def isITK2D(img: Any, raiseError: bool = False) -> bool:
    """Check if input is a 2D itk.Image object and raise error if requested."""

    if isITK(img, raiseError=raiseError) and img.ndim == 2:
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is not an instance of a 2D itk.Image object.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isITK3D(img: Any, raiseError: bool = False) -> bool:
    """Check if input is a 3D itk.Image object and raise error if requested."""

    if isITK(img, raiseError=raiseError) and img.ndim == 3:
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is not an instance of a #D itk.Image object.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isITK4D(img: Any, raiseError: bool = False) -> bool:
    """Check if input is a 4D itk.Image object and raise error if requested."""

    if isITK(img, raiseError=raiseError) and img.ndim == 4:
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is not an instance of a 4D itk.Image object.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isITK(img: Any, raiseError: bool = False) -> bool:
    """Check if input is an itk.Image object and raise error if requested."""

    if isinstance(img, ITKImage):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is not an instance of an itk.Image object.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False

#### SimleITK Image checkers ####


def isSITK2D(img: Any, raiseError: bool = False) -> bool:
    """Check if input is a 2D SimpleITK.Image object and raise error if requested."""

    if isSITK(img, raiseError=raiseError) and img.GetDimension() == 2:
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is not an instance of a 2D SimpleITK.Image object.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK3D(img: Any, raiseError: bool = False) -> bool:
    """Check if input is a 3D SimpleITK.Image object and raise error if requested."""

    if isSITK(img, raiseError=raiseError) and img.GetDimension() == 3:
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is not an instance of a 3D SimpleITK.Image object.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK4D(img: Any, raiseError: bool = False) -> bool:
    """Check if input is a 4D SimpleITK.Image object and raise error if requested."""

    if isSITK(img, raiseError=raiseError) and img.GetDimension() == 4:
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is not an instance of a 4D SimpleITK.Image object.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK(img: Any, raiseError: bool = False) -> bool:
    """Check if input is an SimpleITK.Image object and raise error if requested."""

    if isinstance(img, SITKImage):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is not an instance of a SimpleITK.Image object.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK_point(img: Any, raiseError: bool = False) -> bool:

    if isSITK(img, raiseError=raiseError) and img.GetSize().count(1) == (img.GetDimension() - 0):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is an instance of a SimpleITK.Image object but does not describe a point. Size of 'img' is {img.GetSize()}.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK_profile(img: Any, raiseError: bool = False) -> bool:

    if isSITK(img, raiseError=raiseError) and img.GetSize().count(1) == (img.GetDimension() - 1):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is an instance of a SimpleITK.Image object but does not describe a profile. Size of 'img' is {img.GetSize()}.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK_slice(img: Any, raiseError: bool = False) -> bool:

    if isSITK(img, raiseError=raiseError) and img.GetSize().count(1) == (img.GetDimension() - 2):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is an instance of a SimpleITK.Image object but does not describe a slice. Size of 'img' is {img.GetSize()}.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK_volume(img: Any, raiseError: bool = False) -> bool:

    if isSITK(img, raiseError=raiseError) and img.GetSize().count(1) == (img.GetDimension() - 3):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is an instance of a SimpleITK.Image object but does not describe a volume. Size of 'img' is {img.GetSize()}.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK_timevolume(img: Any, raiseError: bool = False) -> bool:

    if isSITK(img, raiseError=raiseError) and img.GetSize().count(1) == (img.GetDimension() - 4):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is an instance of a SimpleITK.Image object but does not describe a time volume. Size of 'img' is {img.GetSize()}.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK_vector(img: Any, raiseError: bool = False) -> bool:

    if isSITK(img, raiseError=raiseError) and "vector" in img.GetPixelIDTypeAsString():
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is an instance of a SimspleITK.Image object but not a vector image.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK_transform(img: SITKTransform, raiseError: bool = False) -> bool:

    if isinstance(img, SITKTransform):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is not an instance of a SimpleITK.Transform object.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False

#### SimleITK Image Mask checkers ####


def isSITK_maskBinary(img: Any, raiseError: bool = False) -> bool:
    import fredtools as ft
    from SimpleITK import sitkUInt8

    stat = ft.getStatistics(img)

    if isSITK(img, raiseError=raiseError) and ((stat.GetMaximum() in [0, 1]) and (stat.GetMinimum() in [0, 1]) and (img.GetPixelID() == sitkUInt8)):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is an instance of a SimspleITK.Image object but does not describe a binary mask. Binary mask image must be of type '8-bit unsigned integer' and contain only voxels with values 0 or 1.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK_maskFloating(img: Any, raiseError: bool = False) -> bool:
    import fredtools as ft
    from SimpleITK import sitkFloat32, sitkFloat64

    stat = ft.getStatistics(img)

    if isSITK(img, raiseError=raiseError) and ((stat.GetMaximum() <= 1) and (stat.GetMinimum() >= 0) and ((img.GetPixelID() == sitkFloat64) or (img.GetPixelID() == sitkFloat32))):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is an instance of a SimspleITK.Image object but does not describe a floating mask. Floating mask image must be of type '32-bit float' or '64-bit float' and contain only voxels with values in range 0-1.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def isSITK_mask(img: Any, raiseError: bool = False) -> bool:

    if isSITK(img, raiseError=raiseError) and ((isSITK_maskBinary(img) or isSITK_maskFloating(img))):
        return True
    else:
        error = TypeError(f"The object '{type(img)}' is an instance of a SimspleITK.Image object but does not describe floating nor binary mask.")

        if raiseError:
            _logger.error(error)
            raise error
        else:
            return False


def getMaskType(img: Any) -> str:

    isSITK(img, raiseError=True)
    isSITK_mask(img, raiseError=True)

    if isSITK_maskBinary(img):
        maskType = "binary"
    elif isSITK_maskFloating(img):
        maskType = "floating"
    else:
        maskType = "unknown"

    _logger.debug(f"The mask type is {maskType}")

    return maskType
