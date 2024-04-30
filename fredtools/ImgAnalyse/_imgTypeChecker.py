from itk import Image as ITKImage
from SimpleITK import Image as SITKImage
from SimpleITK import Transform as SITKTransform


def isITK2D(img, raiseError=False):
    """Check if input is a 2D itk.Image object and raise error if requested."""
    if not isITK(img, raiseError=raiseError):
        return False
    elif not img.ndim == 2:
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is not an instance of a 2D itk.Image object.")
        return False
    else:
        return True


def isITK3D(img, raiseError=False):
    """Check if input is a 3D itk.Image object and raise error if requested."""
    if not isITK(img, raiseError=raiseError):
        return False
    elif not img.ndim == 3:
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is not an instance of a 3D itk.Image object.")
        return False
    else:
        return True


def isITK4D(img, raiseError=False):
    """Check if input is a 4D itk.Image object and raise error if requested."""
    if not isITK(img, raiseError=raiseError):
        return False
    elif not img.ndim == 4:
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is not an instance of a 4D itk.Image object.")
        return False
    else:
        return True


def isITK(img: ITKImage, raiseError=False):
    """Check if input is an itk.Image object and raise error if requested."""
    if isinstance(img, ITKImage):
        return True
    elif raiseError:
        raise TypeError(f"The object '{type(img)}' is not an instance of an itk.Image object.")
    else:
        return False


def isSITK2D(img, raiseError=False):
    """Check if input is a 2D SimpleITK.Image object and raise error if requested."""
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not img.GetDimension() == 2:
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is not an instance of a 2D SimpleITK.Image object.")
        return False
    else:
        return True


def isSITK3D(img, raiseError=False):
    """Check if input is a 3D SimpleITK.Image object and raise error if requested."""
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not img.GetDimension() == 3:
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is not an instance of a 3D SimpleITK.Image object.")
        return False
    else:
        return True


def isSITK4D(img, raiseError=False):
    """Check if input is a 4D SimpleITK.Image object and raise error if requested."""
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not img.GetDimension() == 4:
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is not an instance of a 4D SimpleITK.Image object.")
        return False
    else:
        return True


def isSITK(img, raiseError=False):
    """Check if input is an SimpleITK.Image object and raise error if requested."""
    if isinstance(img, SITKImage):
        return True
    elif raiseError:
        raise TypeError(f"The object '{type(img)}' is not an instance of an SimpleITK.Image object.")
    else:
        return False


def isSITK_volume(img, raiseError=False):
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not img.GetSize().count(1) == (img.GetDimension() - 3):
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK.Image object but does not describe a volume. Size of 'img' is {img.GetSize()}.")
        return False
    else:
        return True


def isSITK_timevolume(img, raiseError=False):
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not img.GetSize().count(1) == (img.GetDimension() - 4):
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK.Image object but does not describe a time volume. Size of 'img' is {img.GetSize()}.")
        return False
    else:
        return True


def isSITK_slice(img, raiseError=False):
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not img.GetSize().count(1) == (img.GetDimension() - 2):
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK.Image object but does not describe a slice. Size of 'img' is {img.GetSize()}.")
        return False
    else:
        return True


def isSITK_profile(img, raiseError=False):
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not img.GetSize().count(1) == (img.GetDimension() - 1):
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK.Image object but does not describe a profile. Size of 'img' is {img.GetSize()}.")
        return False
    else:
        return True


def isSITK_point(img, raiseError=False):
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not img.GetSize().count(1) == img.GetDimension():
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is an instance of SimpleITK.Image object but does not describe a point. Size of 'img' is {img.GetSize()}.")
        return False
    else:
        return True


def isSITK_vector(img, raiseError=False):
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not "vector" in img.GetPixelIDTypeAsString():
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is an instance of SimspleITK.Image object but not a vector image.")
        return False
    else:
        return True


def isSITK_transform(img, raiseError=False):
    if isinstance(img, SITKTransform):
        return True
    elif raiseError:
        raise TypeError(f"The object '{type(img)}' is not an instance of an SimpleITK.Transform object.")
    else:
        return False


def isSITK_maskBinary(img, raiseError=False):
    import fredtools as ft
    from SimpleITK import sitkUInt8

    stat = ft.getStatistics(img)

    if not isSITK(img, raiseError=raiseError):
        return False
    elif not ((stat.GetMaximum() in [0, 1]) and (stat.GetMinimum() in [0, 1]) and (img.GetPixelID() == sitkUInt8)):
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is an instance of SimspleITK.Image object but does not describe a binary mask. Binary mask image must be of type '8-bit unsigned integer' and contain only voxels with values 0 or 1.")
        return False
    else:
        return True


def isSITK_maskFloating(img, raiseError=False):
    import fredtools as ft
    from SimpleITK import sitkFloat32, sitkFloat64

    stat = ft.getStatistics(img)

    if not isSITK(img, raiseError=raiseError):
        return False
    elif not ((stat.GetMaximum() <= 1) and (stat.GetMinimum() >= 0) and ((img.GetPixelID() == sitkFloat64) or (img.GetPixelID() == sitkFloat32))):
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is an instance of SimspleITK.Image object but does not describe a floating mask. Floating mask image must be of type '32-bit float' or '64-bit float' and contain only voxels with values in range 0-1.")
        return False
    else:
        return True


def isSITK_mask(img, raiseError=False):
    if not isSITK(img, raiseError=raiseError):
        return False
    elif not (isSITK_maskBinary(img) or isSITK_maskFloating(img)):
        if raiseError:
            raise TypeError(f"The object '{type(img)}' is an instance of SimspleITK.Image object but does not describe floating nor binary mask.")
        return False
    else:
        return True


def getMaskType(img):
    import fredtools as ft
    import SimpleITK as sitk

    isSITK(img, raiseError=True)
    isSITK_mask(img, raiseError=True)

    if isSITK_maskBinary(img):
        return "binary"
    elif isSITK_maskFloating(img):
        return "floating"
    else:
        return "unknown"
