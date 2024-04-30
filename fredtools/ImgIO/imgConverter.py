def SITK2ITK(imgSITK):
    """Convert image from SimpleITK.Image object to ITK.Image object."""
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


def ITK2SITK(imgITK):
    """Convert image from ITK.Image object to SimpleITK.Image object."""
    import SimpleITK as sitk
    import fredtools as ft
    import itk

    ft.isITK(imgITK, raiseError=True)
    imgSITK = sitk.GetImageFromArray(itk.GetArrayFromImage(imgITK), isVector=imgITK.GetNumberOfComponentsPerPixel() > 1)
    imgSITK.SetOrigin(list(imgITK.GetOrigin()))
    imgSITK.SetSpacing(list(imgITK.GetSpacing()))
    imgSITK.SetDirection(itk.GetArrayFromMatrix(imgITK.GetDirection()).flatten())
    return imgSITK
