def _setSITKInterpolator(interpolation="linear", splineOrder=3):
    """Set SimpleITK interpolator for interpolation method.

    The function is setting a specific interpolation method for
    SimpleITK image objects.

    Parameters
    ----------
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)

    Returns
    -------
    interpolator
        An object of a SimpleITK interpolator.
    """
    import SimpleITK as sitk

    # set interpolation method
    if interpolation.lower() == "linear":
        return sitk.sitkLinear
    elif interpolation.lower() == "nearest":
        return sitk.sitkNearestNeighbor
    elif interpolation.lower() == "spline":
        if splineOrder > 5 or splineOrder < 0:
            raise ValueError(f"Spline order must be in range 0-5.")
        if splineOrder == 0:
            return sitk.sitkBSplineResampler
        elif splineOrder == 1:
            return sitk.sitkBSplineResamplerOrder1
        elif splineOrder == 2:
            return sitk.sitkBSplineResamplerOrder2
        elif splineOrder == 3:
            return sitk.sitkBSplineResamplerOrder3
        elif splineOrder == 4:
            return sitk.sitkBSplineResamplerOrder4
        elif splineOrder == 5:
            return sitk.sitkBSplineResamplerOrder5
    else:
        raise ValueError(f"Interpolation type '{interpolation}' cannot be recognized. Only 'linear', 'nearest' and 'spline' are supported.")


def _copyImgMetaData(imgSrc, imgDes):
    """Copy meta data to the image source to the image destination"""
    _isSITK(imgSrc, raiseError=True)
    _isSITK(imgDes, raiseError=True)
    for key in imgSrc.GetMetaDataKeys():
        imgDes.SetMetaData(key, imgSrc.GetMetaData(key))
    return imgDes


re_number = r"[-+]?[\d]+\.?[\d]*[Ee]?(?:[-+]?[\d]+)?"
