def setSITKInterpolator(interpolation="linear", splineOrder=3):
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


def copyImgMetaData(imgSrc, imgDes):
    """Copy meta data to the image source to the image destination"""
    isSITK(imgSrc, raiseError=True)
    isSITK(imgDes, raiseError=True)
    for key in imgSrc.GetMetaDataKeys():
        imgDes.SetMetaData(key, imgSrc.GetMetaData(key))
    return imgDes


def checkJupyterMode():
    """Check if the FREDtools was loaded from jupyter"""
    try:
        if get_ipython().config["IPKernelApp"]:
            return True
    except:
        return False


def checkMatplotlibBackend():
    import matplotlib

    if "inline" in matplotlib.get_backend():
        return "inline"
    elif "ipympl" in matplotlib.get_backend():
        return "ipympl"
    else:
        return "unknown"


def currentFuncName(n=0):
    """Get name of the function where the currentFuncName() is called.
    currentFuncName(1) get the name of the caller.
    """
    import sys
    return sys._getframe(n + 1).f_code.co_name
