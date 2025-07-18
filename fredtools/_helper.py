from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def setSITKInterpolator(interpolation: Literal['linear', 'nearest', 'spline'] = "linear", splineOrder:  Annotated[int, Field(strict=True, ge=0, le=5)] = 3) -> int:
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
    interpolator = sitk.sitkLinear
    match interpolation.lower():
        case "linear":
            interpolator = sitk.sitkLinear
        case "nearest":
            interpolator = sitk.sitkNearestNeighbor
        case "spline":
            if splineOrder > 5 or splineOrder < 0:
                error = ValueError(f"Spline order must be in range 0-5.")
                _logger.error(error)
                raise error
            match splineOrder:
                case 0:
                    interpolator = sitk.sitkBSplineResampler
                case 1:
                    interpolator = sitk.sitkBSplineResamplerOrder1
                case 2:
                    interpolator = sitk.sitkBSplineResamplerOrder2
                case 3:
                    interpolator = sitk.sitkBSplineResamplerOrder3
                case 4:
                    interpolator = sitk.sitkBSplineResamplerOrder4
                case 5:
                    interpolator = sitk.sitkBSplineResamplerOrder5
        case _:
            error = ValueError(f"Interpolation type '{interpolation}' cannot be recognized. Only 'linear', 'nearest' and 'spline' are supported.")
            _logger.error(error)
            raise error

    _logger.debug(f"Setting SimpleITK interpolation method to '{interpolation}'" + (f" with spline order {splineOrder}." if interpolation is "spline" else "."))
    return interpolator


def get1DInterpolator(x: Iterable[Numberic], y: Iterable[Numberic], interpolation: Literal['linear', 'nearest', 'spline'] = "linear", splineOrder:  Annotated[int, Field(strict=True, ge=0, le=5)] = 3):
    from scipy.interpolate import make_interp_spline
    from functools import partial
    import numpy as np

    def _nearest1Dinterpolator(x, y, xNew):
        x = np.asarray(x)
        y = np.asarray(y)
        xHalfWayPoints = (x[:-1] + x[1:]) / 2.0   # halfway points
        idx = np.searchsorted(xHalfWayPoints, xNew, side='left')
        idx = np.clip(idx, 0, len(x) - 1)  # clip the indices so that they are within the range of x indices.

        return y[idx]

    def nearest1Dinterpolator(x, y):
        return partial(_nearest1Dinterpolator, x, y)

    match interpolation.lower():
        case "linear":
            interpolator = make_interp_spline(x, y, k=1)
        case "nearest":
            interpolator = nearest1Dinterpolator(x, y)
        case "spline":
            if splineOrder > 5 or splineOrder < 0:
                error = ValueError(f"Spline order must be in range 0-5.")
                _logger.error(error)
                raise error
            interpolator = make_interp_spline(x, y, k=splineOrder)
            match splineOrder:
                case 0:
                    _logger.warning("The spline order 0 is not equivalent to 'nearest' but to 'previous' interpolation type.")
                case 1:
                    _logger.info("The spline order 1 is equivalent to 'linear' interpolation type.")
        case _:
            error = ValueError(f"Interpolation type '{interpolation}' cannot be recognized. Only 'linear', 'nearest' and 'spline' are supported.")
            _logger.error(error)
            raise error

    # _logger.debug(f"Setting 1D interpolation method to '{interpolation}'" + (f" with spline order {splineOrder}." if interpolation is "spline" else "."))
    return interpolator


def copyImgMetaData(imgSrc: SITKImage, imgDes: SITKImage) -> SITKImage:
    """Copy meta data to the image source to the image destination"""
    import fredtools as ft

    ft._imgTypeChecker.isSITK(imgSrc, raiseError=True)
    ft._imgTypeChecker.isSITK(imgDes, raiseError=True)
    for key in imgSrc.GetMetaDataKeys():
        imgDes.SetMetaData(key, imgSrc.GetMetaData(key))

    return imgDes


def checkJupyterMode() -> bool:
    """Check if the FREDtools was loaded from jupyter"""
    from IPython.core.getipython import get_ipython
    try:
        ipython = get_ipython()
        if ipython and ipython.config["IPKernelApp"]:
            return True
    except:
        pass
    return False


def checkMatplotlibBackend() -> str:
    """Check the matplotlib backend"""
    import matplotlib

    if "inline" in matplotlib.get_backend():
        return "inline"
    elif "ipympl" in matplotlib.get_backend():
        return "ipympl"
    else:
        return "unknown"


def checkGPUcupy() -> bool:
    """Check if the GPU is available and cupy is working"""

    try:
        import cupy as cp
    except ModuleNotFoundError:
        _logger.debug('Cupy is not installed. Install cupy to use GPU acceleration.')
        return False

    try:
        if cp.cuda.is_available():
            cp.arange(1)
            _logger.debug('Cupy and GPU are available and working.')
            return True
        else:
            _logger.debug('GPU is not available.')
            return False
    except RuntimeError as e:
        _logger.debug("Cupy is available but not working properly. Try checking the CUDA version and install correct cupy version (see more at https://docs.cupy.dev/en/latest/install.html).")
        _logger.debug(e)
        return False
