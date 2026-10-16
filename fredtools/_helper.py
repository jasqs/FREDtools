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

    Raises
    ------
    ValueError
        If the interpolation method cannot be recognised or the spline order is not in range 0-5.
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
            error = ValueError(f"Interpolation type '{interpolation}' cannot be recognised. Only 'linear', 'nearest' and 'spline' are supported.")
            _logger.error(error)
            raise error

    _logger.debug(f"Setting SimpleITK interpolation method to '{interpolation}'" + (f" with spline order {splineOrder}." if interpolation.lower() == "spline" else "."))
    return interpolator


def getSITKPixelID(dtype: DTypeLike, vector: bool = False) -> int:
    """Get SimpleITK pixel ID for a data type.

    The function converts a data type, given in any form accepted
    by `numpy.dtype`, to a SimpleITK pixel ID of a scalar or vector image.

    Parameters
    ----------
    dtype : data-type
        The data type in any form accepted by `numpy.dtype`, e.g. 'float32',
        'f4', numpy.int16 or float. The supported types are (u)int8, (u)int16,
        (u)int32, (u)int64, float32 and float64. The type bool is converted to uint8,
        because SimpleITK does not have a boolean pixel type.
    vector : bool, optional
        If True, the pixel ID of a vector image is returned. (def. False)

    Returns
    -------
    int
        SimpleITK pixel ID, e.g. sitk.sitkFloat32 or sitk.sitkVectorFloat32.

    Raises
    ------
    TypeError
        If `dtype` is None, cannot be interpreted as a numpy data type
        or is not supported.
    """
    import numpy as np
    import SimpleITK as sitk

    # numpy interprets None as float64, which might be misleading
    if dtype is None:
        error = TypeError("The data type cannot be None.")
        _logger.error(error)
        raise error

    # SimpleITK images always use the native byte order, so e.g. '>f4' is converted to float32
    try:
        dtypeNP = np.dtype(dtype).newbyteorder("=")
    except TypeError:
        error = TypeError(f"The data type '{dtype}' cannot be interpreted as a numpy data type.")
        _logger.error(error)
        raise error

    # SimpleITK has no boolean pixel type, a boolean image is a uint8 image with 0/1 values
    if dtypeNP == np.bool_:
        dtypeNP = np.dtype(np.uint8)

    # mapping of numpy data types to SimpleITK pixel IDs of scalar and vector images
    pixelIDs = {
        np.dtype(np.uint8): (sitk.sitkUInt8, sitk.sitkVectorUInt8),
        np.dtype(np.uint16): (sitk.sitkUInt16, sitk.sitkVectorUInt16),
        np.dtype(np.uint32): (sitk.sitkUInt32, sitk.sitkVectorUInt32),
        np.dtype(np.uint64): (sitk.sitkUInt64, sitk.sitkVectorUInt64),
        np.dtype(np.int8): (sitk.sitkInt8, sitk.sitkVectorInt8),
        np.dtype(np.int16): (sitk.sitkInt16, sitk.sitkVectorInt16),
        np.dtype(np.int32): (sitk.sitkInt32, sitk.sitkVectorInt32),
        np.dtype(np.int64): (sitk.sitkInt64, sitk.sitkVectorInt64),
        np.dtype(np.float32): (sitk.sitkFloat32, sitk.sitkVectorFloat32),
        np.dtype(np.float64): (sitk.sitkFloat64, sitk.sitkVectorFloat64),
    }

    if dtypeNP not in pixelIDs:
        error = TypeError(f"The data type '{dtypeNP}' is not supported. Only {', '.join([str(dtypeSupported) for dtypeSupported in pixelIDs])} and bool are supported.")
        _logger.error(error)
        raise error

    pixelID = pixelIDs[dtypeNP][1 if vector else 0]

    _logger.debug(f"Data type '{dtypeNP}' converted to SimpleITK pixel type '{sitk.GetPixelIDValueAsString(pixelID)}'.")
    return pixelID


def get1DInterpolator(x: Iterable[Numeric], y: Iterable[Numeric], interpolation: Literal['linear', 'nearest', 'spline'] = "linear", splineOrder:  Annotated[int, Field(strict=True, ge=0, le=5)] = 3):
    """Get a 1D interpolator for data points.

    The function creates a one-dimensional interpolator for the data points
    given as `x` and `y` values, using the requested interpolation method.
    The 'linear' and 'spline' interpolations are implemented with
    scipy.interpolate.make_interp_spline, whereas the 'nearest' interpolation
    is implemented locally: the nearest neighbour is selected based on the
    half-way points between consecutive `x` values, and the indices are clipped
    at the edges so that positions outside the `x` range map to the first or
    last data point. Note that a spline of order 0 is not equivalent to the
    'nearest' interpolation but to the 'previous' interpolation type.

    Parameters
    ----------
    x : array_like
        `X` values of the data points.
    y : array_like
        `Y` values of the data points.
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)

    Returns
    -------
    callable
        A callable interpolator that takes the position (or positions) to
        interpolate at and returns the interpolated value (or values).

    Raises
    ------
    ValueError
        If the interpolation method cannot be recognised or the spline order is not in range 0-5.
    """
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
            error = ValueError(f"Interpolation type '{interpolation}' cannot be recognised. Only 'linear', 'nearest' and 'spline' are supported.")
            _logger.error(error)
            raise error

    # _logger.debug(f"Setting 1D interpolation method to '{interpolation}'" + (f" with spline order {splineOrder}." if interpolation.lower() == "spline" else "."))
    return interpolator


def copyImgMetaData(imgSrc: SITKImage, imgDes: SITKImage) -> SITKImage:
    """Copy meta data to the image source to the image destination.

    Raises
    ------
    TypeError
        If `imgSrc` or `imgDes` is not an instance of a SimpleITK image object.
    """
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
