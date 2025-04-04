from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def transformIndexToPhysicalPoint(img: SITKImage, indices: Iterable[PointLike]) -> Tuple[Tuple[float, ...], ...]:
    """Transform indices to physical points.

    The function transforms an iterable of indices into a tuple of
    physical points based on the field of reference (FoR) of an image
    defined as an instance of a SimpleITK image object. The function is
    a wrapper for `TransformIndexToPhysicalPoint` SimpleITK function, but
    it works for multiple points. The shape of `indices` must be NxD, where N
    is the number of points to be converted, and D is the dimension of the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    indices : NxD array_like
        An iterable (numpy array, list of lists, etc) of N points.
        Every index must be of the image dimension size and of any
        integer type (int64, uint16, etc.).

    Returns
    -------
    tuple
        A NxD tuple of tuples with physical points.

    See Also
    --------
        transformContinuousIndexToPhysicalPoint : transform indices to physical points.
        transformPhysicalPointToIndex : transform physical points to indices.
        transformPhysicalPointToContinuousIndex : transform physical points to continuous indices.
    """
    import numpy as np
    import fredtools as ft

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # transform iterable to numpy array
    indices = np.array(indices)

    # correct numpy array in case of single point
    if indices.ndim == 1:
        indices = np.expand_dims(indices, 0)

    # check if type of indices is correct
    if not np.issubdtype(indices.dtype, np.integer):
        error = AttributeError(f"The 'indices' parameter must of any integer type (int64, uint16, etc.).")
        _logger.error(error)
        raise error

    # check if shape of indices is correct
    if indices.ndim != 2 or indices.shape[1] != img.GetDimension():
        error = AttributeError(f"The 'indices' parameter must be an iterable of Nx{img.GetDimension()} shape for {img.GetDimension()}D image.")
        _logger.error(error)
        raise error

    return tuple(map(img.TransformIndexToPhysicalPoint, indices.tolist()))


def transformContinuousIndexToPhysicalPoint(img: SITKImage, indices: Iterable[PointLike]) -> Tuple[Tuple[float, ...], ...]:
    """Transform indices to physical points.

    The function transforms an iterable of indices into a tuple of
    physical points based on the field of reference (FoR) of an image
    defined as an instance of a SimpleITK image object. The function is
    a wrapper for `TransformContinuousIndexToPhysicalPoint` SimpleITK function, but
    it works for multiple points. The shape of `indices` must be NxD, where N
    is the number of points to be converted, and D is the dimension of the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    indices : NxD array_like
        An iterable (numpy array, list of lists, etc) of N points.
        Every index must be of the image dimension size and can be
        of float or integer type.

    Returns
    -------
    tuple
        A NxD tuple of tuples with physical points.

    See Also
    --------
        transformIndexToPhysicalPoint : transform indices to physical points.
        transformPhysicalPointToIndex : transform physical points to indices.
        transformPhysicalPointToContinuousIndex : transform physical points to continuous indices.
    """
    import numpy as np
    import fredtools as ft

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # transform iterable to numpy array
    indices = np.array(indices)
    # correct numpy array in case of single point
    if indices.ndim == 1:
        indices = np.expand_dims(indices, 0)

    # check if shape of indices is correct
    if indices.ndim != 2 or indices.shape[1] != img.GetDimension():
        error = AttributeError(f"The 'indices' parameter must be an iterable of Nx{img.GetDimension()} shape for {img.GetDimension()}D image.")
        _logger.error(error)
        raise error

    return tuple(map(img.TransformContinuousIndexToPhysicalPoint, indices.tolist()))


def transformPhysicalPointToIndex(img: SITKImage, points: Iterable[PointLike]) -> Tuple[Tuple[int, ...], ...]:
    """Transform physical points to indices.

    The function transforms an iterable of points into a tuple of
    indices based on the field of reference (FoR) of an image
    defined as an instance of a SimpleITK image object. The function is
    a wrapper for `TransformPhysicalPointToIndex` SimpleITK function, but
    it works for multiple points. The shape of `points` must be NxD, where N
    is the number of points to be converted, and D is the dimension of the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    points : NxD array_like
        An iterable (numpy array, list of lists, etc) of N points.
        Every point must be of the image dimension size.

    Returns
    -------
    tuple
        A NxD tuple of tuples with indices.

    See Also
    --------
        transformIndexToPhysicalPoint : transform indices to physical points.
        transformContinuousIndexToPhysicalPoint : transform indices to physical points.
        transformPhysicalPointToContinuousIndex : transform physical points to continuous indices.
    """
    import numpy as np
    import fredtools as ft

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # transform iterable to numpy array
    points = np.array(points)
    # correct numpy array in case of single point
    if points.ndim == 1:
        points = np.expand_dims(points, 0)

    # check if shape of points is correct
    if points.ndim != 2 or points.shape[1] != img.GetDimension():
        error = AttributeError(f"The 'points' parameter must be an iterable of Nx{img.GetDimension()} shape for {img.GetDimension()}D image.")
        _logger.error(error)
        raise error

    return tuple(map(img.TransformPhysicalPointToIndex, points.tolist()))


def transformPhysicalPointToContinuousIndex(img: SITKImage, points: Iterable[PointLike]) -> Tuple[Tuple[float, ...], ...]:
    """Transform physical points to continuous indices.

    The function transforms an iterable of points into a tuple of
    continuous indices based on the field of reference (FoR) of an image
    defined as an instance of a SimpleITK image object. The function is
    a wrapper for `TransformPhysicalPointToContinuousIndex` SimpleITK function, but
    it works for multiple points. The shape of `points` must be NxD, where N
    is the number of points to be converted, and D is the dimension of the image.

    Parameters
    ----------
    img : SimpleITK Image
        An object of a SimpleITK image.
    points : NxD array_like
        An iterable (numpy array, list of lists, etc) of N points.
        Every point must be of the image dimension size.

    Returns
    -------
    tuple
        A NxD tuple of tuples with continuous indices.

    See Also
    --------
        transformIndexToPhysicalPoint : transform indices to physical points.
        transformContinuousIndexToPhysicalPoint : transform indices to physical points.
        transformPhysicalPointToIndex : transform physical points to indices.
    """
    import numpy as np
    import fredtools as ft

    ft._imgTypeChecker.isSITK(img, raiseError=True)

    # transform iterable to numpy array
    points = np.array(points)
    # correct numpy array in case of single point
    if points.ndim == 1:
        points = np.expand_dims(points, 0)

    # check if shape of points is correct
    if points.ndim != 2 or points.shape[1] != img.GetDimension():
        raise AttributeError(f"The 'points' parameter must be an iterable of Nx{img.GetDimension()} shape for {img.GetDimension()}D image.")

    return tuple(map(img.TransformPhysicalPointToContinuousIndex, points.tolist()))
