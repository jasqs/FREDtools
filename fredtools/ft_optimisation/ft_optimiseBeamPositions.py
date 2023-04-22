def optimiseBeamPositions(contourPolygon, spotDistance, algorithm="regular", **kwargs):
    """Calculate and optimize the beam positions in a contour.

    The function calculates optimized beam positions in a contour defined
    as an instance of the shapely.Polygon object. Various optimization
    algorithms are implemented. Refer to 'See Also' section to read more about
    each algorithm.

    Parameters
    ----------
    contourPolygon : shapely Polygon
        Object of the shapely.Polygon.
    spotDistance : scalar
        The nominal spot distance is to be used to optimize the beam positions.
        Depending on the algorithm, the distance between neighboring spots does
        not have to be equal to this parameter. Therefore it describes only
        the nominal distance.
    algorithm :  {'regular', 'hexagonal', 'concentric', 'delaunay'}, optional
        Algorithm to be used to optimize the beam positions. Only 'regular'
        and 'hexagonal' are implemented so far. (def. 'regular')
    **kwargs : keyword args, optional
        Additional parameters are passed to the given optimization algorithm. Refer
        to the given algorithm routine for more description. (def. None)

    Returns
    -------
    numpy array (Nx2)
        Numpy array of size (Nx2) describing the position of N beams, where the first
        column is X and the second Y directions.

    See Also
    --------
        optimiseBeamPositionsRegular: Optimise beam positions in a regular grid.
        optimiseBeamPositionsHexagonal: Optimise beam positions in a hexagonal grid.
    """
    import shapely as sph

    # validate contourPolygon
    if not isinstance(contourPolygon, sph.geometry.Polygon):
        raise TypeError(f"The contour must be an instance of a shapely Polygon class.")

    # validate algorithm
    if not algorithm.lower() in {"regular", "reg", "hexagonal", "hex"}:
        raise ValueError(f"Can not recognize the '{algorithm}' algorithm. Only 'regular' or 'hexagonal' are possible.")

    if algorithm.lower() in ["regular", "reg"]:
        beamPositions = optimiseBeamPositionsRegular(contourPolygon, spotDistance)
    elif algorithm.lower() in ["hexagonal", "hex"]:
        beamPositions = optimiseBeamPositionsHexagonal(contourPolygon, spotDistance, **kwargs)
    elif algorithm.lower() in ["concentric", "con"]:
        beamPositions = optimiseBeamPositionsConcentric(contourPolygon, spotDistance, **kwargs)
    elif algorithm.lower() in ["delaunay", "del"]:
        beamPositions = optimiseBeamPositionsDelaunay(contourPolygon, spotDistance, **kwargs)

    return beamPositions


def optimiseBeamPositionsRegular(contourPolygon, spotDistance):
    """Calculate the beam positions using regular grid algorithm.

    The function calculates beam positions in a contour defined
    as an instance of the shapely.Polygon object using
    regular grid algorithm. The algorithm is optimized to place
    the central beam position at the polygon centroid.

    Parameters
    ----------
    contourPolygon : shapely Polygon
        Object of the shapely.Polygon.
    spotDistance : scalar
        The spot distance to be used to calculate regular grid beam positions.

    Returns
    -------
    numpy array (Nx2)
        Numpy array of size (Nx2) describing the position of N beams, where the first
        column is X and the second Y directions.

    See Also
    --------
        optimiseBeamPositions: Optimise beam positions using various algorithms.

    Notes
    -----
    The regular grid algorithm distributes the beams with the same spacing in X and
    Y directions. The grid size is calculated to fit the given contour polygon and is
    moved so that the central beam is at the polygon centroid. All the beam positions
    which are not inside the polygon are removed.
    """
    import numpy as np
    import shapely as sph

    # construct regular grid with spotDistance step, size of the Polygon bounds and point in the Polygon centroid
    Xmin, Ymin, Xmax, Ymax = contourPolygon.bounds
    XdistPos = np.abs(contourPolygon.centroid.x - Xmax)
    XdistNeg = np.abs(contourPolygon.centroid.x - Xmin)
    YdistPos = np.abs(contourPolygon.centroid.y - Ymax)
    YdistNeg = np.abs(contourPolygon.centroid.y - Ymin)

    Xneg = np.arange(-np.ceil(XdistNeg / spotDistance) - 2, 0) * spotDistance
    Xpos = np.arange(1, np.ceil(XdistPos / spotDistance) + 3) * spotDistance
    Yneg = np.arange(-np.ceil(YdistNeg / spotDistance) - 2, 0) * spotDistance
    Ypos = np.arange(1, np.ceil(YdistPos / spotDistance) + 3) * spotDistance
    x = np.concatenate([Xneg, [0], Xpos]) + contourPolygon.centroid.x
    y = np.concatenate([Yneg, [0], Ypos]) + contourPolygon.centroid.y

    grid = np.stack(np.meshgrid(x, y, sparse=False, indexing="xy"), axis=1).astype("float")
    grid = np.stack([grid[:, 0].flatten(), grid[:, 1].flatten()]).T

    beamPositionsMultiPoint = sph.geometry.MultiPoint(grid)

    # keep only points inside the contour polygon
    beamPositionsMultiPoint = beamPositionsMultiPoint.intersection(contourPolygon.buffer(0.2))

    # convert multipoint to numpy array
    if isinstance(beamPositionsMultiPoint, sph.MultiPoint):
        beamPosition = np.array([[beamPositions.x, beamPositions.y] for beamPositions in beamPositionsMultiPoint.geoms])
    else:
        beamPosition = np.array([[beamPositionsMultiPoint.x, beamPositionsMultiPoint.y]])

    return beamPosition


def optimiseBeamPositionsHexagonal(contourPolygon, spotDistance, direction="X"):
    """Calculate the beam positions using the hexagonal grid algorithm.

    The function calculates beam positions in a contour defined
    as an instance of the shapely.Polygon object using
    hexagonal grid algorithm. The algorithm is optimized to place
    the central beam position at the polygon centroid.

    Parameters
    ----------
    contourPolygon : shapely Polygon
        Object of the shapely.Polygon.
    spotDistance : scalar
        The spot distance is to be used to calculate regular grid beam positions.
    direction : {'X', 'Y'}, optional
        The direction along which the beams should be shifted to create a hexagonal
        grid. This parameter can be used to align the hexagonal direction to the faster
        direction of the pencil beam scanning. (def. 'X')

    Returns
    -------
    numpy array (Nx2)
        Numpy array of size (Nx2) describing the position of N beams, where the first
        column is X and the second Y directions.

    See Also
    --------
        optimiseBeamPositions: Optimise beam positions using various algorithms.

    Notes
    -----
    The hexagonal grid algorithm distributes the beams with the same spacing in X
    (or in Y) and every second row (or column) of the beam positions is shifted by
    half of the `spotDistance`. The grid size is calculated to fit the given
    contour polygon and is moved so that the central beam is at the polygon centroid.
    All the beam positions which are not inside the polygon are removed.

    The user can choose in which direction, X or Y, the hexagonal grid should be aligned.
    This might be important when optimizing the beam positions for a given machine where
    the scanning is faster in one direction than in the other.
    """
    import numpy as np
    import shapely as sph

    # construct regular grid with spotDistance step, size of the Polygon bounds and point in the Polygon centroid
    Xmin, Ymin, Xmax, Ymax = contourPolygon.bounds
    XdistPos = np.abs(contourPolygon.centroid.x - Xmax)
    XdistNeg = np.abs(contourPolygon.centroid.x - Xmin)
    YdistPos = np.abs(contourPolygon.centroid.y - Ymax)
    YdistNeg = np.abs(contourPolygon.centroid.y - Ymin)

    Xneg = np.arange(-np.ceil(XdistNeg / spotDistance) - 2, 0) * spotDistance
    Xpos = np.arange(1, np.ceil(XdistPos / spotDistance) + 2) * spotDistance
    Yneg = np.arange(-np.ceil(YdistNeg / spotDistance) - 3, 0) * spotDistance
    Ypos = np.arange(1, np.ceil(YdistPos / spotDistance) + 3) * spotDistance

    if direction.lower() == "x":
        Yneg *= np.sqrt(3) / 2
        Ypos *= np.sqrt(3) / 2
    elif direction.lower() == "y":
        Xneg *= np.sqrt(3) / 2
        Xpos *= np.sqrt(3) / 2
    else:
        raise ValueError("The 'direction' parameter must be 'X' or 'Y'.")

    x = np.concatenate([Xneg, [0], Xpos]) + contourPolygon.centroid.x
    y = np.concatenate([Yneg, [0], Ypos]) + contourPolygon.centroid.y
    grid = np.stack(np.meshgrid(x, y, sparse=False, indexing="xy"), axis=1).astype("float")
    # move every second line (make hexagonal from regular grid)
    if direction.lower() == "x":
        grid[(len(Yneg) - 1) :: -2, 0, :] += spotDistance / 2
        grid[(len(Yneg) + 1) :: 2, 0, :] += spotDistance / 2
    elif direction.lower() == "y":
        grid[:, 1, (len(Xneg) - 1) :: -2] += spotDistance / 2
        grid[:, 1, (len(Xneg) + 1) :: 2] += spotDistance / 2
    else:
        raise ValueError("The 'direction' parameter must be 'X' or 'Y'.")
    grid = np.stack([grid[:, 0].flatten(), grid[:, 1].flatten()]).T

    beamPositionsMultiPoint = sph.geometry.MultiPoint(grid)

    # keep only points inside the contour polygon
    beamPositionsMultiPoint = beamPositionsMultiPoint.intersection(contourPolygon.buffer(0.2))

    # convert multipoint to numpy array
    if isinstance(beamPositionsMultiPoint, sph.MultiPoint):
        beamPosition = np.array([[beamPositions.x, beamPositions.y] for beamPositions in beamPositionsMultiPoint.geoms])
    else:
        beamPosition = np.array([[beamPositionsMultiPoint.x, beamPositionsMultiPoint.y]])

    return beamPosition


def optimiseBeamPositionsConcentric(contourPolygon, spotDistance):
    """Calculate the beam positions using the hexagonal grid algorithm.

    The function calculates beam positions in a contour defined
    as an instance of the shapely.Polygon object using
    hexagonal grid algorithm. The algorithm is optimized to place
    the central beam position at the polygon centroid.

    Parameters
    ----------
    contourPolygon : shapely Polygon
        Object of the shapely.Polygon.
    spotDistance : scalar
        The spot distance is to be used to calculate regular grid beam positions.
    direction : {'X', 'Y'}, optional
        The direction along which the beams should be shifted to create a hexagonal
        grid. This parameter can be used to align the hexagonal direction to the faster
        direction of the pencil beam scanning. (def. 'X')

    Returns
    -------
    numpy array (Nx2)
        Numpy array of size (Nx2) describing the position of N beams, where the first
        column is X and the second Y directions.

    See Also
    --------
        optimiseBeamPositions: Optimise beam positions using various algorithms.

    Notes
    -----
    The hexagonal grid algorithm distributes the beams with the same spacing in X
    (or in Y) and every second row (or column) of the beam positions is shifted by
    half of the `spotDistance`. The grid size is calculated to fit the given
    contour polygon and is moved so that the central beam is at the polygon centroid.
    All the beam positions which are not inside the polygon are removed.

    The user can choose in which direction, X or Y, the hexagonal grid should be aligned.
    This might be important when optimizing the beam positions for a given machine where
    the scanning is faster in one direction than in the other.
    """
    raise NotImplementedError("The method is not yet implemented")


def optimiseBeamPositionsDelaunay(contourPolygon, spotDistance):
    """Calculate the beam positions using the hexagonal grid algorithm.

    The function calculates beam positions in a contour defined
    as an instance of the shapely.Polygon object using
    hexagonal grid algorithm. The algorithm is optimized to place
    the central beam position at the polygon centroid.

    Parameters
    ----------
    contourPolygon : shapely Polygon
        Object of the shapely.Polygon.
    spotDistance : scalar
        The spot distance is to be used to calculate regular grid beam positions.
    direction : {'X', 'Y'}, optional
        The direction along which the beams should be shifted to create a hexagonal
        grid. This parameter can be used to align the hexagonal direction to the faster
        direction of the pencil beam scanning. (def. 'X')

    Returns
    -------
    numpy array (Nx2)
        Numpy array of size (Nx2) describing the position of N beams, where the first
        column is X and the second Y directions.

    See Also
    --------
        optimiseBeamPositions: Optimise beam positions using various algorithms.

    Notes
    -----
    The hexagonal grid algorithm distributes the beams with the same spacing in X
    (or in Y) and every second row (or column) of the beam positions is shifted by
    half of the `spotDistance`. The grid size is calculated to fit the given
    contour polygon and is moved so that the central beam is at the polygon centroid.
    All the beam positions which are not inside the polygon are removed.

    The user can choose in which direction, X or Y, the hexagonal grid should be aligned.
    This might be important when optimizing the beam positions for a given machine where
    the scanning is faster in one direction than in the other.
    """
    raise NotImplementedError("The method is not yet implemented")
