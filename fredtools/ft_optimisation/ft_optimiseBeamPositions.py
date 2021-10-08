def optimiseBeamPositions(contourPolygon, spotDistance, algorithm="regular"):
    """Calculate and optimise the beam positions in a contour.

    The function calculates optimized beam positions in a contour defined
    as an instance of the shapely.geometry.Polygon object. Various optimization
    algorithms are implemented. Refer to 'See Also' section to read more about
    each algorithm.

    Parameters
    ----------
    contourPolygon : shapely Polygon
        Object of the shapely.geometry.Polygon.
    spotDistance : scalar
        The nominal spot distance to be used to optimize the beam positions.
        Depending on the algorithm, the distance between neighboring spots does
        not have to be equal to this parameter. Therefore it describes only
        the nominal distance.
    algorithm :  {'regular', 'nearest', 'spline'}, optional
        Algorithm to be used to optimise the beam positions. (def. 'regular')

    Returns
    -------
    numpy array (Nx2)
        Numpy array of size (Nx2) describing position of N beams, where the first
        column is X and the second Y directions.

    See Also
    --------
        optimiseBeamPositionsRegular: Optimise beam positions in a regular grid.
    """
    import shapely as sph

    # validate contourPolygon
    if not isinstance(contourPolygon, sph.geometry.Polygon):
        raise TypeError(f"The contour must be an instance of a shapely Polygon class.")
    # validate algorithm
    if not algorithm.lower() in {"regular", "hexagonal"}:
        raise ValueError(f"Can not recognise the '{algorithm}' algorithm. Only 'regular' or 'hexagonal' are possible.")

    if algorithm.lower() == "regular":
        beamPositions = optimiseBeamPositionsRegular(contourPolygon, spotDistance)
    if algorithm.lower() == "hexagonal":
        beamPositions = optimiseBeamPositionsHexagonal(contourPolygon, spotDistance)

    return beamPositions


def optimiseBeamPositionsRegular(contourPolygon, spotDistance):
    """Calculate the beam positions using regular grid algorithm.

    The function calculates beam positions in a contour defined
    as an instance of the shapely.geometry.Polygon object using
    regular grid algorithm.

    Parameters
    ----------
    contourPolygon : shapely Polygon
        Object of the shapely.geometry.Polygon.
    spotDistance : scalar
        The spot distance to be used to calculate regular grid beam positions.

    Returns
    -------
    numpy array (Nx2)
        Numpy array of size (Nx2) describing position of N beams, where the first
        column is X and the second Y directions.

    See Also
    --------
        optimiseBeamPositions: Optimise beam positions using various algorithms.

    Notes
    -----
    The regular grid algorithm distributes the beams with the same spacing in X and
    Y directions. The grid size is calculated in this way to fit the given contour
    polygon and is moved to the envelope center of the polygon. All the beam positions
    which are not inside the polygon are removed.
    """
    import numpy as np
    import shapely as sph

    # construct regular grid with spotDistance step and size of the contour bounds
    contourSize = np.array([contourPolygon.bounds[2] - contourPolygon.bounds[0], contourPolygon.bounds[3] - contourPolygon.bounds[1]])
    gridSize = np.ceil(contourSize / spotDistance)
    grid = np.stack(np.meshgrid(np.arange(gridSize[0]), np.arange(gridSize[1]), sparse=False, indexing="xy"), axis=1).astype("float")
    grid = np.stack([grid[:, 0].flatten(), grid[:, 1].flatten()]).T
    grid *= spotDistance

    # move the grid envelope centroid to the contour polygon envelope centroid
    beamPositionsMultiPoint = sph.geometry.MultiPoint(grid)
    beamPositions = grid + (np.array(contourPolygon.envelope.centroid.xy) - np.array(beamPositionsMultiPoint.envelope.centroid.xy)).T

    # keep only points inside the contour polygon
    beamPositionsMultiPoint = sph.geometry.MultiPoint(beamPositions)
    beamPositionsInsideMultiPoint = []
    for beamPositionPoint in beamPositionsMultiPoint:
        if contourPolygon.contains(beamPositionPoint):
            beamPositionsInsideMultiPoint.append(beamPositionPoint)
    beamPositionsInsideMultiPoint = sph.geometry.MultiPoint(beamPositionsInsideMultiPoint)

    # convert multipoint to numpy array
    beamPosition = []
    for beamPositionsInside in beamPositionsInsideMultiPoint:
        beamPosition.append([beamPositionsInside.x, beamPositionsInside.y])
    beamPosition = np.array(beamPosition)

    return beamPosition


def optimiseBeamPositionsHexagonal(contourPolygon, spotDistance):
    """Calculate the beam positions using regular hexagonal algorithm.

    The function calculates beam positions in a contour defined
    as an instance of the shapely.geometry.Polygon object using
    hexagonal grid algorithm.

    Parameters
    ----------
    contourPolygon : shapely Polygon
        Object of the shapely.geometry.Polygon.
    spotDistance : scalar
        The spot distance to be used to calculate regular grid beam positions.

    Returns
    -------
    numpy array (Nx2)
        Numpy array of size (Nx2) describing position of N beams, where the first
        column is X and the second Y directions.

    See Also
    --------
        optimiseBeamPositions: Optimise beam positions using various algorithms.

    Notes
    -----
    Not implemented yet.
    """
    import numpy as np
    import shapely as sph

    raise ValueError(f"the function is not implemented yet.")

    return []
