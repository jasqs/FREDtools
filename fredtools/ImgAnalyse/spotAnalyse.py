def _singleGaussModel(pos, amplitude, centre, sigma):
    import numpy as np

    return amplitude * np.exp(-((pos - centre) ** 2) / (2 * sigma ** 2))


def fitSpotProfile(pos, vec, cutLevel=0, fixAmplitude=False, fixCentreToZero=False, method="singleGauss"):
    """Fit a gaussian-like function to profile.

    The function fits a gaussian-like function defined by `method` to a profile
    defined as `pos` and `vec` parameters, where they are vectors of positions and values, respectively.

    Parameters
    ----------
    pos : array_like 1xN
        An iterable with the positions of the values.
    vec : array_like 1xN
        An iterable with the values correcponding to `pos`.
    cutLevel : scalar, optional
        Fraction of the maximum value of `vec` for which the fit will be performed.
    fixAmplitude : bool, optional
        Fix the amplitude to the maximum value of `vec` and do not use it
        in the fitting. (def. False)
    fixCentreToZero : bool, optional
        Fix the centre to zero and do not use it in the fitting. (def. False)
    method : {"singleGauss"}, optional
        Method of the fitting. Only single gaussian fitting is implemented now. (def. "singleGauss")

    Returns
    -------
    lmfit.model.ModelResult
        An instance of lmfit.model.ModelResult class.
    """
    from lmfit import Model
    import numpy as np

    # check if pos and vec are both iterable, are vectors and have the same length
    if not hasattr(pos, "__iter__") and not hasattr(vec, "__iter__"):
        raise TypeError(f"The input `pos` and `vec` must be both iterable.")
    if not np.array(pos).ndim == 1 and np.array(vec).ndim == 1:
        raise TypeError(f"The input `pos` and `vec` must be both one-dimensional vectors.")
    if len(list(pos)) != len(list(vec)):
        raise TypeError(f"The input `pos` and `vec` must be of the same length.")

    prof = [np.array(pos), np.array(vec)]

    # cut data above cutLevel
    cutConst = np.where(prof[1] >= (np.max(prof[1]) * cutLevel))
    prof[0] = prof[0][cutConst]
    prof[1] = prof[1][cutConst]

    if method == "singleGauss":
        ### single gaussian fit to profile
        # calculate initial params
        initAmplitude = np.max(prof[1])
        initCentre = np.mean(prof[0][np.where(prof[1] == initAmplitude)])
        initSigma = np.ptp(prof[0][np.where(prof[1] >= (initAmplitude / 2))[0]] / 2.355)

        if fixCentreToZero:
            initCentre = 0

        gmodel = Model(_singleGaussModel)
        gmodel.set_param_hint("amplitude", vary=not fixAmplitude)
        gmodel.set_param_hint("centre", vary=not fixCentreToZero)
        result = gmodel.fit(data=prof[1], pos=prof[0], amplitude=initAmplitude, centre=initCentre, sigma=initSigma)
        return result
    else:
        raise ValueError(f"The method '{method}' can not be recognized. Only 'singleGauss' is available at the moment.")
