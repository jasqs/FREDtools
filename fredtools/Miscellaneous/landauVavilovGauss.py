from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def pdfLandau(x: Numberic | Iterable[Numberic], mpv: Numberic, xi: Numberic, amp: Numberic = 1) -> Numberic | Iterable[Numberic]:
    """Landau probability density function (PDF).

    The function generates a Landau probability density with a given most probable
    value (`mpv`), width (described with `xi`) and amplitude at `mpv`. It was adapted
    from 3 which was implemented based on the ROOT implementation. See [1]_ and [2]_ for more details.

    Parameters
    ----------
    x : scalar or array_like
        Point (or points) where to calculate the PDF.
    mpv : scalar
        Position of the most probable value (MPV) of the Landau distribution.
    xi : float
        Parameter 'xi' of the Landau distribution, it is a measure of its width.
    amp : scalar, optional
        Amplitude of the PDF at MPV. (def. 1)

    Returns
    -------
    scalar or numpy array
        Single value or array of values of the Landau PDF.

    See Also
    --------
    fitLandau : fit Landau distribution to data.

    References
    ----------
    .. [1] `landaupy python package <https://pypi.org/project/landaupy/>`_
    .. [2] `landaupy package documentation <https://github.com/SengerM/landaupy>`_
    """
    from landaupy import landau
    import numpy as np

    # check para3eters
    if not isinstance(mpv, Numberic):
        error = TypeError(f"The 'mpv' parameter must be a scalar but it is {type(mpv)}")
        _logger.error(error)
        raise error
    if not isinstance(xi, Numberic):
        error = TypeError(f"The 'xi' parameter must be a scalar but it is {type(xi)}")
        _logger.error(error)
        raise error
    if not isinstance(amp, Numberic):
        error = TypeError(f"The 'amp' parameter must be a scalar but it is {type(amp)}")
        _logger.error(error)
        raise error

    if not (isinstance(xi, Numberic) and 0 < xi):
        error = ValueError("The 'xi' parameter must be a scalar and xi > 0.")
        _logger.error(error)
        raise error
    if not (isinstance(amp, Numberic) and 0 <= amp):
        error = ValueError("The 'amp' parameter must be a scalar and amp >= 0.")
        _logger.error(error)
        raise error

    return amp * landau.pdf(x, x_mpv=mpv, xi=xi) / landau.pdf(mpv, x_mpv=mpv, xi=xi)


def pdfLandauGauss(x: Numberic | Iterable[Numberic], mpv: Numberic, xi: Numberic, sigma: Numberic = 0, amp: Numberic = 1) -> Numberic | Iterable[Numberic]:
    """Probability density function (PDF) of Landau convoluted with a Gaussian.

    The function generates a Landau convoluted with a Gaussian probability density with a given
    most probable value of the convoluted function (`mpv`), the width of Landau (described with `xi`),
    the standard deviation of Gaussian and amplitude at `mpv`. It was adapted from [3]_ which was implemented
    based on the ROOT implementation. See [4]_ for more details.

    Parameters
    ----------
    x : scalar or array_like
        Point (or points) where to calculate the PDF.
    mpv : scalar
        Position of the most probable value (MPV) of the convoluted distribution.
    xi : float
        Parameter 'xi' of the Landau distribution, it is a measure of its width.
    sigma : scalar, optional
        Standard deviation of the Gaussian distribution. (def. 0)
    amp : scalar, optional
        Amplitude of the PDF at MPV. (def. 1)

    Returns
    -------
    scalar or numpy array
        Single value or array of values of the Landau convoluted with Gaussian PDF.

    See Also
    --------
    fitLandauGauss : fit Landau distribution convoluted with a Gaussian to data.

    Notes
    -----
    The 'mpv' parameter does not describe the MPV of the landau distribution but the MPV,
    i.e. the position of the maximum value, of the whole Landau-gauss convoluted PDF.

    References
    ----------
    .. [3] `landaupy python package <https://pypi.org/project/landaupy/>`_
    .. [4] `landaupy package documentation <https://github.com/SengerM/landaupy>`_
    """
    from landaupy import langauss
    import numpy as np

    # check parameters
    if not isinstance(mpv, Numberic):
        error = TypeError(f"The 'mpv' parameter must be a scalar but it is {type(mpv)}")
        _logger.error(error)
        raise error
    if not isinstance(xi, Numberic):
        error = TypeError(f"The 'xi' parameter must be a scalar but it is {type(xi)}")
        _logger.error(error)
        raise error
    if not isinstance(amp, Numberic):
        error = TypeError(f"The 'amp' parameter must be a scalar but it is {type(amp)}")
        _logger.error(error)
        raise error
    if not isinstance(sigma, Numberic):
        error = TypeError(f"The 'sigma' parameter must be a scalar but it is {type(sigma)}")
        _logger.error(error)
        raise error

    if not (isinstance(xi, Numberic) and 0 < xi):
        error = ValueError("The 'xi' parameter must be a scalar and xi > 0.")
        _logger.error(error)
        raise error
    if not (isinstance(sigma, Numberic) and 0 <= sigma):
        error = ValueError("The 'sigma' parameter must be a scalar and sigma >= 0.")
        _logger.error(error)
        raise error
    if not (isinstance(amp, Numberic) and 0 <= amp):
        error = ValueError("The 'amp' parameter must be a scalar and amp >= 0.")
        _logger.error(error)
        raise error

    x = np.array(x)
    xInternal = x.copy()

    # move x position to the expected mpv
    mpvInternal = _getMPV(xInternal, langauss.pdf(xInternal, landau_x_mpv=float(mpv), landau_xi=float(xi), gauss_sigma=float(sigma)))
    xInternal += mpvInternal[0] - mpv

    # normalize PDF to the amplitude
    yInternal = langauss.pdf(xInternal, landau_x_mpv=float(mpv), landau_xi=float(xi), gauss_sigma=float(sigma))
    yInternal /= mpvInternal[1]
    yInternal *= amp

    return yInternal


def fitLandau(x: Iterable[Numberic], y: Iterable[Numberic], fixAmplitude: bool = False) -> LMFitModelResult:
    """Fit Landau distribution.

    The function fits Landau distribution to the data given as `x` and `y` values,
    using the least square algorithm.

    Parameters
    ----------
    x : array_like
        `X` values.
    y : array_like
        `Y` values.
    fixAmplitude : bool, optional
        determine if the `amp` parameter of the PDF should be used in the fitting.

    Returns
    -------
    lmfit.model.ModelResult
        Model results of the LMFit package.

    See Also
    --------
    fitLandauGauss : fit Landau distribution convoluted with a Gaussian to data.
    fitVavilov : fit Vavilov distribution to data.
    """
    import lmfit
    import numpy as np

    fitModel = lmfit.Model(pdfLandau)

    # calculate starting parameters
    x, y = np.array(x), np.array(y)
    amp0 = np.nanmax(y)
    mpv0 = x[np.where(np.array(y) == amp0)[0]][0]
    xi0 = np.sqrt(np.cov(x, aweights=y)) * 0.3

    # prepare constraints for the parameters
    fitModel.set_param_hint("mpv", min=0, max=np.inf, value=mpv0, vary=True)
    fitModel.set_param_hint("amp", min=0, max=np.inf, value=amp0, vary=not fixAmplitude)
    fitModel.set_param_hint("xi", min=1e-5, max=np.inf, value=xi0, vary=True)

    # perform fit
    fitResult = fitModel.fit(data=y, x=x)

    return fitResult


def fitLandauGauss(x: Iterable[Numberic], y: Iterable[Numberic], fixAmplitude: bool = False) -> LMFitModelResult:
    """Fit Landau convoluted with Gaussian distribution.

    The function fits Landau convoluted with Gaussian distribution
    to the data given as `x` and `y` values, using the least square algorithm.

    Parameters
    ----------
    x : array_like
        `X` values.
    y : array_like
        `Y` values.
    fixAmplitude : bool, optional
        determine if the `amp` parameter of the PDF should be used in the fitting.

    Returns
    -------
    lmfit.model.ModelResult
        Model results of the LMFit package.

    See Also
    --------
    fitLandau : fit Landau distribution to data.
    fitVavilov : fit Vavilov distribution to data.
    """
    import lmfit
    import numpy as np

    fitModel = lmfit.Model(pdfLandauGauss)

    # calculate starting parameters
    x, y = np.array(x), np.array(y)
    amp0 = np.nanmax(y)
    mpv0 = x[np.where(np.array(y) == amp0)[0]][0]
    xi0 = np.sqrt(np.cov(x, aweights=y)) * 0.3
    sigma0 = 0.1

    # prepare constraints for the parameters
    fitModel.set_param_hint("mpv", min=0, max=np.inf, value=mpv0, vary=True)
    fitModel.set_param_hint("amp", min=0, max=np.inf, value=amp0, vary=not fixAmplitude)
    fitModel.set_param_hint("xi", min=1e-5, max=np.inf, value=xi0, vary=True)
    fitModel.set_param_hint("sigma", min=0, max=np.inf, value=sigma0, vary=True)

    # perform fit
    fitResult = fitModel.fit(data=y, x=x)

    return fitResult


def pdfVavilov(x: Numberic | Iterable[Numberic], mpv: Numberic, kappa: Numberic, beta: Numberic, scaling: Numberic, amp: Numberic = 1) -> Numberic | Iterable[Numberic]:
    """Probability density function (PDF) of Vavilov.

    The function generates a Vavilov probability density with a given
    most probable value function (`mpv`), amplitude (`amp`), as well as `kappa`,
    `beta` and `scaling` parameters. It uses the implementation of pyamtrack library [5]_
    that adopts the ROOT implementation [6]_. The implemented PDF is not a true Vavilov distribution
    and the `scaling` parameter is not included in the original ROOT implementation. Therefore, the parameters
    `kappa` and `beta` might not describe the real kappa and beta parameters of the ROOT Vavilov.
    Nevertheless, the PDF can be used for fitting the distribution to the measurement data
    and to retrieve the MPV but the user must be aware that, for instance, the energy calculated based on
    the `beta` parameter might be wrong.

    Parameters
    ----------
    x : scalar or array_like
        Point (or points) where to calculate the PDF.
    mpv : scalar
        Position of the most probable value (MPV) of the distribution.
    kappa : float
        Parameter 'kappa' of the Vavilov distribution.
    beta : float
        Parameter 'beta' of the Vavilov distribution.
    scaling : float
        Scaling factor of the distribution.
    amp : scalar, optional
        Amplitude of the PDF at MPV. (def. 1)

    Returns
    -------
    scalar or numpy array
        Single value or array of values of the Vavilov PDF.

    See Also
    --------
    fitVavilov : fit Vavilov distribution to data.

    References
    ----------
    .. [5] `pyamtrack python package <https://github.com/libamtrack/pyamtrack>`_
    .. [6] `ROOT Vavilov class reference <https://root.cern/doc/master/classROOT_1_1Math_1_1Vavilov.html>`_
    """
    from pyamtrack.libAT import AT_Vavilov_PDF
    import numpy as np

    # check parameters
    if not isinstance(mpv, Numberic):
        error = TypeError(f"The 'mpv' parameter must be a scalar but it is {type(mpv)}")
        _logger.error(error)
        raise error
    if not isinstance(kappa, Numberic):
        error = TypeError(f"The 'kappa' parameter must be a scalar but it is {type(kappa)}")
        _logger.error(error)
        raise error
    if not isinstance(beta, Numberic):
        error = TypeError(f"The 'beta' parameter must be a scalar but it is {type(beta)}")
        _logger.error(error)
        raise error
    if not isinstance(amp, Numberic):
        error = TypeError(f"The 'amp' parameter must be a scalar but it is {type(amp)}")
        _logger.error(error)
        raise error

    if not (isinstance(kappa, Numberic) and 0.01 <= kappa <= 12):
        error = ValueError("The 'kappa' parameter must be a scalar in range 0.01 <= kappa <= 12.")
        _logger.error(error)
        raise error
    if not (isinstance(beta, Numberic) and 0 <= beta <= 1):
        error = ValueError("The 'beta' parameter must be a scalar in range 0 <= beta <= 1.")
        _logger.error(error)
        raise error
    if not (isinstance(amp, Numberic) and 0 <= amp):
        error = ValueError("The 'amp' parameter must be a scalar amp >= 0.")
        _logger.error(error)
        raise error
    if not (isinstance(scaling, Numberic) and 0 <= scaling):
        error = ValueError("The 'scaling' parameter must be a scalar scaling >= 0.")
        _logger.error(error)
        raise error

    x = np.array(x)
    xInternal = x.copy()
    xInternal = np.asarray(xInternal, dtype=float)
    yInternal = np.zeros(xInternal.size)

    # move x position to the expected mpv
    xMPVCalc = np.linspace(-10, 10, 1000)
    yMPVCalc = np.zeros(xMPVCalc.size)
    AT_Vavilov_PDF(xMPVCalc.tolist(), p_kappa=kappa, p_beta=beta, p_density=yMPVCalc)
    mpvInternal = _getMPV(xMPVCalc, yMPVCalc)
    xInternal /= scaling
    xInternal += mpvInternal[0]
    xInternal -= mpv / scaling

    AT_Vavilov_PDF(xInternal.tolist(), p_kappa=kappa, p_beta=beta, p_density=yInternal)

    # normalize PDF to the amplitude
    yInternal /= mpvInternal[1]
    yInternal *= amp

    return yInternal


def fitVavilov(x: Iterable[Numberic], y: Iterable[Numberic], beta0: Numberic = 0.5, kappa0: Numberic = 0.3, scaling0: Numberic = -1, fixAmplitude: bool = False) -> LMFitModelResult:
    """Fit Vavilov distribution.

    The function fits the Vavilov distribution to the data given as `x` and `y` values,
    using the least square algorithm. The fitting routine is sensitive to the initial
    values of kappa, beta and scaling. Therefore, the results should be always validated
    and different initial values of the parameters can be used if needed.

    Parameters
    ----------
    x : array_like
        `X` values.
    y : array_like
        `Y` values.
    fixAmplitude : bool, optional
        determine if the `amp` parameter of the PDF should be used in the fitting.
    beta0 : scalar, optional
        Initial value of `beta` parameter. (def. 0.5)
    kappa0 : scalar, optional
        Initial value of `kappa` parameter. (def. 0.3)
    scaling0 : scalar, optional
        Initial value of `scaling` parameter. If it is less than 0 then
        it is calculated based on the standard deviation of the distribution. (def. -1)

    Returns
    -------
    lmfit.model.ModelResult
        Model results of the LMFit package.

    See Also
    --------
    fitLandau : fit Landau distribution to data.
    fitLandauGauss : fit Landau distribution convoluted with a Gaussian to data.
    """
    import lmfit
    import numpy as np

    fitModel = lmfit.Model(pdfVavilov)

    # calculate starting parameters
    x, y = np.array(x), np.array(y)
    amp0 = np.nanmax(y)
    mpv0 = x[np.where(np.array(y) == amp0)[0]][0]
    if scaling0 < 0:
        scaling0 = float(np.sqrt(np.cov(x, aweights=y)) * 0.3)

    # prepare constraints for the parameters
    fitModel.set_param_hint("mpv", min=0, max=np.inf, value=mpv0, vary=True)
    fitModel.set_param_hint("amp", min=0, max=np.inf, value=amp0, vary=not fixAmplitude)
    fitModel.set_param_hint("kappa", min=0.01, max=12, value=kappa0, vary=True)
    fitModel.set_param_hint("scaling", min=0.0, max=np.inf, value=scaling0, vary=True)
    fitModel.set_param_hint("beta", min=0, max=1, value=beta0, vary=True)

    # perform fit
    fitResult = fitModel.fit(data=y, x=x)

    return fitResult


def _getMPV(x: Iterable[Numberic], y: Iterable[Numberic]) -> tuple[Numberic, Numberic]:
    # calculate MPV and the maximum value
    from scipy.interpolate import InterpolatedUnivariateSpline
    import numpy as np
    x, y = np.array(x), np.array(y)
    interpFun = InterpolatedUnivariateSpline(x, y, k=4)
    cr_pts = interpFun.derivative().roots()
    cr_pts = np.append(cr_pts, (x[0], x[-1]))
    cr_vals = interpFun(cr_pts)
    max_index = np.argmax(cr_vals)
    return cr_pts[max_index], cr_vals[max_index]
