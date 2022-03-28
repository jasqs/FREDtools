from matplotlib.pyplot import sca


def mergePDF(PDFFileNames, mergedPDFFileName, removeSource=False, displayInfo=False):
    """Merge multiple PDF files to a single PDF.

    The function merges multiple PDF files given as a list of
    path strings to a single PDF.

    Parameters
    ----------
    PDFFileNames : list of strings
        List of path strings to PDF files to be merged.
    mergedPDFFileName : string
        Path string where the merged PDF will be saved.
    removeSource : bool, optional
        Determine if the source PDF files should be
        removed after merge. (def. False)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    mergedPDFFileName
        Absolute path string where the merged PDF was be saved.
    """
    import fitz  # from pymupdf
    import os
    import fredtools as ft

    # check if it is a single string
    if isinstance(PDFFileNames, str):
        PDFFileNames = [PDFFileNames]

    # check if all files to be merged exist
    for PDFFileName in PDFFileNames:
        if not os.path.exists(PDFFileName):
            raise FileNotFoundError(f"The file {PDFFileName} dose not exist.")

    mergedPDF = fitz.open()

    for PDFFileName in PDFFileNames:
        with fitz.open(PDFFileName) as mfile:
            mergedPDF.insert_pdf(mfile)

    if removeSource:
        for PDFFileName in PDFFileNames:
            os.remove(PDFFileName)

    mergedPDF.save(mergedPDFFileName)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Merged PDF files:\n# " + "\n# ".join(PDFFileNames))
        print(f"# Saved merged PDF to: ", mergedPDFFileName)
        if removeSource:
            print(f"# Removed the source PDF files")
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return os.path.abspath(mergedPDFFileName)


def getGIcmap(maxGI, N=256):
    """Get colormap for Gamma Index images.

    The function creates a colormap for Gamma Index (GI) images,
    that can be used by matplotlib.pyplot.imshow function for
    displaying 2D images. The colormap is created from 0 to
    the `maxGI` value, whereas from 0 to 1 (GI test passed) the colour
    is changing from dark blue to white, and from 1 to `maxGI` it is
    changing from light red to red.

    Parameters
    ----------
    maxGI : scalar
        Maximum value of the colormap.
    N : scalar, optional
        Number of segments of the colormap. (def. 256)

    Returns
    -------
    colormap
        An instance of matplotlib.colors.LinearSegmentedColormap object.

    See Also
    --------
        calcGammaIndex: calculate Gamma Index for two images.

    Examples
    --------
    It is assumed that the img is an image describing a slice
    of Gamma Index (GI) values calculate up to maximum value 3.
    To plot the GI map with the GI colormap:

    >>> plt.imshow(ft.arr(img), cmap=getGIcmap(maxGI=3))
    """
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import warnings

    if maxGI < 1:
        warnings.warn(f"Warning: the value of the parameter 'maxGI' cannot be less than 1 and a value {maxGI} was given. It was set to 1.")
        maxGI = 1

    colorLowStart = np.array([1, 0, 128]) / 255
    colorLowEnd = np.array([253, 253, 253]) / 255
    colorHighStart = np.array([254, 193, 192]) / 255
    colorHighEnd = np.array([255, 67, 66]) / 255

    cdict = {
        "red": ((0.0, 0.0, colorLowStart[0]), (1 / maxGI, colorLowEnd[0], colorHighStart[0]), (1.0, colorHighEnd[0], 0.0)),
        "green": ((0.0, 0.0, colorLowStart[1]), (1 / maxGI, colorLowEnd[1], colorHighStart[1]), (1.0, colorHighEnd[1], 0.0)),
        "blue": ((0.0, 1.0, colorLowStart[2]), (1 / maxGI, colorLowEnd[2], colorHighStart[2]), (1.0, colorHighEnd[2], 0.0)),
    }

    cmapGI = LinearSegmentedColormap(name="GIcmap", segmentdata=cdict, N=N)

    return cmapGI


def getHistogram(dataX, dataY=None, bins=None, kind="mean", returnBinCenters=True):
    """Get histogram or differential histogram.

    The function creates a histogram data from a given dataX iterable in the defined bins.
    It is possible to generate a differential histogram where the values of the histogram
    (usually Y-axis on a plot) are a given quantity, instead of frequency of `dataX` values
    occurrance.

    Parameters
    ----------
    dataX : 1D array_like
        1D array-like iterable with the data to calculate histogram.
        For instance, it can be: a single-column pandas DataFrame,
        pandas Series, 1D numpy array, 1D list, 1D tuple etc.
    dataY : 1D array_like, optional
        1D array-like iterable with the data to calculate differential histogram.
        It must be of the same size as `dataX`. For instance, it can be: a single-column
        pandas DataFrame, pandas Series, 1D numpy array, 1D list, 1D tuple etc. (def. None)
    bins : 1D array_like, optional
        1D array-like iterable with the bins' edges to calculate histogram.
        If none, then the bins will be generated automatically between
        minimum and maximum value of `dataX` in 100 steps linearly. (def. None)
    kind : {'mean', 'sum', 'std', 'median', 'min', 'max', 'mean-std', 'mean+std'}, optional
        Determine the `dataY` quantity evaluation for a differential histogram.
        It can be: mean, standard deviation, median, minimum, maximum, sum
        value or mean +/- standard deviation. (def. 'mean')
    returnBinCenters : bool, optional
        Determine if the first element of returned list is going to
        be the bin centres (True) or bin edges (False). (def. True)

    Returns
    -------
    List of two ndarrays
        Two-element tuple of 1D numpy ndarrays, where the first element
        is a list of bin centres (or edges) and the second is a list of
        histogram values.
    """
    import numpy as np

    # check if dataX and dataY are iterable
    from collections.abc import Iterable

    if not isinstance(dataX, Iterable):
        raise TypeError(f"The variable 'dataX' is not an iterable. It must be a 1D iterable.")
    if dataY is not None and not isinstance(dataY, Iterable):
        raise TypeError(f"The variable 'dataY' is not an iterable. It must be a 1D iterable.")

    # convert dataX to ndarray if needed
    if not isinstance(dataX, np.ndarray):
        dataX = np.array(dataX).squeeze()

    # check if dataX is 1D array
    if dataX.ndim != 1:
        raise ValueError(f"The parameter 'dataX' must be a 1D iterable, e.g. a single column pandas DataFrame, 1D list or tuple, etc.")

    # convert dataY to ndarray if needed
    if dataY is not None and not isinstance(dataY, np.ndarray):
        dataY = np.array(dataY).squeeze()

    # check if dataY is 1D array
    if dataY is not None and dataY.ndim != 1:
        raise ValueError(f"The parameter 'dataY' must be a 1D iterable, e.g. a single column pandas DataFrame, 1D list or tuple, etc.")

    # check if dataY is of the same length as dataX
    if dataY is not None and len(dataX) != len(dataY):
        raise ValueError(f"The length of the 'dataY' iterable must be the same as the length of the 'dataX' iterable but they have {len(dataY)} and {len(dataX)} lengths, respectively.")

    # create bins if not given
    if bins is None:
        bins = np.linspace(np.nanmin(dataX), np.nanmax(dataX), 100)

    # validate kind parameter
    if dataY is not None and kind not in [
        "sum",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "mean-std",
        "mean+std",
    ]:
        raise ValueError(f"The value of 'kind' parameter must be 'sum', 'mean', 'std', 'median', 'mean-std', 'mean+std', 'min' or 'max' but '{kind}' was given.")

    # creates a histogram for dataX
    hist = list(np.histogram(dataX, bins=bins))[::-1]
    hist[0] = hist[0].astype("float")

    # creates a differential histogram if dataY is given
    if dataY is not None:
        hist[1] = hist[1].astype("float")
        for i in range(hist[0].size - 1):
            histEntry = dataY[(dataX >= hist[0][i]) & (dataX < hist[0][i + 1])]

            if not len(histEntry):
                histEntry = np.nan
            else:
                if kind == "sum":
                    histEntry = histEntry.sum()
                elif kind == "mean":
                    histEntry = histEntry.mean()
                elif kind == "std":
                    histEntry = histEntry.std()
                elif kind == "median":
                    histEntry = histEntry.median()
                elif kind == "min":
                    histEntry = histEntry.min()
                elif kind == "max":
                    histEntry = histEntry.max()
                elif kind == "mean-std":
                    histEntry = histEntry.mean() - histEntry.std()
                elif kind == "mean+std":
                    histEntry = histEntry.mean() + histEntry.std()
            hist[1][i] = histEntry

    # calculate bin centres instead of bin edges if requested
    if returnBinCenters:
        hist[0] = hist[0][:-1] + np.diff(hist[0]) / 2

    # convert hist[1] to float (useful for postprocessing normalistion)
    hist[1] = hist[1].astype("float")

    return hist


def pdfLandau(x, mpv, xi, amp=1):
    """Landau probability density function (PDF).

    The function generates a Landau probability density with a given most probable
    value (`mpv`), width (described with `xi`) and amplitude at `mpv`. It was adapted
    from [1]_ which was implemented based on the ROOT implementation. See [2]_ for more details.

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

    # check parameters
    if not np.isscalar(mpv):
        raise TypeError(f"The 'mpv' parameter must be a scalar but it is {type(mpv)}")
    if not np.isscalar(xi):
        raise TypeError(f"The 'xi' parameter must be a scalar but it is {type(xi)}")
    if not np.isscalar(amp):
        raise TypeError(f"The 'amp' parameter must be a scalar but it is {type(amp)}")

    if not (0 < xi):
        raise ValueError("The 'xi' parameter must be xi > 0.")
    if not (0 <= amp):
        raise ValueError("The 'amp' parameter must be amp >= 0.")

    return amp * landau.pdf(x, x_mpv=mpv, xi=xi) / landau.pdf(mpv, x_mpv=mpv, xi=xi)


def pdfLandauGauss(x, mpv, xi, sigma=0, amp=1):
    """Probability density function (PDF) of Landau convoluted with a Gaussian.

    The function generates a Landau convoluted with a Gaussian probability density with a given
    most probable value of the convoluted function (`mpv`), width of Landau (described with `xi`),
    standard deviation of gaussian and amplitude at `mpv`. It was adapted from [3]_ which was implemented
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
        Standard deviation of the gaussian distribution. (def. 0)
    amp : scalar, optional
        Amplitude of the PDF at MPV. (def. 1)

    Returns
    -------
    scalar or numpy array
        Single value or array of values of the Landau convoluted with gaussian PDF.

    See Also
    --------
    fitLandauGauss : fit Landau distribution convoluted with a Gaussian to data.

    Notes
    -----
    The 'mpv' parameter dose not describe the MPV of the landau distribution but the MPV,
    i.e the position of the maximum value, of the whole Landau-gauss convoluted PDF.

    References
    ----------
    .. [3] `landaupy python package <https://pypi.org/project/landaupy/>`_
    .. [4] `landaupy package documentation <https://github.com/SengerM/landaupy>`_
    """
    from landaupy import langauss
    import numpy as np
    from scipy.interpolate import InterpolatedUnivariateSpline

    def getMPV(x, y):
        # calculate MPV and the maximum value
        interpFun = InterpolatedUnivariateSpline(x, y, k=4)
        cr_pts = interpFun.derivative().roots()
        cr_pts = np.append(cr_pts, (x[0], x[-1]))
        cr_vals = interpFun(cr_pts)
        max_index = np.argmax(cr_vals)
        return cr_pts[max_index], cr_vals[max_index]

    # check parameters
    if not np.isscalar(mpv):
        raise TypeError(f"The 'mpv' parameter must be a scalar but it is {type(mpv)}")
    if not np.isscalar(xi):
        raise TypeError(f"The 'xi' parameter must be a scalar but it is {type(xi)}")
    if not np.isscalar(amp):
        raise TypeError(f"The 'amp' parameter must be a scalar but it is {type(amp)}")
    if not np.isscalar(sigma):
        raise TypeError(f"The 'sigma' parameter must be a scalar but it is {type(sigma)}")

    if not (0 < xi):
        raise ValueError("The 'xi' parameter must be xi > 0.")
    if not (0 <= sigma):
        raise ValueError("The 'sigma' parameter must be sigma >= 0.")
    if not (0 <= amp):
        raise ValueError("The 'amp' parameter must be amp >= 0.")

    xInternal = x.copy()

    # move x position to the expected mpv
    mpvInternal = getMPV(xInternal, langauss.pdf(xInternal, landau_x_mpv=mpv, landau_xi=xi, gauss_sigma=sigma))
    xInternal += mpvInternal[0] - mpv

    # normalize PDF to the amplitude
    yInternal = langauss.pdf(xInternal, landau_x_mpv=mpv, landau_xi=xi, gauss_sigma=sigma)
    yInternal /= mpvInternal[1]
    yInternal *= amp

    return yInternal


def fitLandau(x, y, fixAmplitude=False):
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
        determine if the `amp` parameter of the PDF should be used in the fiting.

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


def fitLandauGauss(x, y, fixAmplitude=False):
    """Fit Landau convoluted with gaussian distribution.

    The function fits Landau convoluted with gaussian distribution
    to the data given as `x` and `y` values, using the least square algorithm.

    Parameters
    ----------
    x : array_like
        `X` values.
    y : array_like
        `Y` values.
    fixAmplitude : bool, optional
        determine if the `amp` parameter of the PDF should be used in the fiting.

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


def pdfVavilov(x, mpv, kappa, beta, scaling, amp=1):
    """Probability density function (PDF) of Vavilov.

    The function generates a Vavilov probability density with a given
    most probable value function (`mpv`), amplitude (`amp`), as well as `kappa`,
    `beta` and `scaling` parameters. It uses the implementation of pyamtrack library [5]_
    that adopts the ROOT implementation [6]_. The implemented PDF is not a true Vavilov distribution
    and the `scaling` parameter is not included in the original ROOT implementation. Therefore, the parameters
    `kappa` and `beta` might not describe the the real kappa and beta parameters of the ROOT Vavilov.
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
    from scipy.interpolate import InterpolatedUnivariateSpline

    def getMPV(x, y):
        # calculate MPV and the maximum value
        interpFun = InterpolatedUnivariateSpline(x, y, k=4)
        cr_pts = interpFun.derivative().roots()
        cr_pts = np.append(cr_pts, (x[0], x[-1]))
        cr_vals = interpFun(cr_pts)
        max_index = np.argmax(cr_vals)
        return cr_pts[max_index], cr_vals[max_index]

    # check parameters
    if not np.isscalar(mpv):
        raise TypeError(f"The 'mpv' parameter must be a scalar but it is {type(mpv)}")
    if not np.isscalar(kappa):
        raise TypeError(f"The 'kappa' parameter must be a scalar but it is {type(kappa)}")
    if not np.isscalar(beta):
        raise TypeError(f"The 'beta' parameter must be a scalar but it is {type(beta)}")
    if not np.isscalar(amp):
        raise TypeError(f"The 'amp' parameter must be a scalar but it is {type(amp)}")

    if not (0.01 <= kappa <= 12):
        raise ValueError("The 'kappa' parameter must be in range 0.01 <= kappa <= 12.")
    if not (0 <= beta <= 1):
        raise ValueError("The 'beta' parameter must be in range 0 <= beta <= 1.")
    if not (0 <= amp):
        raise ValueError("The 'amp' parameter must be amp >= 0.")
    if not (0 <= scaling):
        raise ValueError("The 'scaling' parameter must be scaling >= 0.")

    xInternal = x.copy()
    xInternal = np.asarray(xInternal, dtype=float)
    yInternal = np.zeros(xInternal.size)

    # move x position to the expected mpv
    xMPVCalc = np.linspace(-10, 10, 1000)
    yMPVCalc = np.zeros(xMPVCalc.size)
    AT_Vavilov_PDF(xMPVCalc.tolist(), p_kappa=kappa, p_beta=beta, p_density=yMPVCalc)
    mpvInternal = getMPV(xMPVCalc, yMPVCalc)
    xInternal /= scaling
    xInternal += mpvInternal[0]
    xInternal -= mpv / scaling

    AT_Vavilov_PDF(xInternal.tolist(), p_kappa=kappa, p_beta=beta, p_density=yInternal)

    # normalize PDF to the amplitude
    yInternal /= mpvInternal[1]
    yInternal *= amp

    return yInternal


def fitVavilov(x, y, beta0=0.5, kappa0=0.3, scaling0=-1, fixAmplitude=False):
    """Fit Vavilov distribution.

    The function fits Vavilov distribution to the data given as `x` and `y` values,
    using the least square algorithm. The fiting routine is sensitive for the initial
    values of kappa, beta and scaling. Therefore, the results should be always validated
    and different initial values of the parameters can be used if needed.

    Parameters
    ----------
    x : array_like
        `X` values.
    y : array_like
        `Y` values.
    fixAmplitude : bool, optional
        determine if the `amp` parameter of the PDF should be used in the fiting.
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
    amp0 = np.nanmax(y)
    mpv0 = x[np.where(np.array(y) == amp0)[0]][0]
    if scaling0 < 0:
        scaling0 = np.sqrt(np.cov(x, aweights=y)) * 0.3

    # prepare constraints for the parameters
    fitModel.set_param_hint("mpv", min=0, max=np.inf, value=mpv0, vary=True)
    fitModel.set_param_hint("amp", min=0, max=np.inf, value=amp0, vary=not fixAmplitude)
    fitModel.set_param_hint("kappa", min=0.01, max=12, value=kappa0, vary=True)
    fitModel.set_param_hint("scaling", min=0.0, max=np.inf, value=scaling0, vary=True)
    fitModel.set_param_hint("beta", min=0, max=1, value=beta0, vary=True)

    # perform fit
    fitResult = fitModel.fit(data=y, x=x)

    return fitResult
