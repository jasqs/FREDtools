from fredtools._typing import *
re_number = r"[-+]?[\d]+\.?[\d]*[Ee]?(?:[-+]?[\d]+)?"


def mergePDF(PDFFileNames, mergedPDFFileName, removeSource=False, displayInfo: bool = False):
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
        removed after the merge. (def. False)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    mergedPDFFileName
        Absolute path string where the merged PDF will be saved.
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
        print(f"### {ft.currentFuncName()} ###")
        print(f"# Merged PDF files:\n# " + "\n# ".join(PDFFileNames))
        print(f"# Saved merged PDF to: ", mergedPDFFileName)
        if removeSource:
            print(f"# Removed the source PDF files")
        print("#" * len(f"### {ft.currentFuncName()} ###"))

    return os.path.abspath(mergedPDFFileName)


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
        the minimum and maximum value of `dataX` in 100 steps linearly. (def. None)
    kind : {'mean', 'sum', 'std', 'median', 'min', 'max', 'mean-std', 'mean+std'}, optional
        Determine the `dataY` quantity evaluation for a differential histogram.
        It can be: mean, standard deviation, median, minimum, maximum, sum
        value or mean +/- standard deviation. (def. 'mean')
    returnBinCenters : bool, optional
        Determine if the first element of the returned list is going to
        be the bin centers (True) or bin edges (False). (def. True)

    Returns
    -------
    List of two ndarrays
        A two-element tuple of 1D numpy ndarrays, where the first element
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


def sigma2fwhm(sigma):
    """Convert sigma to FWHM.

    The function recalculates the sigma parameter of a Gaussian distribution
    to full width at half maximum (FWHM).

    Parameters
    ----------
    sigma : scalar
        Sigma value.

    Returns
    -------
    scalar
        FWHM value.

    See Also
    --------
    fwhm2sigma : convert FWHM to sigma.
    """
    from numpy import log, sqrt

    return 2 * sqrt(2 * log(2)) * sigma


def fwhm2sigma(fwhm):
    """Convert FWHM to sigma.

    The function recalculates full width at half maximum (FWHM)
    of a Gaussian distribution to sigma.

    Parameters
    ----------
    fwhm : scalar
        FWHM value.

    Returns
    -------
    scalar
        Sigma value.

    See Also
    --------
    sigma2fwhm : convert sigma to FWHM.
    """
    from numpy import log, sqrt

    return fwhm / (2 * sqrt(2 * log(2)))


def getLineFromFile(pattern, fileName, kind="all", startLine=1, removeEoL=True, comment="#"):
    """Read the line and line number from an ASCI file.

    The function searches an ASCI file for lines matching a pattern and returns
    the line or lines number and the line strings. The pattern follows the Python
    regular expression [7]_.

    Parameters
    ----------
    pattern : string
        A string describing the regular expression. It is recommended
        the string be a row string, starting with r'...'.
    fileName : string
        Path String to ASCI file.
    kind : {'all', 'first', 'last'}, optional
        Determine which line is to be returned: the first only, the last, or 
        all the lines. (def. 'all')
    startLine : int, optional
        The line number to start the search (def. 1)
    removeEoL : bool, optional
        Determine if the end-of-line should be removed from 
        each returned line. (def. True)
    comment : strung, optional
        If not None or an empty string, then no lines starting with this
        string (leading white spaces are removed) will be returned. (def. '#')

    Returns
    -------
    line index, line string
        If kind='all': a tuple of two tuples where the first is the 
        matched line numbers and the second is the line strings.
        If kind='first' or kind='last': a tuple with the first or last 
        reached line number and the line string.

    References
    ----------
    .. [7] `Regular expression operations <https://docs.python.org/3/library/re.html>`_
    """
    import re

    with open(fileName, "r") as f:
        fileLines = f.readlines()

    lineIdx = []
    lineString = []
    for i, line in enumerate(fileLines, start=1):
        # start saving from startLine
        if i < startLine:
            continue
        # if 'comment' is not empty, do not save lines starting with comment sign
        if comment:
            if re.findall(rf"^\s*{comment}", line):
                continue

        if re.findall(pattern, line):
            lineIdx.append(i)
            # remove end-of-line sign '\n' if reqiested
            if removeEoL:
                line = line.replace("\n", "")
            lineString.append(line)

    # return None if not matching was found
    if not lineIdx:
        return None

    if kind.lower() == "first":
        lineIdx = lineIdx[0]
        lineString = lineString[0]
    elif kind.lower() == "last":
        lineIdx = lineIdx[-1]
        lineString = lineString[-1]
    elif kind.lower() == "all":
        lineIdx = tuple(lineIdx)
        lineString = tuple(lineString)
    else:
        raise AttributeError(f"Unrecognized kind = '{kind}' parameter. Only 'first', 'last', and 'all' are supported.")

    return lineIdx, lineString


def getCPUNo(CPUNo: Literal["auto"] | NonNegativeInt = "auto") -> NonNegativeInt:
    """Get a number of CPU cores.

    The function returns the number of CPU cores. Usually, it is used in functions utilizing
    multiprocessing.

    Parameters
    ----------
    CPUNo : {'auto'} or integer, optional
        A string of 'auto' for all the available CPU cores, or a positive integer showing the number of CPU cores. (def. 'auto')

    Returns
    -------
    integer
        Number of CPU cores.
    """
    from os import cpu_count
    import fredtools as ft
    logger = ft.getLogger()

    if isinstance(CPUNo, str) and CPUNo.lower() in ["auto"]:
        CPUNumber = cpu_count()
        if not CPUNumber:
            error = ValueError(f"Could not get the number of CPUs with os.cpu_count() routine.")
            logger.error(error)
            raise error
        else:
            return CPUNumber

    elif isinstance(CPUNo, int) and CPUNo > 0:
        return CPUNo

    else:
        error = ValueError(f"The parameter CPUno '{CPUNo}' cannot be recognized. Only a positive integer or 'auto' are possible.")
        logger.error(error)
        raise error
