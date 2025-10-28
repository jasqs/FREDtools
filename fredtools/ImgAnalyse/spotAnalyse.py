from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def findSpots(img: SITKImage, DCO: Annotated[Numberic, Field(strict=True, ge=0, le=1)] = 0.1, margin: PositiveFloat | Iterable[PositiveFloat] = 3, displayInfo: bool = False) -> SITKImage:
    """Find spots in 2D SITK image.

    The function identifies spots in a 2D SimpleITK image by applying a dose cut-off (DCO) to define the spot 
    region, followed by morphological operations to refine the regions. The identified spots are then 
    labeled and returned as a labeled image.

    Parameters
    ----------
    img : SITKImage
        2D SITK image.
    DCO : Numberic, optional
        Dose cut-off to define spot region, by default 0.1
    margin : Numberic | Iterable[Numberic], optional
        Margin around spot region in mm, by default 3
    displayInfo : bool, optional
        Whether to display information about found spots, by default False

    Returns
    -------
    SITKImage
        Labeled image of found spots.

    Raises
    ------
    ValueError
        If DCO is not a positive scalar between 0 and 1.
    TypeError
        If margin is not a scalar or an iterable.
    """
    import fredtools as ft
    import SimpleITK as sitk
    import numpy as np

    ft.ImgAnalyse._imgTypeChecker.isSITK2D(img, raiseError=True)
    # check DCO parameter
    if not isinstance(DCO, Numberic) or DCO <= 0 or DCO >= 1:
        error = ValueError(f"The value of DCO {DCO} is not correct. It must be a positive scalar between 0 and 1.")
        _logger.error(error)
        raise error
    # check margin parameter
    if isinstance(margin, Iterable):
        margin = list(margin)
    elif np.isscalar(margin):
        margin = list([margin]*img.GetDimension())
    else:
        error = TypeError(f"The `margin` parameter must be a scalar or an iterable. The parameter {margin} was used.")
        _logger.error(error)
        raise error

    imgROI = sitk.BinaryThreshold(sitk.Median(img, [5, 5]), lowerThreshold=ft.getStatistics(img).GetMaximum() * DCO, upperThreshold=ft.getStatistics(img).GetMaximum()*1.1)
    margin = np.array(margin)/np.array(img.GetSpacing())
    imgROI = sitk.BinaryDilate(imgROI, np.round(margin).astype(int).tolist())

    imgLabel = sitk.BinaryImageToLabelMap(imgROI, fullyConnected=True)
    imgLabel = sitk.Cast(imgLabel, sitk.sitkUInt8)
    imgLabel = sitk.BinaryFillhole(imgLabel, fullyConnected=True)
    imgLabel = sitk.RelabelComponent(imgLabel, minimumObjectSize=20, sortByObjectSize=True)

    # label irregular mask to box mask
    labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelShapeStatistics.Execute(imgLabel)
    for label in labelShapeStatistics.GetLabels():
        boundingBox = labelShapeStatistics.GetBoundingBox(label)
        imgLabel[boundingBox[0]:(boundingBox[0]+boundingBox[2]+1), boundingBox[1]:(boundingBox[1]+boundingBox[3]+1)] = label

    if displayInfo:
        strLog = [f"Found {labelShapeStatistics.GetNumberOfLabels()} spots."]
        _logger.info("\n\t".join(strLog) + "\n\t" + ft.ImgAnalyse.imgInfo._displayImageInfo(img))

    return imgLabel


def _single1DGaussModel(pos: ArrayLike, amplitude: Numberic, centre: Numberic, sigma: Numberic) -> NDArray:
    import numpy as np
    pos = np.array(pos)
    return amplitude * np.exp(-((pos - centre) ** 2) / (2 * sigma ** 2))


def fitSpotProfile(pos: ArrayLike, vec: ArrayLike, cutLevel: NonNegativeFloat = 0, fixAmplitude: bool = False, fixCentreToZero: bool = False, method: Literal["singleGauss"] = "singleGauss") -> LMFitModelResult:
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
    if not isinstance(pos, Iterable) or not isinstance(vec, Iterable):
        error = TypeError(f"The input `pos` and `vec` must be both iterable.")
        _logger.error(error)
        raise error
    if not np.array(pos).ndim == 1 and np.array(vec).ndim == 1:
        error = TypeError(f"The input `pos` and `vec` must be both one-dimensional vectors.")
        _logger.error(error)
        raise error
    if len(list(pos)) != len(list(vec)):
        error = TypeError(f"The input `pos` and `vec` must be of the same length.")
        _logger.error(error)
        raise error

    prof = [np.array(pos), np.array(vec)]

    # cut data above cutLevel
    cutConst = np.where(prof[1] >= (np.max(prof[1]) * cutLevel))
    prof[0] = prof[0][cutConst]
    prof[1] = prof[1][cutConst]

    match method.lower():
        case "singlegauss":
            # single gaussian fit to profile
            # calculate initial params
            initAmplitude = np.max(prof[1])
            initCentre = np.mean(prof[0][np.where(prof[1] == initAmplitude)])
            initSigma = np.ptp(prof[0][np.where(prof[1] >= (initAmplitude / 2))[0]] / 2.355)

            if fixCentreToZero:
                initCentre = 0

            gmodel = Model(_single1DGaussModel)
            gmodel.set_param_hint("amplitude", vary=not fixAmplitude)
            gmodel.set_param_hint("centre", vary=not fixCentreToZero)
            result = gmodel.fit(data=prof[1], pos=prof[0], amplitude=initAmplitude, centre=initCentre, sigma=initSigma)
            return result
        case _:
            error = ValueError(f"The method '{method}' can not be recognized. Only 'singleGauss' is available at the moment.")
            _logger.error(error)
            raise error


def _single2DGaussModel(x: NDArray, y: NDArray, amplitude: Numberic, centerX: Numberic, centerY: Numberic, sigmaX: Numberic, sigmaY: Numberic, rotation: Numberic) -> NDArray:
    import numpy as np
    rotationRad = np.deg2rad(rotation)
    # rotationRad = ft.wrapAngle(rotationRad)
    gauss2D = amplitude * np.exp(-((x - centerX)*np.cos(rotationRad) + (y - centerY)*np.sin(rotationRad))**2/(2*sigmaX**2)
                                 - ((x - centerX)*np.sin(rotationRad) - (y - centerY)*np.cos(rotationRad))**2/(2*sigmaY**2))
    return gauss2D


def fitSpotImg(img: SITKImage, cutLevel: NonNegativeFloat = 0, fixAmplitude: bool = False, fixCentreToZero: bool = False, method: Literal["singleGauss"] = "singleGauss") -> LMFitModelResult:
    """Fit a 2D gaussian-like function to spot image.

    The function fits a 2D gaussian-like function defined by `method` to a spot image
    defined as `img` parameter, which is a 2D SimpleITK image.

    Parameters
    ----------
    img : SITKImage
        2D SimpleITK image of the spot.
    cutLevel : scalar, optional
        Fraction of the maximum value of `img` for which the fit will be performed.
    fixAmplitude : bool, optional
        Fix the amplitude to the maximum value of `img` and do not use it
        in the fitting. (def. False)
    fixCentreToZero : bool, optional
        Fix the centre to zero and do not use it in the fitting. (def. False
    method : {"singleGauss"}, optional
        Method of the fitting. Only single gaussian fitting is implemented now. (def. "singleGauss")
    Returns
    -------
    lmfit.model.ModelResult
        An instance of lmfit.model.ModelResult class.
    """
    from lmfit import Model
    import numpy as np
    import SimpleITK as sitk
    import fredtools as ft
    from scipy.spatial.transform import Rotation

    ft.ImgAnalyse._imgTypeChecker.isSITK2D(img, raiseError=True)

    # cut data above cutLevel
    imgTh = sitk.Threshold(img, lower=ft.getStatistics(img).GetMaximum()*cutLevel, upper=ft.getStatistics(img).GetMaximum()*1.1, outsideValue=0)

    match method.lower():
        case "singlegauss":
            # single gaussian fit to profile
            # calculate initial params
            initAmplitude = ft.getStatistics(imgTh).GetMaximum()
            labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
            labelShapeStatistics.ComputeOrientedBoundingBoxOn()
            labelShapeStatistics.Execute(sitk.BinaryThreshold(imgTh, lowerThreshold=ft.getStatistics(imgTh).GetMaximum()*0.5, upperThreshold=ft.getStatistics(imgTh).GetMaximum()))
            rotationMatrix = np.identity(3)
            rotationMatrix[1:3, 1:3] = np.reshape(labelShapeStatistics.GetOrientedBoundingBoxDirection(1), [2, 2])
            initRotation = Rotation.from_matrix(rotationMatrix)
            initRotation = ft.wrapAngle(initRotation.as_euler("xyz")[0])
            initRotation = np.rad2deg(initRotation % np.pi)
            initSigma = ft.fwhm2sigma(labelShapeStatistics.GetOrientedBoundingBoxSize(1)[::-1])
            initCenter = ft.getMassCenter(imgTh)
            _logger.debug(f"Initial parameters: Amplitude={initAmplitude:.5f}, Centre={initCenter}, Sigma={initSigma}, Rotation={initRotation:.5f}°")

            gmodel = Model(_single2DGaussModel, independent_vars=['x', 'y'])
            params = gmodel.make_params()
            if fixAmplitude:
                params.add("amplitude", value=initAmplitude, vary=False)
            else:
                params.add("amplitude", value=initAmplitude, min=0)

            if fixCentreToZero:
                params.add("centerX", value=0, vary=False)
                params.add("centerY", value=0, vary=False)
            else:
                params.add("centerX", value=initCenter[0])
                params.add("centerY", value=initCenter[1])

            params.add("sigmaX", value=initSigma[0], min=1E-6)
            params.add("sigmaY", value=initSigma[1], min=1E-6)
            params.add("rotation", value=initRotation)  # constrain rotation to [0, 180) degrees

            arr = sitk.GetArrayViewFromImage(imgTh)
            PhysicalPointImageSource = sitk.PhysicalPointImageSource()
            PhysicalPointImageSource.SetReferenceImage(imgTh)
            imgPhysPos = PhysicalPointImageSource.Execute()
            arrPhysPos = sitk.GetArrayViewFromImage(imgPhysPos)

            result = gmodel.fit(data=arr, x=arrPhysPos[:, :, 0], y=arrPhysPos[:, :, 1], params=params)
            return result

        case _:
            error = ValueError(f"The method '{method}' can not be recognized. Only 'singleGauss' is available at the moment.")
            _logger.error(error)
            raise error
