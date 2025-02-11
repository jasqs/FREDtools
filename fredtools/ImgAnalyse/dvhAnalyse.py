from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


class DVH(object):
    """Class that stores dose volume histogram (DVH) data."""

    def __init__(self, volume: Iterable[Numberic], dose: Iterable[Numberic], type: Literal['cumulative', 'differential'] = 'cumulative', dosePrescribed: Numberic | None = None, name: str | None = None, color: str | Sequence[Numberic] | None = None):
        """DVH class to store dose volume histogram data.

        The class stores the DVH data in the form of counts and bins. The counts
        can be either in the form of volume or percent counts and can be provided 
        in a cumulative or differential form. The class can be used to calculate 
        various DVH statistics such as D98, D2cc, V100, V20Gy, etc. 

        Parameters
        ----------
        counts : iterable
            An iterable of absolute or relative volume for each bin.
        bins : iterable
            An iterable of a quantity (e.g. dose) bin edges. The size of the bins
            should be one more than the size of the counts and should be in the
            increasing order.
        type : {'cumulative', 'differential'}, optional
            Choice of 'cumulative' or 'differential' type of DVH (def. 'cumulative')
            Absolute volume units, i.e. 'cm3' or relative units '%'
        dosePrescribed : Numberic or None, optional
            Prescription quantity (e.g. dose) value used to normalize dose bins. If not provided,
            the average dose will be used as the prescription dose. (def. None)
        name : str | None, optional
            Name of the structure of the DVH. If not provided, it will be set to 'unknown'. (def. None)
        color : 3x1 RGB color triplet | str | None, optional
            Color triplet used for plotting the DVH. If not provided, it will be set to 'b' (blue). (def. None)

        Raises
        ------
        AttributeError
            If the 'type' is not 'cumulative' or 'differential'.
            If the size of the 'bins' is not one more than the size of the 'counts'.
            If the 'bins' are not in the increasing order.

        Notes
        -----
        The class has been prepared based on the iomplementation in the dicompyler-core package.
        """
        import numpy as np

        # check type
        if type not in ['cumulative', 'differential']:
            error = AttributeError("The 'type' should be either 'cumulative' or 'differential'.")
            _logger.error(error)
            raise error

        # check volume
        volume = np.array(volume)
        if volume.size == 0 or volume[-1] != 0:
            error = AttributeError("The 'volume' should be a non-empty iterable with the last element equal to 0.")
            _logger.error(error)
            raise error
        match type:
            case 'differential':
                self._volumeDiff = np.asarray(volume)
                self._volumeCum = np.cumsum(self._volumeDiff[::-1])[::-1]
            case 'cumulative':
                self._volumeCum = np.asarray(volume)
                self._volumeDiff = np.abs(np.diff(np.append(self._volumeCum, 0)))

        # check dose
        dose = np.array(dose)
        if np.diff(dose).min() < 0:
            error = AttributeError("The 'bins' should be in the increasing order.")
            _logger.error(error)
            raise error
        match type:
            case 'differential':
                if dose.size != self._volumeDiff.size + 1:
                    error = AttributeError("For differential DVH type, the 'dose' should describe the dose bin edges, hence its size should be one more than the number of 'volume'.")
                    _logger.error(error)
                    raise error
                self._doseDiffEdges = dose
                self._doseDiffCenters = 0.5 * (dose[1:] + dose[:-1])
                self._doseCum = dose[0:-1]
            case 'cumulative':
                if dose.size != self._volumeCum.size:
                    error = AttributeError("For cumulative DVH type, the 'dose' should describe the dose levels, hence its size should be equal to the number of 'volume'.")
                    _logger.error(error)
                    raise error
                self._doseCum = dose
                self._doseDiffEdges = np.append(dose, dose[-1] + (dose[-1]-dose[-2]))
                self._doseDiffCenters = 0.5 * (self._doseDiffEdges[1:] + self._doseDiffEdges[:-1])

        # check dosePrescribed
        if dosePrescribed is None:
            self._dosePrescribed = float(np.average(self._doseDiffCenters, weights=self._volumeDiff))
            _logger.debug("The prescription dose was not provided, hence it was calculated as the average dose.")
        else:
            self._dosePrescribed = float(dosePrescribed)

        # check name
        self.name = name
        if self.name is None:
            self.name = 'unknown'
            _logger.debug("The name was not provided, hence it was set to 'unknown'.")
        # check color
        self.color = color
        if self.color is None:
            self.color = 'b'
            _logger.debug("The color was not provided, hence it was set to 'b'.")

    def __repr__(self) -> str:
        """String representation of the class."""
        return f'DVH({self._doseCum.size} bins [{self._doseCum.min()}:{self._doseCum.max()}] Gy, volume: {self.volume:.3f} cm3, name: {self.name}, dosePrescribed: {self._dosePrescribed:.3f} Gy)'

    def __eq__(self, other) -> bool:
        """Comparison method between two DVH objects.

        The method compares the DVH objects in terms of bins and counts, using numpy.allclose method.

        Parameters
        ----------
        other : DVH
            Other DVH object to compare with.

        Returns
        -------
        bool
            True or False if the DVHs are equal or not.
        """
        if not isinstance(other, DVH):
            return False
        return np.allclose(self._volumeCum, other._volumeCum) and np.allclose(self._doseCum, other._doseCum)

    def __getattr__(self, name: str) -> Numberic | Self:
        """Method used to dynamically determine dose or volume stats.

        Parameters
        ----------
        name : str
            Property name called to determine dose or volume statistics

        Returns
        -------
        Numberic
            Value from the dose or volume statistic calculation.
        """
        if len(name) > 1 and name[0] == '_':
            return self

        return self.statistic(name)

    # ============================= DVH properties ============================= #

    @property
    def doseLevels(self) -> NDArray:
        """Return a numpy array containing the dose levels for cumulative type."""
        return np.asarray(self._doseCum)

    @property
    def doseDiffCenters(self) -> NDArray:
        """Return a numpy array containing the dose bin centers for differential type."""
        return np.asarray(self._doseDiffCenters)

    @property
    def doseDiffEdges(self) -> NDArray:
        """Return a numpy array containing the bin edges for differential type."""
        return np.asarray(self._doseDiffEdges)

    @property
    def volumeDiffAbs(self) -> NDArray:
        """Return a numpy array containing absolute differential counts."""
        return np.asarray(self._volumeDiff)

    @property
    def volumeDiffRel(self) -> NDArray:
        """Return a numpy array containing relative differential counts."""
        return np.asarray(self._volumeDiff/self.volume*100)

    @property
    def volumeCumAbs(self) -> NDArray:
        """Return a numpy array containing absolute cumulative counts."""
        return np.asarray(self._volumeCum)

    @property
    def volumeCumRel(self) -> NDArray:
        """Return a numpy array containing relative cumulative counts."""
        return np.asarray(self._volumeCum/self.volume*100)

    @property
    def dosePrescribed(self) -> float:
        """Return the prescribed dose."""
        return float(self._dosePrescribed)

    @property
    def max(self) -> float:
        """Return the maximum dose."""
        # Find the the maximum non-zero dose bin
        # return float(self._doseDiffCenters[np.nonzero(self._volumeCum)[0][-1]])
        return float(self._doseDiffEdges[np.nonzero(self._volumeDiff)[0][-1]+1])

    @property
    def min(self) -> float:
        """Return the minimum dose."""
        # Find the the minimum non-zero dose bin
        # return float(self._doseDiffEdges[np.nonzero(self._volumeDiff)[0][0]])
        return float(self._doseDiffEdges[np.nonzero(self._volumeDiff)[0][0]])

    @property
    def mean(self) -> float:
        """Return the mean dose."""
        return float(np.average(self._doseDiffCenters, weights=self._volumeDiff))

    @property
    def stdDev(self) -> float:
        """Return the standard deviation of the dose."""
        return np.sqrt(np.average((self._doseDiffCenters - np.average(self._doseDiffCenters, weights=self._volumeDiff))**2, weights=self._volumeDiff))

    @property
    def median(self) -> float:
        """Return the median dose."""
        return float(self.doseConstraint(50, absolute=False))

    @property
    def volume(self) -> float:
        """Return the volume of the structure."""
        return self._volumeDiff.sum()

    def volumeConstraint(self, dose: Numberic, absolute: bool = False) -> float:
        """Calculate volume constraint for a specific dose.

        The method calculates the volume that receives at least a specific relative or absolute dose.
        e.g.: V100, V50, V20Gy, etc.

        Parameters
        ----------
        dose : Numberic
            Dose value used to determine minimum volume that receives
            this dose. Can either be in relative or absolute dose units.
        absolute : bool, optional
            If True, the dose is considered in absolute units. (def. False)

        Returns
        -------
        Numberic
            Volume that receives at least a specific dose.
        """
        # check if dose is positive
        if dose < 0:
            error = AttributeError("The dose value must be positive.")
            _logger.error(error)
            raise error

        # Determine whether to lookup relative or absolute counts
        if not absolute:
            dose = dose / 100 * self._dosePrescribed

        return float(np.interp(float(dose), self._doseDiffCenters, self._volumeCum-self._volumeDiff/2, left=self.volume, right=0))

    def doseConstraint(self, volume: Numberic, absolute: bool = False) -> float:
        """Calculate dose constraint for a specific volume.

        The method calculates the maximum dose that a specific absolute or relative volume receives.
        e.g.: D90, D100, D2cc , etc. The results is always given in the absolute dose units.

        Parameters
        ----------
        volume : Numberic
            Volume used to determine the maximum dose that the volume receives.
            Can either be in relative or absolute volume units.
        absolute : bool, optional
            If True, the volume is considered in absolute units. (def. False)

        Returns
        -------
        number
            Absolute dose that a specific volume receives.
        """
        if volume < 0:
            error = AttributeError("The volume value must be positive.")
            _logger.error(error)
            raise error

        # Determine whether to lookup relative volume or absolute volume
        if not absolute:
            volume = volume / 100 * self.volume

        counts = (self._volumeCum-self._volumeDiff/2)[::-1]
        uniqueCountsIndex = np.unique(counts, return_index=True, return_inverse=True)[1]
        uniqueCountsIndex = np.append(uniqueCountsIndex[1]-1, uniqueCountsIndex[1:])

        return float(np.interp(float(volume), counts[uniqueCountsIndex], self._doseDiffCenters[::-1][uniqueCountsIndex]))

    def statistic(self, name: str) -> float:
        """DVH dose or volume statistics

        The method returns DVH dose or volume statistics. The statistics can
        be in the form of D90, D100, D2cc, V100, V20Gy, etc.

        Parameters
        ----------
        name : str
            DVH statistic in the form of D90, D100, D2cc, V100, V20Gy, etc.

        Returns
        -------
        Numberic
            Value from the dose or volume statistic calculation.

        Raises
        ------
        AttributeError
            If the attribute name cannot be resolved.
        """
        import re
        # Compile a regex to determine dose & volume statistics
        p = re.compile(r"(\S+)?(D|V){1}(\d+[.]?\d*)(.+)?(?!\S+)", re.IGNORECASE)
        match = re.match(p, name)
        # Return the default attribute if not a dose or volume statistic
        if (match is None) or (match.groups()[0] is not None):
            error = AttributeError(f"Cannot resolve attribute '{name}'.")
            _logger.error(error)
            raise error

        # Process the regex match
        c = [x.lower() for x in match.groups() if x]
        if c[0] == ('v'):
            # Relative Volume Constraint (e.g. V100)
            if len(c) == 2:
                return self.volumeConstraint(float(c[1]), absolute=False)
            # Absolute Volume Constraints (e.g. V20Gy)
            return self.volumeConstraint(float(c[1]), absolute=True)
        elif c[0] == ('d'):
            # Relavive Dose Constraints (e.g. D90)
            if len(c) == 2:
                return self.doseConstraint(float(c[1]), absolute=False)
            # Absolute Dose Constraints (e.g. D2cc)
            return self.doseConstraint(float(c[1]), absolute=True)
        else:
            error = AttributeError(f"Cannot resolve attribute '{name}'")
            _logger.error(error)
            raise error

    def _displayInfo(self) -> str:
        strLog = [f"DVH statistics for structure '{self.name}':",
                  f"Prescribed dose: {self._dosePrescribed:.3f}",
                  f"Volume:          {self.volume:0.3f}",
                  f"Min/Max Dose:    {self.min:0.3f} / {self.max:0.3f}",
                  f"Mean/Std Dose:   {self.mean:0.3f} / {self.stdDev:0.3f}",
                  f"D98:       {self.D98:.3f}",
                  f"D02:       {self.D02:.3f}",
                  f"D50:       {self.D50:.3f}",
                  f"D2cc:      {self.D2cc:.3f}",
                  f"V100:      {self.V100:.3f}"]
        return "\n\t".join(strLog)

    def displayInfo(self) -> None:
        """Describe a summary of DVH statistics in a text based format."""
        _logger.info(self._displayInfo())

    def compare(self, other: Self) -> None:
        """Compare two DVHs.

        The method compares two DVHs in terms of basic DVH parameters and plots the DVHs.

        Parameters
        ----------
        dvh : DVH
            DVH instance to compare against.

        Raises
        ------
        TypeError
            If the comparison object is not a DVH instance.
        """

        if not isinstance(other, DVH):
            error = TypeError("The comparison object must be a DVH instance.")
            _logger.error(error)
            raise error

        def fmtcmp(attr: str, ref: Self = self, comp: Self = other) -> Tuple[str, Numberic, Numberic, Numberic, Numberic]:
            """Generate arguments for string formatting.

            Parameters
            ----------
            attr : string
                Attribute used for comparison
            units : string
                Units used for the value

            Returns
            -------
            tuple
                tuple used in a string formatter
            """
            if attr in ['volume', 'max', 'min', 'mean', 'stdDev']:
                val = ref.__getattribute__(attr)
                cmpval = comp.__getattribute__(attr)
            else:
                val = ref.statistic(attr)
                cmpval = comp.statistic(attr)
            return attr + ":", val, cmpval, 0 if not val else ((cmpval - val) / val) * 100, cmpval - val

        strLog = ["{:11} {:>14} {:>17} {:>17} {:>14}".format('Structure:', self.name, other.name, 'Rel Diff', 'Abs diff'),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('volume')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('max')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('min')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('mean')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('stdDev')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('D100')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('D95')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('D50')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('D05')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('V95')),
                  "{:18} {:9.2f} {:17.2f} {:+14.2f}% {:+14.2f}".format(*fmtcmp('D05')),
                  ]
        _logger.info(f"Comparizon of the basic DVH parameters for structures '{self.name}' and '{other.name}':\n\t" + "\n\t".join(strLog))

        try:
            import matplotlib.pyplot as plt
        except (ImportError, RuntimeError):
            error = ImportError("Matplotlib could not be loaded. Install and try again.")
            _logger.error(error)
        else:
            fig, ax = plt.subplots()
            if isinstance(self.color, Sequence) and not isinstance(self.color, str):
                color = np.array(self.color) / 255
            else:
                color = self.color
            ax.plot(self.doseLevels, self.volumeCumRel, label=self.name, color=color)
            ax.plot(other.doseLevels, other.volumeCumRel, label=self.name, color=color)
            ax.set_xlabel(f'Bin centers (Dose) [abs. units]')
            ax.set_ylabel(f'Volume [%]')
            ax.set_xlim(0,)
            ax.set_ylim(0, 105)
            ax.grid()
            ax.legend(loc='best')
        pass

    def plot(self) -> None:
        """Plot the DVH using Matplotlib if present."""
        try:
            import matplotlib.pyplot as plt
        except (ImportError, RuntimeError):
            error = ImportError("Matplotlib could not be loaded. Install and try again.")
            _logger.error(error)
        else:
            fig, ax = plt.subplots()
            if isinstance(self.color, Sequence) and not isinstance(self.color, str):
                color = np.array(self.color) / 255
            else:
                color = self.color
            ax.plot(self.doseLevels, self.volumeCumRel, label=self.name, color=color)
            ax.set_xlabel(f'Bin centers (Dose) [abs. units]')
            ax.set_ylabel(f'Volume [%]')
            ax.set_xlim(0,)
            ax.set_ylim(0, 105)
            ax.grid()
            ax.axvline(float(self._dosePrescribed), ymin=0, ymax=100/105, color=color, linestyle='--', label='Prescribed dose')
            ax.legend(loc='best')
        pass


def getDVHMask(img: SITKImage, imgMask: SITKImage, dosePrescribed: NonNegativeFloat, displayInfo: bool = False) -> DVH:
    """Calculate DVH for the mask.

    The function calculates a dose-volume histogram (DVH) for voxels inside
    a mask with the same field of reference (FoR). The mask must defined as
    a SimpleITK image object describing a binary or floating mask.
    The routine exploits and returns DVH class to hold the DVH. The class
    has been adapted from dicompyler-core DVH module (https://dicompyler-core.readthedocs.io/en/latest/index.html)

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK 3D image.
    imgMask : SimpleITK Image
        Object of a SimpleITK 3D image describing the binary of floating mask.
    dosePrescribed : scalar
        Target prescription dose.
    doseLevelStep : scalar, optional
        Size of dose bins. (def. 0.01)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    DVH
        An instance of a DVH class holding the DVH.

    See Also
    --------
        resampleImg : resample image.
        mapStructToImg : map structure to image (refer for more information about mapping algorithms).
        getDVHStruct : calculate DVH for structure.
    """
    import fredtools as ft
    import numpy as np
    import SimpleITK as sitk
    import re

    ft._imgTypeChecker.isSITK3D(img, raiseError=True)
    ft._imgTypeChecker.isSITK3D(imgMask, raiseError=True)
    ft._imgTypeChecker.isSITK_mask(imgMask, raiseError=True)

    # check FoR matching of img and mask
    if not ft.compareImgFoR(img, imgMask, displayInfo=False):
        error = TypeError("Both input images, 'img' and 'imgMask' must have the same FoR.")
        _logger.error(error)
        raise error

    # convert images to arrays
    arrImg = sitk.GetArrayViewFromImage(img)
    arrMask = sitk.GetArrayViewFromImage(imgMask)

    # remove all voxels not within mask
    arrMaskValid = arrMask > 0
    arrImg = arrImg[arrMaskValid]
    arrMask = arrMask[arrMaskValid]

    # calculate DVH
    def getVolumeCumAbs(arrImg, arrMask, doseLevel):
        return arrMask[arrImg > doseLevel].sum()

    doseMax = np.nanmax(arrImg)
    doseMin = np.nanmin(arrImg)
    # first round with fixed dose levels
    doseLevels = np.array([0, doseMin-1E-7, doseMin])
    doseLevels = np.append(doseLevels, np.linspace(doseMin+1E-7, doseMax, 50))
    doseLevels = np.append(doseLevels, [doseMax-1E-7, doseMax+1E-7])
    doseLevels = np.unique(doseLevels)
    volumeCumAbs = np.array([getVolumeCumAbs(arrImg, arrMask, doseLevel) for doseLevel in doseLevels])

    # second round with adaptive dose levels
    volume = arrMask.sum()
    volumeMaxRelativeChange = 0.01
    doseMaxRelativeStep = 0.001
    doseMaxAbsoluteStep = doseMaxRelativeStep * doseMax
    maxIter = 20
    doseLevelsNew = doseLevels[:-1] + np.diff(doseLevels) / 2
    doseLevelsNew = doseLevelsNew[(-np.diff(volumeCumAbs) / volume) >= volumeMaxRelativeChange]

    iter = 0
    while np.any(doseLevelsNew) and iter < maxIter:
        volumeCumAbsNew = [getVolumeCumAbs(arrImg, arrMask, doseLevelNew) for doseLevelNew in doseLevelsNew]
        # Append new dose levels to the original doseLevels
        doseLevels = np.append(doseLevels, doseLevelsNew)
        volumeCumAbs = np.append(volumeCumAbs, volumeCumAbsNew)
        # Sort the dose levels
        volumeCumAbs = volumeCumAbs[np.argsort(doseLevels)]
        doseLevels = np.sort(doseLevels)
        # Get the new dose levels
        doseLevelsNew = doseLevels[:-1] + np.diff(doseLevels) / 2
        doseLevelsNew = doseLevelsNew[(-np.diff(volumeCumAbs) / volume) >= volumeMaxRelativeChange]
        doseLevelsNew = doseLevelsNew[doseLevelsNew >= doseMaxAbsoluteStep]

        iter += 1
        if iter >= maxIter:
            _logger.debug(f"Stopped adaptive dose level stage after reaching max iterations.")
    _logger.debug(f"Performed {iter} iterations in adaptive dose level stage to calculate DVH.")

    # calculate volume in real units [mm3]
    volumeCumAbs = volumeCumAbs * np.prod(img.GetSpacing()) / 1E3

    # get color and name
    maskColor = imgMask.GetMetaData("ROIColor") if "ROIColor" in imgMask.GetMetaDataKeys() else "[0, 0, 255]"
    maskColor = [int(x) for x in re.findall(r'(\d+)', maskColor)]  # convert string to list of integers
    maskName = imgMask.GetMetaData("ROIName") if "ROIName" in imgMask.GetMetaDataKeys() else "unknown"

    # generate DVH
    # dvhMask = DVH(volume / 1e3, doseBins, dosePrescribed=dosePrescribed, name=maskName, color=maskColor, type="differential")
    dvhMask = DVH(volumeCumAbs, doseLevels, dosePrescribed=dosePrescribed, name=maskName, color=maskColor, type="cumulative")

    if displayInfo:
        _logger.info(dvhMask._displayInfo())

    return dvhMask


def getDVHStruct(img: SITKImage, RSfileName: str, structName: str, dosePrescribed: float, resampleImg: float | Iterable[float] | None = None, displayInfo: bool = False) -> DVH:
    """Calculate DVH for the structure.

    The function calculates a dose-volume histogram (DVH) for voxels inside
    a structure named `structName` and is defined in the structure dicom file.
    The image can be resampled before mapping to increase the resolution. The routine
    exploits and returns DVH class to hold the DVH. The class has been adapted
    from dicompyler-core DVH module (https://dicompyler-core.readthedocs.io/en/latest/index.html)
    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK 3D image.
    RSfileName : path
        String path to RS dicom file.
    structName : string
        Name of the structure to calculate the DVH in.
    dosePrescribed : scalar
        Target prescription dose.
    doseLevelStep : scalar, optional
        Size of dose bins. (def. 0.01)
    resampleImg : scalar, array_like or None, optional
        Define if and how to resample the image while mapping the structure.
        Can be a scalar, then the same number will be used for each axis,
        3-element iterable defining the voxel size for each axis, or `None` meaning
        no interpolation. (def. None)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dicompylercore DVH
        An instance of a dicompylercore.dvh.DVH class holding the DVH.

    See Also
    --------
        dicompylercore : more information about the dicompylercore.dvh.DVH.
        resampleImg : resample image.
        mapStructToImg : map structure to image (refer for more information about mapping algorithms).
        getDVHMask : calculate DVH for a mask.

    Notes
    -----
    The structure mapping is performed in 'floating' mode meaning that the fractional structure
    occupancy is assigned to each voxel of the image. Use `getDVHMask` to use a specially prepared
    mask for the DVH calculation.
    """
    import fredtools as ft
    import numpy as np

    ft._imgTypeChecker.isSITK3D(img, raiseError=True)
    if not ft.ImgIO.dicom_io._isDicomRS(RSfileName):
        error = TypeError(f"The file {RSfileName} is not a proper dicom describing structures.")
        _logger.error(error)
        raise error

    # get structure info
    structList = ft.getRSInfo(RSfileName)
    if structName not in list(structList.ROIName):
        error = AttributeError(f"Cannot find the structure '{structName}' in {RSfileName} dicom file with structures. Available structures are: {list(structList.ROIName)}.")
        _logger.error(error)
        raise error

    # resample image if requested
    if resampleImg:
        # check if rescaleImg is in the proper format
        if not isinstance(resampleImg, Iterable):
            resampleImg = [resampleImg] * img.GetDimension()
        else:
            resampleImg = list(resampleImg)
            if not len(resampleImg) == img.GetDimension():
                error = ValueError(f"Shape of 'spacing' is {resampleImg} but must match the dimension of 'img' {img.GetDimension()} or be a scalar.")
                _logger.error(error)
                raise error

        # resample image
        _logger.debug(f"Resampling image with spacing {resampleImg}.")
        img = ft.resampleImg(img=img, spacing=np.array(resampleImg))

    # map structure to resampled img generating a floating mask
    imgROI = ft.mapStructToImg(img=img, RSfileName=RSfileName, structName=structName, binaryMask=False, displayInfo=False)

    dvhROI = getDVHMask(img, imgROI, dosePrescribed=dosePrescribed)

    if displayInfo:
        _logger.info(dvhROI._displayInfo())

    return dvhROI
