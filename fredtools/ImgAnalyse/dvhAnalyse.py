from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


class DVHValue(object):
    """Class that stores DVH values with the appropriate units."""

    def __init__(self, value: Numberic, units: str = ''):
        """Initialization for a DVH value that will also store units."""
        self.value = value
        self.units = units

    def __repr__(self) -> str:
        """Representation of the DVH value."""
        return "DVHValue(" + self.value.__repr__() + ", '" + self.units + "')"

    def __str__(self) -> str:
        """String representation of the DVH value."""
        if not self.units:
            return format(self.value, '0.2f')
        else:
            return format(self.value, '0.2f') + ' ' + self.units

    def __eq__(self, other) -> bool:
        """Comparison method between two DVHValue objects.

        Parameters
        ----------
        other : DVHValue
            Other DVHValue object to compare with

        Returns
        -------
        bool
            True or False if the DVHValues have equal attributes
        """
        attribs_eq = self.units == other.units
        return attribs_eq and np.allclose(self.value, other.value)


class DVH(object):
    """Class that stores dose volume histogram (DVH) data."""

    def __init__(self, counts: Iterable[Numberic], bins: Iterable[Numberic], type: Literal['cumulative', 'differential'] = 'cumulative', dose_units: str = 'Gy', volume_units: str = 'cm3', dosePrescribed: Numberic | None = None, name: str | None = None, color: str | Sequence[Numberic] | None = None):
        """Initialization for a DVH from existing histogram counts and bins.

        The class has been adapted from the dicompyler-core package.

        Parameters
        ----------
        counts : iterable or numpy array
            An iterable of volume or percent count data
        bins : iterable or numpy array
            An iterable of dose bins
        type : str, optional
            Choice of 'cumulative' or 'differential' type of DVH
        dose_units : str, optional
            Absolute dose units, i.e. 'gy' or relative units '%'
        volume_units : str, optional
            Absolute volume units, i.e. 'cm3' or relative units '%'
        rx_dose : number, optional
            Prescription dose value used to normalize dose bins (in Gy)
        name : String, optional
            Name of the structure of the DVH
        color : numpy array, optional
            RGB color triplet used for plotting the DVH

        Notes
        -----
        Copyright (c) 2009-2023 Aditya Panchal and dicompyler-core contributors

        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

            Redistributions of source code must retain the above copyright
            notice, this list of conditions and the following disclaimer.

            Redistributions in binary form must reproduce the above copyright
            notice, this list of conditions and the following disclaimer in the
            documentation and/or other materials provided with the
            distribution.

            The name of Aditya Panchal may not be used to endorse or promote
            products derived from this software without specific prior written
            permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
        PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
        OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
        PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
        PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
        LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
        NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """

        self.counts = np.array(counts)
        self.bins = np.array(bins) if bins[0] == 0 else np.append([0], np.array(bins))
        self.type = type
        self.dose_units = dose_units
        self.volume_units = volume_units
        self.dosePrescribed = dosePrescribed
        self.name = name
        self.color = color

    def __repr__(self) -> str:
        """String representation of the class."""
        return f'DVH({self.type}, {self.counts.size} bins: [{self.bins.min()}:{self.bins.max()}] {self.dose_units}, volume: {self.volume:.3f} {self.volume_units}, name: {self.name}, rx_dose: {0 if not self.dosePrescribed else self.dosePrescribed} {self.dose_units})'

    def __eq__(self, other) -> bool:
        """Comparison method between two DVH objects.

        Parameters
        ----------
        other : DVH
            Other DVH object to compare with

        Returns
        -------
        bool
            True or False if the DVHs have equal attributes and via numpy.allclose
        """
        attribs = ('type', 'dose_units', 'volume_units')
        attribs_eq = {k: self.__dict__[k] for k in attribs} == {k: other.__dict__[k] for k in attribs}
        return attribs_eq and np.allclose(self.counts, other.counts) and np.allclose(self.bins, other.bins)

    def __getattr__(self, name: str) -> DVHValue:
        """Method used to dynamically determine dose or volume stats.

        Parameters
        ----------
        name : string
            Property name called to determine dose & volume statistics

        Returns
        -------
        number
            Value from the dose or volume statistic calculation.
        """
        return self.statistic(name)

    # ============================= DVH properties ============================= #

    @property
    def bincenters(self) -> NDArray[Any]:
        """Return a numpy array containing the bin centers."""
        return 0.5 * (self.bins[1:] + self.bins[:-1])

    @property
    def differential(self):
        """Return a differential DVH from a cumulative DVH."""
        type = 'differential'
        if self.type == type:
            return self
        else:
            return DVH(**dict(self.__dict__, counts=abs(np.diff(np.append(self.counts, 0))), type=type))

    @property
    def cumulative(self):
        """Return a cumulative DVH from a differential DVH."""
        type = 'cumulative'
        if self.type == type:
            return self
        else:
            return DVH(**dict(self.__dict__, counts=self.counts[::-1].cumsum()[::-1], type=type))

    def absolute_dose(self, rx_dose: Numberic | None = None, dose_units: str = 'Gy'):
        """Return an absolute dose DVH.

        Parameters
        ----------
        rx_dose : number, optional
            Prescription dose value used to normalize dose bins
        dose_units : str, optional
            Units for the absolute dose

        Raises
        ------
        AttributeError
            Description
        """
        if not (self.dose_units == '%'):
            return self
        else:
            # Raise an error if no rx_dose defined
            if not self.dosePrescribed and not rx_dose:
                error = AttributeError("'DVH' has no defined prescription dose.")
                _logger.error(error)
                raise error
            else:
                rxdose = rx_dose if self.dosePrescribed is None else self.dosePrescribed
            return DVH(**dict(self.__dict__, bins=self.bins * rxdose / 100, dose_units=dose_units))

    def relative_dose(self, rx_dose: Numberic | None = None) -> Self:
        """Return a relative dose DVH based on a prescription dose.

        Parameters
        ----------
        rx_dose : number, optional
            Prescription dose value used to normalize dose bins

        Raises
        ------
        AttributeError
            Raised if prescription dose was not present either during
            class initialization or passed via argument.
        """
        dose_units = '%'
        if self.dose_units == dose_units:
            return self
        else:
            # Raise an error if no rx_dose defined
            if not self.dosePrescribed and not rx_dose:
                error = AttributeError("'DVH' has no defined prescription dose.")
                _logger.error(error)
                raise error
            else:
                rxdose = rx_dose if rx_dose else self.dosePrescribed

            if rxdose is None:
                error = ValueError("Prescription dose (rxdose) must be provided for relative dose calculation.")
                _logger.error(error)
                raise error
            return DVH(**dict(self.__dict__, bins=100 * self.bins / rxdose, dose_units=dose_units))

    def absolute_volume(self, volume: Numberic, volume_units: str = 'cm3'):
        """Return an absolute volume DVH.

        Parameters
        ----------
        volume : number
            Absolute volume of the structure
        volume_units : str, optional
            Units for the absolute volume
        """
        if not (self.volume_units == '%'):
            return self
        else:
            return DVH(**dict(self.__dict__, counts=volume * self.counts / 100, volume_units=volume_units))

    @property
    def relative_volume(self):
        """Return a relative volume DVH."""
        volume_units = '%'
        if self.volume_units == '%':
            return self
        # Convert back to cumulative before returning a relative volume
        elif self.type == 'differential':
            return self.cumulative.relative_volume.differential
        else:
            return DVH(**dict(self.__dict__, counts=100 * self.counts / (1 if (self.max == 0) else self.counts.max()), volume_units=volume_units))

    @ property
    def max(self) -> Numberic:
        """Return the maximum dose."""
        if self.counts.size <= 1 or max(self.counts) == 0:
            return 0
        diff = self.differential
        # Find the the maximum non-zero dose bin
        return diff.bins[1:][diff.counts > 0][-1]

    @ property
    def min(self) -> Numberic:
        """Return the minimum dose."""
        if self.counts.size <= 1 or max(self.counts) == 0:
            return 0
        diff = self.differential
        # Find the the minimum non-zero dose bin
        return diff.bins[1:][diff.counts > 0][0]

    @ property
    def mean(self) -> Numberic:
        """Return the mean dose."""
        if self.counts.size <= 1 or max(self.counts) == 0:
            return 0
        diff = self.differential
        # Find the area under the differential histogram
        return (diff.bincenters * diff.counts).sum() / diff.counts.sum()

    @ property
    def volume(self) -> Numberic:
        """Return the volume of the structure."""
        return self.differential.counts.sum()

    def displayInfo(self) -> None:
        """Describe a summary of DVH statistics in a text based format."""
        strLog = [f"Statistics for structure '{self.name}':",
                  f"DVH Type:  {self.type}, {'rel dose' if self.dose_units == '%' else f'abs dose: {self.dose_units}'}, {'rel volume' if self.volume_units == '%' else f'abs volume: {self.volume_units}'}",
                  f"Volume:    {self.volume:0.2f} {self.volume_units}",
                  f"Max Dose:  {self.max:0.2f} {self.dose_units}",
                  f"Min Dose:  {self.min:0.2f} {self.dose_units}",
                  f"Mean Dose: {self.mean:0.2f} {self.dose_units}",
                  f"D100:      {self.D100}",
                  f"D98:       {self.D98}",
                  f"D95:       {self.D95}"]
        if self.dose_units == '%':
            strLog.extend([f"V100:      {self.V100}",
                           f"V95:       {self.V95}",
                           f"V5:        {self.V5}"])
        strLog.append(f"D2cc:      {self.D2cc}")
        _logger.info("\n\t".join(strLog))

    def compare(self, dvh: Self) -> None:
        """Compare the DVH properties with another DVH.

        Parameters
        ----------
        dvh : DVH
            DVH instance to compare against

        Raises
        ------
        AttributeError
            If DVHs do not have equivalent dose & volume units
        """

        # check if the DVHs have equivalent dose & volume units
        if not (self.dose_units == dvh.dose_units):
            error = AttributeError(f"DVH dose units are not equivalent ('{self.dose_units}' vs '{dvh.dose_units}')")
            _logger.error(error)
        if not (self.volume_units == dvh.volume_units):
            error = AttributeError(f"DVH volume units are not equivalent ('{self.volume_units}' vs '{dvh.volume_units}')")
            _logger.error(error)

        def fmtcmp(attr: str, units: str, ref: Self = self, comp: Self = dvh) -> Tuple[str, Numberic, str, Numberic, str, Numberic, Numberic]:
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
            if attr in ['volume', 'max', 'min', 'mean']:
                val = ref.__getattribute__(attr)
                cmpval = comp.__getattribute__(attr)
            else:
                val = ref.statistic(attr).value
                cmpval = comp.statistic(attr).value
            return attr.capitalize() + ":", val, units, cmpval, units, 0 if not val else ((cmpval - val) / val) * 100, cmpval - val

        strLog = ["{:11} {:>14} {:>17} {:>17} {:>14}".format('Structure:', self.name, dvh.name, 'Rel Diff', 'Abs diff'),
                  "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f}% {:+14.2f}".format(*fmtcmp('volume', self.volume_units)),
                  "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f}% {:+14.2f}".format(*fmtcmp('max', self.dose_units)),
                  "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f}% {:+14.2f}".format(*fmtcmp('min', self.dose_units)),
                  "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f}% {:+14.2f}".format(*fmtcmp('mean', self.dose_units)),
                  "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f}% {:+14.2f}".format(*fmtcmp('D100', self.dose_units)),
                  "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f}% {:+14.2f}".format(*fmtcmp('D98', self.dose_units)),
                  "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f}% {:+14.2f}".format(*fmtcmp('D95', self.dose_units))]
        if self.dose_units == '%':
            strLog.extend(["{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f} % {:+14.2f}".format(*fmtcmp('V100', self.dose_units, self.relative_dose(), dvh.relative_dose())),
                           "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f} % {:+14.2f}".format(*fmtcmp('V95', self.dose_units, self.relative_dose(), dvh.relative_dose())),
                           "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f} % {:+14.2f}".format(*fmtcmp('V5', self.dose_units, self.relative_dose(), dvh.relative_dose())),
                           "{:11} {:9.2f} {:3}{:15.2f} {:3}{:+16.2f} % {:+14.2f}".format(*fmtcmp('D2cc', self.dose_units))])
        _logger.info("\n\t" + "\n\t".join(strLog))
        self.plot()
        dvh.plot()

    def plot(self) -> Self:
        """Plot the DVH using Matplotlib if present."""
        try:
            import matplotlib.pyplot as plt
        except (ImportError, RuntimeError):
            error = ImportError("Matplotlib could not be loaded. Install and try again.")
            _logger.error(error)
        else:
            if isinstance(self.color, Sequence) and not isinstance(self.color, str):
                color = np.array(self.color) / 255
            else:
                color = self.color
            plt.plot(self.bincenters, self.counts, label=self.name, color=color)
            plt.xlabel(f'Dose [{self.dose_units}]')
            plt.ylabel(f'Volume [{self.volume_units}]')
            plt.xlim(0,)
            plt.ylim(0,)
            plt.grid()
            if self.name:
                plt.legend(loc='best')
        return self

    def volume_constraint(self, dose: Numberic, dose_units: str | None = None) -> DVHValue:
        """Calculate the volume that receives at least a specific dose.

        i.e. V100, V150 or V20Gy

        Parameters
        ----------
        dose : number
            Dose value used to determine minimum volume that receives
            this dose. Can either be in relative or absolute dose units.

        Returns
        -------
        number
            Volume in self.volume_units units.
        """
        # Determine whether to lookup relative dose or absolute dose
        if not dose_units:
            dose_bins = self.relative_dose().bins
        else:
            dose_bins = self.absolute_dose().bins
        index = np.argmin(np.fabs(dose_bins - dose))
        # TODO: Add interpolation
        if index >= self.counts.size:
            return DVHValue(0.0, self.volume_units)
        else:
            return DVHValue(self.counts[index], self.volume_units)

    def dose_constraint(self, volume: Numberic, volume_units: str | None = None) -> DVHValue:
        """Calculate the maximum dose that a specific volume receives.

        i.e. D90, D100 or D2cc

        Parameters
        ----------
        volume : number
            Volume used to determine the maximum dose that the volume receives.
            Can either be in relative or absolute volume units.

        Returns
        -------
        number
            Dose in self.dose_units units.
        """
        # Determine whether to lookup relative volume or absolute volume
        if not volume_units:
            volume_counts = self.relative_volume.counts
        else:
            volume_counts = self.absolute_volume(self.volume).counts

        if volume_counts.size == 0 or volume > volume_counts.max():
            return DVHValue(0.0, self.dose_units)

        # D100 case
        if volume == 100 and not volume_units:
            # Flipping the difference volume array
            reversed_difference_of_volume = np.flip(np.fabs(volume_counts - volume), 0)

            # Index of the first minimum value in reversed array
            index_min_value = np.argmin(reversed_difference_of_volume)
            index_range = len(reversed_difference_of_volume) - 1

            return DVHValue(self.bins[index_range - index_min_value], self.dose_units)

        # TODO: Add interpolation
        return DVHValue(self.bins[np.argmin(np.fabs(volume_counts - volume))], self.dose_units)

    def statistic(self, name: str) -> DVHValue:
        """Return a DVH dose or volume statistic.

        Parameters
        ----------
        name : str
            DVH statistic in the form of D90, D100, D2cc, V100 or V20Gy, etc.

        Returns
        -------
        number
            Value from the dose or volume statistic calculation.
        """
        import re
        # Compile a regex to determine dose & volume statistics
        p = re.compile(r"(\S+)?(D|V){1}(\d+[.]?\d*)(gy|cc)?(?!\S+)", re.IGNORECASE)
        match = re.match(p, name)
        # Return the default attribute if not a dose or volume statistic
        if (match is None) or (match.groups()[0] is not None):
            raise AttributeError(f"'DVH' has no attribute '{name}'")

        # Process the regex match
        c = [x.lower() for x in match.groups() if x]
        if c[0] == ('v'):
            # Volume Constraints (i.e. V100) & return a volume
            if len(c) == 2:
                return self.cumulative.volume_constraint(float(c[1]))
            # Volume Constraints in abs dose (i.e. V20Gy) & return a volume
            return self.cumulative.volume_constraint(float(c[1]), c[2])
        elif c[0] == ('d'):
            # Dose Constraints (i.e. D90) & return a dose
            if len(c) == 2:
                return self.cumulative.dose_constraint(float(c[1]))
            # Dose Constraints in abs volume (i.e. D2cc) & return a dose
            return self.cumulative.dose_constraint(float(c[1]), c[2])
        else:
            error = AttributeError(f"'DVH' has no attribute '{name}'")
            _logger.error(error)
            raise error


def getDVHMask(img: SITKImage, imgMask: SITKImage, dosePrescribed: NonNegativeFloat, doseLevelStep: float = 0.01, displayInfo: bool = False) -> DVH:
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

    # convert images to vectors
    arrImg = sitk.GetArrayViewFromImage(img)
    arrMask = sitk.GetArrayViewFromImage(imgMask)

    # remove all voxels not within mask
    arrMaskValid = arrMask > 0
    arrImg = arrImg[arrMaskValid]
    arrMask = arrMask[arrMaskValid]

    # calculate DVH
    doseBins = np.arange(0, np.nanmax(arrImg) + doseLevelStep, doseLevelStep)
    volume, doseBins = np.histogram(arrImg, doseBins, weights=arrMask.astype("float"))

    # calculate volume in real units [mm3]
    volume *= np.prod(img.GetSpacing())

    # get color and name
    maskColor = imgMask.GetMetaData("ROIColor") if "ROIColor" in imgMask.GetMetaDataKeys() else "[0, 0, 255]"
    maskColor = [int(x) for x in re.findall(r'(\d+)', maskColor)]  # convert string to list of integers
    maskName = imgMask.GetMetaData("ROIName") if "ROIName" in imgMask.GetMetaDataKeys() else "unknown"

    # generate DVH
    dvhMask = DVH(volume / 1e3, doseBins.tolist(), dosePrescribed=dosePrescribed, name=maskName, color=maskColor, type="differential").cumulative

    if displayInfo:
        dvhStatLog = [f"Prescribed dose: {dosePrescribed:.3f} Gy",
                      f"Volume: {dvhMask.volume:.3f} {dvhMask.volume_units}",
                      f"Dose max/min: {dvhMask.max:.3f}/{dvhMask.min:.3f} {dvhMask.dose_units}",
                      f"Dose mean: {dvhMask.mean:.3f} {dvhMask.dose_units}",
                      f"Absolute HI (D02-D98): {dvhMask.statistic('D02').value-dvhMask.statistic('D98').value:.3f} {dvhMask.dose_units}",
                      f"D98: {dvhMask.statistic('D98').value:.3f} {dvhMask.dose_units}",
                      f"D50: {dvhMask.statistic('D50').value:.3f} {dvhMask.dose_units}",
                      f"D02: {dvhMask.statistic('D02').value:.3f} {dvhMask.dose_units}"]
        _logger.info(f"DVH statistics for '{dvhMask.name}' structure:" + "\n\t" + "\n\t".join(dvhStatLog))

    return dvhMask


def getDVHStruct(img: SITKImage, RSfileName: str, structName: str, dosePrescribed: float, doseLevelStep: float = 0.01, resampleImg: float | Iterable[float] | None = None, displayInfo: bool = False) -> DVH:
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
        error = ValueError(f"The file {RSfileName} is not a proper dicom describing structures.")
        _logger.error(error)
        raise error

    # get structure info
    structList = ft.getRSInfo(RSfileName)
    if structName not in list(structList.ROIName):
        error = ValueError(f"Cannot find the structure '{structName}' in {RSfileName} dicom file with structures. Available structures are: {list(structList.ROIName)}.")
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
        img = ft.resampleImg(img=img, spacing=np.array(resampleImg))

    # map structure to resampled img generating a floating mask
    imgROI = ft.mapStructToImg(img=img, RSfileName=RSfileName, structName=structName, binaryMask=False, displayInfo=False)

    dvhROI = getDVHMask(img, imgROI, dosePrescribed=dosePrescribed, doseLevelStep=doseLevelStep)

    if displayInfo:
        dvhStatLog = [f"Prescribed dose: {dosePrescribed:.3f} Gy",
                      f"Volume: {dvhROI.volume:.3f} {dvhROI.volume_units}",
                      f"Dose max/min: {dvhROI.max:.3f}/{dvhROI.min:.3f} {dvhROI.dose_units}",
                      f"Dose mean: {dvhROI.mean:.3f} {dvhROI.dose_units}",
                      f"Absolute HI (D02-D98): {dvhROI.statistic('D02').value-dvhROI.statistic('D98').value:.3f} {dvhROI.dose_units}",
                      f"D98: {dvhROI.statistic('D98').value:.3f} {dvhROI.dose_units}",
                      f"D50: {dvhROI.statistic('D50').value:.3f} {dvhROI.dose_units}",
                      f"D02: {dvhROI.statistic('D02').value:.3f} {dvhROI.dose_units}"]
        _logger.info(f"DVH statistics for '{dvhROI.name}' structure:" + "\n\t" + "\n\t".join(dvhStatLog))

    return dvhROI
