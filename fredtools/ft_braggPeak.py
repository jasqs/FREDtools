class braggPeak:
    """Class for Bragg curve analysis.

    This class is holding methods for a Bragg peak (BP) analysis and
    properties of the analysis results. The analysis of a Bragg curve given
    as an instance of SimpleITK image object describing a profile is made
    based on two methods: a simple interpolation with a given method (linear,
    nearest or spline) and/or a fit of the Bortfeld equation taken from [1]_ (eq. 27).
    For each method, it is possible to obtain such parameters as: range of the BP
    at given percent of the maximum of the distal fall-off, value of signal
    (for instance dose) at given depth, distal fall-off and width of the BP
    at given percent of the maximum.

    Parameters
    ----------
    img : SimpleITK Image
        Object of a SimpleITK image describing a profile.
    accuracy : float, optional
        Accuracy of the spline and Bortfeld profiles interpolations. (def. 0.01)
    interpolation : {'spline', 'linear', 'nearest'}, optional
        Interpolation method. (def. 'spline')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)
    bortCut : float, optional
        The range of the data that the Bortfeld fit will be performed on.
        It is defined as the range of the BP in proximal region at the percent
        of the maximum of the spline interpolation. The Bortfeld fit will be
        performed for the input data in distal of this range. (def. 0.6)


    References
    ----------
    .. [1] Bortfeld, T. An analytical approximation of the Bragg curve for therapeutic proton beams. Med. Phys. 24, 2024 (1997).
    """

    def __init__(self, img, accuracy=0.01, interpolation="spline", splineOrder=3, bortCut=0.6):
        import fredtools as ft

        ft._isSITK_profile(img, raiseError=True)

        self.__axis = ft.ft_imgAnalyse._getAxesNumberNotUnity(img)[0]
        self.__img = img
        self.__accuracy = accuracy
        self.__interpolation = interpolation
        self.__splineOrder = splineOrder
        self.__bortCut = bortCut

        self.__imgInterp = []
        self.__imgBort = []
        self.__bortfeldFit = []

        self.__bortfeldParamConstant = {
            "p": 1.726,  # [-]
            "alpha": 0.028,  # [mm*MeV^(-p)]
            "epsilon": 0.2,  # [-] Fraction of primary fluence contributing to the "tail" of the energy spectrum
        }

    @property
    def interpolation(self):
        """str: Interpolation method. Available are 'linear', 'nearest' or 'spline'."""
        return self.__interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        # validate argument
        if interpolation not in ["linear", "spline", "nearest"]:
            raise ValueError(f"Interpolation type '{interpolation}' cannot be recognized. Only 'linear', 'nearest' and 'spline' are supported.")
        self.__interpolation = interpolation
        self.__imgInterp = []

    @property
    def splineOrder(self, splineOrder):
        """int: Order of the spline interpolation. Must be in range 0-5."""
        return self.__splineOrder

    @splineOrder.setter
    def splineOrder(self, splineOrder):
        # validate argument
        if not isinstance(splineOrder, int) or splineOrder > 5 or splineOrder < 0:
            raise ValueError(f"Spline order must be a scalar in range 0-5.")
        self.__splineOrder = splineOrder
        self.__imgInterp = []

    @property
    def accuracy(self):
        """float: Accuracy of the spline and Bortfeld profiles interpolations."""
        return self.__accuracy

    @accuracy.setter
    def accuracy(self, accuracy):
        import numpy as np

        # validate argument
        if not np.isscalar(accuracy) or accuracy <= 0:
            raise ValueError(f"The value {accuracy} is not correct. It must be a positive scalar.")
        self.__accuracy = accuracy
        self.__imgInterp = []
        self.__imgBort = []

    @property
    def bortCut(self):
        """float: The range of the data to perform Bortfeld fit on."""
        return self.__bortCut

    @bortCut.setter
    def bortCut(self, bortCut):
        import numpy as np

        # validate argument
        if not np.isscalar(bortCut) or bortCut < 0 or bortCut > 1:
            raise ValueError(f"The value {bortCut} is not correct. It must be a positive scalar in range 0-1.")
        self.__bortCut = bortCut
        self.__imgBort = []
        self.__bortfeldFit = []

    @property
    def imgInterp(self):
        """SimpleITK image: Instance of a SimpleITK image object describing interpolated profile."""
        import fredtools as ft

        if not self.__imgInterp:
            interpSpacing = list(self.__img.GetSpacing())
            interpSpacing[self.__axis] = self.__accuracy
            self.__imgInterp = ft.resampleImg(self.__img, spacing=interpSpacing, interpolation=self.__interpolation, splineOrder=self.__splineOrder)
            return self.__imgInterp
        else:
            return self.__imgInterp

    @property
    def imgBort(self):
        """SimpleITK image: Instance of a SimpleITK image object describing Bortfeld fit profile."""
        import numpy as np
        import fredtools as ft
        import SimpleITK as sitk

        if not self.__imgBort:
            bortfeldFit = self.bortfeldFit
            imgPos = bortfeldFit.userkws["depth"]
            # build sitk image for bortfeld fit curve with accuracy step
            imgBortPos = np.arange(imgPos[0], imgPos[-1], self.__accuracy)
            imgBortVal = bortfeldFit.eval(depth=imgBortPos)
            arrBort = np.expand_dims(imgBortVal, list(range(1, self.__img.GetDimension())))  # add remaining dimensions to get the same dimension as original image
            arrBort = np.moveaxis(arrBort, 0, ft.ft_imgAnalyse._getAxesNumberNotUnity(self.__img)[0])  # move axis with values to the position of the original image
            arrBort = np.moveaxis(arrBort, list(range(self.__img.GetDimension())), list(range(self.__img.GetDimension()))[::-1])  # change axis order to be consistent with sitk Image
            self.__imgBort = sitk.GetImageFromArray(arrBort, isVector=False)
            # set origin
            origin = list(self.__img.GetOrigin())
            origin[ft.ft_imgAnalyse._getAxesNumberNotUnity(self.__img)[0]] = imgBortPos[0]
            self.__imgBort.SetOrigin(origin)
            # set spacing
            spacing = list(self.__img.GetSpacing())
            spacing[ft.ft_imgAnalyse._getAxesNumberNotUnity(self.__img)[0]] = self.__accuracy
            spacing
            self.__imgBort.SetSpacing(spacing)
            return self.__imgBort
        else:
            return self.__imgBort

    def getDInterp(self, R):
        """Get signal value at given range/depth based on profile interpolation.

        The function is calculating the signal value (for instance dose) at given
        range (for instance depth) using the profile interpolation defined with
        interpolation and splineOrder.

        Parameters
        ----------
        R : float
            Range/depth at which the signal value will be calculated.

        Returns
        -------
        float
            Signal value at given range.

        Examples
        --------
        Get maximum value of the interpolation method.

        >>> braggPeak.getDInterp(R=braggPeak.getRInterp(D=1))
        """
        return self.__getD(self.__img, R)

    def getDBort(self, R):
        """Get signal value at given range/depth based on Bortfeld fit.

        The function is calculating the signal value (for instance dose) at given
        range (for instance depth) using the Bortfeld fit.

        Parameters
        ----------
        R : float
            Range/depth at which the signal value will be calculated.

        Returns
        -------
        float
            Signal value at given range.

        Examples
        --------
        Get maximum value of the Bortfeld fit.

        >>> braggPeak.getDBort(R=braggPeak.getRBort(D=1))
        """
        return self.__getD(self.__imgBort, R)

    def __getD(self, img, R):
        """Calculate value for given range/depth for image.

        Parameters
        ----------
        img : SimpleITK Image
            Object of a SimpleITK image describing a profile.
        R : float
            Range/depth at which the signal value will be calculated.

        Returns
        -------
        float
            Signal value at given range.
        """
        import fredtools as ft

        point = list(self.__img.GetOrigin())
        point[self.__axis] = R
        imgPoint = ft.getPoint(img, point=point, interpolation=self.__interpolation, splineOrder=self.__splineOrder)
        return ft.arr(imgPoint).item()

    def getRInterp(self, D, side="distal", percentD=True):
        """Calculate range/depth at given signal level based on profile interpolation.

        The function is calculating range (depth) at distal or proximal part, at the absolute
        or relative signal value using the profile interpolation defined with `interpolation`
        and `splineOrder`. The relative value is calculated to the maximum value of the profile
        interpolation.

        Parameters
        ----------
        D : float
            Absolute or relative signal level.
        side : {'proximal'|'P', 'distal'|'D'}, optional
            Determine the side, proximal or distal to the maximum range, of the BP to calculate the range. (def. 'distal')
        percentD : bool, optional
            Determine if the signal level is relative (True) to the maximum or absolute (False). (def. True)

        Returns
        -------
        float
            Range at signal level.

        Examples
        --------
        Get range at 80 percent of the maximum value of the interpolation at distal fall-off of the BP.

        >>> braggPeak.getRInterp(D=0.8)

        Get range at 50 percent of the maximum value of the interpolation in proximal region.

        >>> braggPeak.getRInterp(D=0.5, side='P')
        """
        return self.__getR(self.imgInterp, D=D, side=side, percentD=percentD)

    def getRBort(self, D, side="distal", percentD=True):
        """Calculate range/depth at given signal level based on Bortfeld fit.

        The function is calculating range (depth) at distal or proximal part, at the absolute
        or relative signal value using the Bortfeld fit. The relative value is calculated
        to the maximum value of the Bortfeld fit.

        Parameters
        ----------
        D : float
            Absolute or relative signal level.
        side : {'proximal'|'P', 'distal'|'D'}, optional
            Determine the side, proximal or distal to the maximum range, of the BP to calculate the range. (def. 'distal')
        percentD : bool, optional
            Determine if the signal level is relative (True) to the maximum or absolute (False). (def. True)

        Returns
        -------
        float
            Range at signal level.

        Examples
        --------
        Get range at 80 percent of the maximum value of the Bortfeld fit at distal fall-off of the BP.

        >>> braggPeak.getRBort(D=0.8)

        Get range at 50 percent of the maximum value of the Bortfeld fit in proximal region.

        >>> braggPeak.getRBort(D=0.5, side='P')
        """
        return self.__getR(self.imgBort, D=D, side=side, percentD=percentD)

    def __getR(self, img, D, side, percentD):
        """Calculate range/depth at given signal level.

        Parameters
        ----------
        img : SimpleITK Image
            Object of a SimpleITK image describing a profile.
        D : float
            Absolute or relative signal level.
        side : {'proximal', 'distal'}
            Determine the side, proximal or distal to the maximum range, of the BP to calculate the range.
        percentD : bool
            Determine if the signal level is relative (True) to the maximum or absolute (False).

        Returns
        -------
        float
            Range at signal level.
        """
        import numpy as np
        import fredtools as ft

        side = side.lower()
        # check if side is in proper format and unify it
        if side not in ["p", "d", "prox", "dist", "proximal", "distal"]:
            raise ValueError(f"Parameter `side` {side} cannot be recognized. Only the values `proximal` (or `P`) or `distal` (or `D`) are allowed.")
        if side in ["p", "prox", "proximal"]:
            side = "proximal"
        elif side in ["d", "dist", "distal"]:
            side = "distal"

        imgPos = np.array(ft.pos(img))
        imgVal = ft.vec(img)

        # determine level
        if percentD:
            level = D * np.nanmax(imgVal)
        else:
            level = D

        # get index of the maximum value
        R100idx = np.argmax(imgVal)

        if side == "proximal":
            imgVal = imgVal[0 : R100idx + 1]
            imgPos = imgPos[0 : R100idx + 1]
            return imgPos[np.where(imgVal >= level)].min() if (any(imgVal >= level) and any(imgVal <= level)) else np.nan
        elif side == "distal":
            imgVal = imgVal[R100idx:]
            imgPos = imgPos[R100idx:]
            return imgPos[np.where(imgVal >= level)].max() if (any(imgVal >= level) and any(imgVal <= level)) else np.nan

    def getWInterp(self, D, percentD=True):
        """Calculate width of the BP at given signal level based on profile interpolation.

        The function is calculating width of the BP at the absolute or relative signal value
        using the profile interpolation defined with `interpolation` and `splineOrder`.
        The relative value is calculated to the maximum value of the profile interpolation.

        Parameters
        ----------
        D : float
            Absolute or relative signal level.
        percentD : bool, optional
            Determine if the signal level is relative (True) to the maximum or absolute (False). (def. True)

        Returns
        -------
        float
            Width at signal level.

        Examples
        --------
        Get width at 50 percent of the maximum value of the interpolation.

        >>> braggPeak.getWInterp(D=0.5)

        This is equivalent to the code.

        >>> braggPeak.getRInterp(D=0.5, side='D') - braggPeak.getRInterp(D=0.5, side='P')
        """
        return self.__getW(self.imgInterp, D=D, percentD=percentD)

    def getWBort(self, D, percentD=True):
        """Calculate width of the BP at given signal level based on Bortfeld fit.

        The function is calculating width of the BP at the absolute or relative signal value
        using the Bortfeld fit.  The relative value is calculated to the maximum value of
        the Bortfeld fit.

        Parameters
        ----------
        D : float
            Absolute or relative signal level.
        percentD : bool, optional
            Determine if the signal level is relative (True) to the maximum or absolute (False). (def. True)

        Returns
        -------
        float
            Width at signal level.

        Examples
        --------
        Get width at 50 percent of the maximum value of the Bortfeld fit.

        >>> braggPeak.getWBort(D=0.5)

        This is equivalent to the code.

        >>> braggPeak.getWBort(D=0.5, side='D') - braggPeak.getWBort(D=0.5, side='P')
        """
        return self.__getW(self.imgBort, D=D, percentD=percentD)

    def __getW(self, img, D, percentD):
        """Calculate width of the BP at given signal level for image.

        Parameters
        ----------
        img : SimpleITK Image
            Object of a SimpleITK image describing a profile.
        D : float
            Absolute or relative signal level.
        percentD : bool, optional
            Determine if the signal level is relative (True) to the maximum or absolute (False). (def. True)

        Returns
        -------
        float
            Width at signal level.
        """
        Rprox = self.__getR(img, D, side="proximal", percentD=percentD)
        Rdist = self.__getR(img, D, side="distal", percentD=percentD)
        return Rdist - Rprox

    def getDFOInterp(self, Dup, Dlow, percentD=True):
        """Calculate width of the distal fall-off of the BP at given signal level based on profile interpolation.

        The function is calculating width of the distal fall-off of the BP at the absolute or
        relative signal values using the profile interpolation defined with `interpolation`
        and `splineOrder`. The relative value is calculated to the maximum value of the profile interpolation.

        Parameters
        ----------
        Dup : float
            Absolute or relative upper signal level.
        Dlow : float
            Absolute or relative lower signal level.
        percentD : bool, optional
            Determine if the signal level is relative (True) to the maximum or absolute (False). (def. True)

        Returns
        -------
        float
            Width of the distal fall-off at signal level.

        Examples
        --------
        Get width of the distal fall-off between 80 and 20 percent of the maximum value of the interpolation.

        >>> braggPeak.getDFOInterp(Dup=0.8, Dlow=0.2)

        This is equivalent to the code.

        >>> braggPeak.getRInterp(D=0.2, side='D') - braggPeak.getRInterp(D=0.8, side='D')
        """
        return self.__getDFO(self.imgInterp, Dup=Dup, Dlow=Dlow, percentD=percentD)

    def getDFOBort(self, Dup, Dlow, percentD=True):
        """Calculate width of the distal fall-off of the BP at given signal level based on Bortfeld fit.

        The function is calculating width of the distal fall-off of the BP at the absolute or
        relative signal values using the Bortfeld fit. The relative value is calculated to the maximum
        value of the Bortfeld fit.

        Parameters
        ----------
        Dup : float
            Absolute or relative upper signal level.
        Dlow : float
            Absolute or relative lower signal level.
        percentD : bool, optional
            Determine if the signal level is relative (True) to the maximum or absolute (False). (def. True)

        Returns
        -------
        float
            Width of the distal fall-off at signal level.

        Examples
        --------
        Get width of the distal fall-off between 80 and 20 percent of the maximum value of the Bortfeld fit.

        >>> braggPeak.getDFOBort(Dup=0.8, Dlow=0.2)

        This is equivalent to the code.

        >>> braggPeak.getRBort(D=0.2, side='D') - braggPeak.getRBort(D=0.8, side='D')
        """
        return self.__getDFO(self.imgBort, Dup=Dup, Dlow=Dlow, percentD=percentD)

    def __getDFO(self, img, Dup, Dlow, percentD):
        """Calculate width of the distal fall-off of the BP at given signal level for image.

        Parameters
        ----------
        img : SimpleITK Image
            Object of a SimpleITK image describing a profile.
        Dup : float
            Absolute or relative upper signal level.
        Dlow : float
            Absolute or relative lower signal level.
        percentD : bool, optional
            Determine if the signal level is relative (True) to the maximum or absolute (False). (def. True)

        Returns
        -------
        float
            Width of the distal fall-off at signal level.
        """
        if Dup < Dlow:
            raise ValueError(f"The parameter Dup must be higher than Dlow.")
        Rup = self.__getR(img, Dup, side="distal", percentD=percentD)
        Rdown = self.__getR(img, Dlow, side="distal", percentD=percentD)
        return Rdown - Rup

    @property
    def bortfeldFit(self):
        """lmfit.model.ModelResult: Result from the Model of the Bortfeld fit."""
        if not self.__bortfeldFit:
            self.__bortfeldFit = self.__fitBortfeld()
            return self.__bortfeldFit
        else:
            return self.__bortfeldFit

    @property
    def bortfeldFitParam(self):
        """dict: Physical parameters of the calculated based on the Bortfeld fit."""
        import fredtools as ft
        import numpy as np

        bortfeldFit = self.bortfeldFit
        bortfeldParam = self.__bortfeldParamConstant
        bortfeldResults = {}
        bortfeldResults["R0_mm"] = bortfeldFit.params["R0"].value - ft.getExtent(self.__img)[ft.ft_imgAnalyse._getAxesNumberNotUnity(self.__img)[0]][0]
        bortfeldResults["E0_MeV"] = (bortfeldResults["R0_mm"] / bortfeldParam["alpha"]) ** (1 / bortfeldParam["p"])
        bortfeldResults["sigmaMono_mm"] = 0.012 * bortfeldResults["R0_mm"] ** 0.935
        bortfeldResults["sigmaE0_MeV"] = np.sqrt(
            (bortfeldFit.params["sigma"].value ** 2 - bortfeldResults["sigmaMono_mm"] ** 2)
            / (bortfeldParam["alpha"] ** 2 * bortfeldParam["p"] ** 2 * bortfeldResults["E0_MeV"] ** (2 * bortfeldParam["p"] - 2))
        )
        return bortfeldResults

    def __fitBortfeld(self):
        """Perform a Bortfeld fit.

        The function os preparing and performing a Bortfeld fit to the original data
        defined as an instance of a SimpleITK object describing a profile. It uses
        the functionality of the lmfit module. The initial parameters for the fit
        are calculated based on the interpolation method.
        """
        from lmfit import Model
        import numpy as np
        import fredtools as ft

        # get image positions and values
        imgPos = np.array(ft.pos(self.__img))
        imgVal = ft.vec(self.__img)

        # crop the positions and values to the distal part from the bortCut
        bortCutPos = self.getRInterp(self.__bortCut, "proximal")
        imgVal = imgVal[np.where(imgPos >= bortCutPos)]
        imgPos = imgPos[np.where(imgPos >= bortCutPos)]

        def bortfeldEquation(depth, R0, phi0, epsilon, sigma):
            # definition of vectorized Bortfeld equation
            def D(depth, R0, phi0, epsilon, sigma):
                # definition of single value Bortfeld equation
                if depth < (R0 - 10 * sigma):
                    fac = (phi0) / (1.0 + 0.012 * R0)
                    term1 = 17.93 * ((R0 - depth) ** -0.435)
                    term2 = ((0.444 + 31.7 * epsilon) / R0) * ((R0 - depth) ** 0.565)
                    return fac * (term1 + term2)
                elif depth < (R0 + 5 * sigma):
                    import scipy.special as sp

                    D565, grad565 = sp.pbdv(-0.565, -((R0 - depth) / sigma))
                    D1565, grad1565 = sp.pbdv(-1.565, -((R0 - depth) / sigma))
                    frontfac = ((np.exp((-((R0 - depth) ** 2)) / (4.0 * (sigma ** 2))) * (sigma ** 0.565)) / (1.0 + 0.012 * R0)) * phi0
                    bracfac = 11.26 * D565 / sigma + ((0.157 + 11.26 * epsilon) / R0) * D1565
                    return frontfac * bracfac
                else:
                    return 0.0

            D = np.vectorize(D, excluded=["R0", "phi0", "epsilon", "sigma"])
            return D(depth, R0, phi0, epsilon, sigma)

        bortfeldParam = self.__bortfeldParamConstant
        bortfeldParam["R0"] = self.getRInterp(0.9, "distal")  # [mm]
        bortfeldParam["phi"] = self.getDInterp(
            self.__img.GetOrigin()[ft.ft_imgAnalyse._getAxesNumberNotUnity(self.__img)[0]]
        )  # [1/mm^2] initial fluence at the beginning of the profile (normalisation factor)
        bortfeldParam["E0"] = ((bortfeldParam["R0"] - ft.getExtent(self.__img)[ft.ft_imgAnalyse._getAxesNumberNotUnity(self.__img)[0]][0]) / bortfeldParam["alpha"]) ** (
            1 / bortfeldParam["p"]
        )  # [MeV] initial energy
        bortfeldParam["sigmaMono"] = (
            0.012 * (bortfeldParam["R0"] - ft.getExtent(self.__img)[ft.ft_imgAnalyse._getAxesNumberNotUnity(self.__img)[0]][0]) ** 0.935
        ) / 10  # [mm] width of Gaussian range straggling
        bortfeldParam["sigmaE0"] = 0.01 * bortfeldParam["E0"]  # [MeV] width of Gaussian energy spectrum
        bortfeldParam["sigma"] = np.sqrt(
            bortfeldParam["sigmaMono"] ** 2 + bortfeldParam["sigmaE0"] ** 2 * bortfeldParam["alpha"] ** 2 * bortfeldParam["p"] ** 2 * bortfeldParam["E0"] ** (2 * bortfeldParam["p"] - 2)
        )  # [mm] proton range dispersion

        bortfeldFitModel = Model(bortfeldEquation)
        bortfeldFit = bortfeldFitModel.fit(
            data=imgVal, depth=imgPos, R0=bortfeldParam["R0"], phi0=bortfeldParam["phi"], epsilon=bortfeldParam["epsilon"], sigma=bortfeldParam["sigma"], method="leastsq"
        )
        return bortfeldFit

    @property
    def displayInfo(self):
        """Display information about the Bragg peak analysis."""
        import fredtools as ft
        import numpy as np

        bortfeldFitParam = self.bortfeldFitParam
        print(f"### {__class__.__name__} ###")
        print("# max value [-]:            {:.2f}".format(np.nanmax(ft.vec(self.__img))))
        print("# D100 Interp/Bort [-]:     {:.2f}/{:.2f}".format(self.getDInterp(self.getRInterp(1, "D")), self.getDBort(self.getRBort(1, "D"))))
        print("# R10D Interp/Bort [mm]:    {:.2f}/{:.2f}".format(self.getRInterp(0.1, "D"), self.getRBort(0.1, "D")))
        print("# R80D Interp/Bort [mm]:    {:.2f}/{:.2f}".format(self.getRInterp(0.8, "D"), self.getRBort(0.8, "D")))
        print("# DFO9010 Interp/Bort [mm]: {:.2f}/{:.2f}".format(self.getDFOInterp(0.9, 0.1), self.getDFOBort(0.9, 0.1)))
        print("# FWHM Interp/Bort [mm]:    {:.2f}/{:.2f}".format(self.getWInterp(0.5), self.getWBort(0.5)))
        print("# Bortfeld fit chi-square [-]:                      {:.5f}".format(self.bortfeldFit.chisqr))
        print("# Bortfeld range (R0) [mm]:                         {:.2f}".format(bortfeldFitParam["R0_mm"]))
        print("# Bortfeld sigma of Gaussian range straggling [mm]: {:.2f}".format(bortfeldFitParam["sigmaMono_mm"]))
        print("# Bortfeld initial energy [MeV]:                    {:.2f}".format(bortfeldFitParam["E0_MeV"]))
        print("# Bortfeld sigma of Gaussian energy spectrum [MeV]: {:.2f}".format(bortfeldFitParam["sigmaE0_MeV"]))
        print("#" * len(f"### {__class__.__name__} ###"))

    @property
    def plot(self):
        """Simple plot of the Bragg peak and analysis methods."""
        import matplotlib.pyplot as plt
        import fredtools as ft

        fig, ax = plt.subplots(figsize=[20, 10])

        ax.plot(ft.pos(self.__img), ft.vec(self.__img), "r.", label="original profile")
        ax.plot(ft.pos(self.imgBort), ft.vec(self.imgBort), "b-", label="interpolated profile")
        ax.plot(ft.pos(self.imgInterp), ft.vec(self.imgInterp), "g-", label="Bortfeld fit profile")
        ax.grid()
        ax.legend()
        ax.set_xlabel("depth [$mm$]")
        ax.set_ylabel("signal [$-$]")
