def showSlice(ax, imgA=None, imgB=None, plane="XY", point=None, imgCmap="jet", imgROIs=None, doseVmax=None, showLegend=True, fontsize=8, raiseWarning=True):
    """Display dose slice on a CT slice including contours.

    The function displays on `ax` a `plane` going through `point`
    of a 3D image describing dose overlapped on an image of a CT.
    Basically, it forms a simple wrapper to matplotlib.pyplot.imshow
    allowing for a quick display of a slice from a 3D image of dose
    and/or CT.

    It does not matter which attribute, `imgA` or `imgB`, will describe
    dose od CT. By default, the image of the integer type is treated as the CT
    and the image of the float type is treated as the dose. The CT image is then
    displayed in 'bone' colormap and the dose image in `imgCmap` colormap.
    If only one image (`imgA` or `imgB`) of is given, then it is treated as
    a CT image if it is of the integer type, or as a dose image if it is of
    the float type. Thus, the function allows to display only a single image
    of dose or CT. At least one variable, `imgA` or `imgB` must be given.

    Parameters
    ----------
    ax : AxesSubplot
        Axis to plot the image on.
    imgA : SimpleITK Image, optional
        Object of a SimpleITK 3D image. (def. None)
    imgB : SimpleITK Image, optional
        Object of a SimpleITK 3D image. (def. None)
    plane : str, optional
        Plane to generate the 2D slice given as a string.
        See fredtools.getSlice for more details. (def. 'XY')
    point : array_like, optional
        3D point to generate the 2D slice through. If None
        then the centre of mass of the 3D dose image (or CT image if
        dose was not given) will be used (def. None)
    imgCmap : string or matplotlib.colors.Colormap, optional
        Colormap to display dose image slice. (def. 'jet')
    imgROIs : SimpleITK Image or list of SimpleITK Images, optional
        An Instance of a SimpleITK image or list of instances of
        SimpleITK image objects describing a 3D mask. (def. None)
    doseVmax : scalar, optional
        Maximum value on dose map. If None then the
        maximum value of 3D dose image will be used (def. None)
    showLegend : bool, optional
        Show legend of the ROI contour names if they exist. (def. True)
    fontsize : scalar, optional
        Basic font size to be used for ticks, labels, legend, etc. (def. 8)
    raiseWarning : bool, optional
        Raise warnings. (def. True)

    Returns
    -------
    matplotlib.image.AxesImage
        Dose image (or CT image if dose was not given) attached to the axes `ax`.

    See Also
    --------
        showSlices: show three projections of an 3D image overlapped on CT, also interactively.
        getSlice: get 2D image slice from SimpleITK Image.

    Examples
    --------
    See `Jupyter notebook of Image Display Tutorial <https://github.com/jasqs/FREDtools/blob/main/examples/Image%20Display%20Tutorial.ipynb>`_.
    """
    import fredtools as ft
    import numpy as np
    import matplotlib as mpl
    import re

    # set background of the axis to black
    ax.set_facecolor("black")

    # check if any of the image is given
    if not (imgA or imgB):
        raise AttributeError(f"At least one image, imgA or imgB, must be given.")

    # determine which image is a CT and which is a dose
    if ft._isSITK(imgA) and not ft._isSITK(imgB):
        if "integer" in imgA.GetPixelIDTypeAsString():
            imgCT = imgA
            imgDose = imgB
        elif "float" in imgA.GetPixelIDTypeAsString():
            imgCT = imgB
            imgDose = imgA
    elif ft._isSITK(imgB) and not ft._isSITK(imgA):
        if "integer" in imgB.GetPixelIDTypeAsString():
            imgCT = imgB
            imgDose = imgA
        elif "float" in imgB.GetPixelIDTypeAsString():
            imgCT = imgA
            imgDose = imgB
    elif ft._isSITK(imgA) and ft._isSITK(imgB):
        if "integer" in imgA.GetPixelIDTypeAsString():
            imgCT = imgA
            imgDose = imgB
        elif "integer" in imgB.GetPixelIDTypeAsString():
            imgCT = imgB
            imgDose = imgA
        else:
            imgCT = imgA
            imgDose = imgB

    # determine point (def. mass centre of dose)
    if not point:
        if imgDose:
            point = ft.getMassCenter(imgDose)
        else:
            point = ft.getMassCenter(imgCT)

    # determine colormap
    if isinstance(imgCmap, mpl.colors.LinearSegmentedColormap):
        imgCmap = imgCmap
    elif isinstance(imgCmap, str):
        imgCmap = mpl.cm.get_cmap(imgCmap)
    else:
        raise ValueError(f"Cannot recognise colormap {imgCmap}.")

    # show CT slice
    if imgCT:
        # check if imgCT is a 3D SimpleITK image
        ft._isSITK_volume(imgCT)

        slCT = ft.getSlice(imgCT, point=point, plane=plane, raiseWarning=raiseWarning)
        axesImage = ax.imshow(ft.arr(slCT), cmap="bone", extent=ft.getExtMpl(slCT))

    # show Dose slice
    if imgDose:
        # check if imgDose is a 3D SimpleITK image
        ft._isSITK_volume(imgDose)

        # use the maximum value of the 3D dose image as vmax id no doseVmax given
        if not doseVmax:
            statDose = ft.getStatistics(imgDose)
            doseVmax = statDose.GetMaximum()

        slDose = ft.getSlice(imgDose, point=point, plane=plane, raiseWarning=raiseWarning)
        axesImage = ax.imshow(ft.arr(slDose), cmap=imgCmap, extent=ft.getExtMpl(slDose), alpha=0.7, vmin=0, vmax=doseVmax)

    # show ROIs slice
    if imgROIs:
        for imgROI in imgROIs if isinstance(imgROIs, list) else [imgROIs]:
            slROI = ft.getSlice(imgROI, point=point, plane=plane, raiseWarning=raiseWarning)
            if "ROIColor" in imgROI.GetMetaDataKeys():
                color = np.array(re.findall("\d+", imgROI.GetMetaData("ROIColor")), dtype="int") / 255
            else:
                color = np.array([0, 0, 1])
            if ft.getStatistics(slROI).GetMaximum() > 0:
                plROI = ax.contour(ft.arr(slROI), extent=ft.getExtMpl(slROI), colors=[color], linewidths=1, origin="upper")
                if "ROIName" in imgROI.GetMetaDataKeys():
                    name = imgROI.GetMetaData("ROIName")
                else:
                    name = "unknown"
                plROI.collections[0].set_label(name)
        if len(ax.get_legend_handles_labels()[0]) > 0 and showLegend:
            ax.legend(fontsize=fontsize)

    # set axis labels
    planeSimple = re.sub("[-+]", "", plane)
    ax.set_xlabel(planeSimple[0] + " [$mm$]", fontsize=fontsize + 1)
    ax.set_ylabel(planeSimple[1] + " [$mm$]", fontsize=fontsize + 1)

    # set font sizes
    ax.tick_params(labelsize=fontsize)

    return axesImage


class showSlices:
    """Class to display three projections of dose slices on CT slices including contours.

    The class creates a figure with three axes and displays
    three projections (planes), 'XY', 'ZY' and 'X-Z' (reversed Z),
    going through `point`, of a 3D image describing dose
    overlapped on an image of a CT. The class can display in
    an interactive mode exploiting ipywidgets functionality,
    allowing to move slices with a mouse well or move slices to
    the point when the mouse button is pressed. All those interactive
    features work with Shift pressed.

    Parameters
    ----------
    imgCT : SimpleITK Image
        Object of a 3D SimpleITK image describing a CT.
    imgDose : SimpleITK Image
        Object of a 3D SimpleITK image describing dose or any
        other quantity distribution.
    imgROIs : SimpleITK Image or list of SimpleITK Images, optional
        An Instance of a SimpleITK image or list of instances of
        SimpleITK image objects describing a 3D mask. (def. None)
    point : array_like, optional
        3D point to generate the 2D slice through. If None
        then the centre of mass of the 3D dose image will
        be used. (def. None)
    DCO : float, optional
        Dose cut-off. The fraction of the maximum value of the dose
        image below which the data will not be displayed. (def. 0.1)
    figsize : 2-element list
        Width and height of the figure in inches. (def. [15, 5])
    imgCmap : string or matplotlib.colors.Colormap, optional
        Colormap to display dose image slice. (def. 'jet')
    interactive : bool, optional
        Display in interactive mode using ipwidgets.
        Works only in jupyter. (def. True)

    Examples
    --------
    See `Jupyter notebook of Image Display Tutorial <https://github.com/jasqs/FREDtools/blob/main/examples/Image%20Display%20Tutorial.ipynb>`_.
    """

    def __init__(self, imgCT, imgDose, imgROIs=None, point=None, DCO=0.1, figsize=[15, 5], imgCmap="jet", interactive=True):
        import ipywidgets as ipyw
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import fredtools as ft
        import SimpleITK as sitk
        import numpy as np
        from IPython import get_ipython

        self.imgCT = imgCT
        self.imgDose = imgDose

        self.imgROIs = imgROIs

        # determine point
        if not point:
            self.point = list(ft.getMassCenter(self.imgDose))
        else:
            self.point = list(point)

        # check if point is correct
        if len(self.point) != 3:
            raise ValueError(f"The `point` must be a 3-element vector and is {self.point}.")

        # set dose threshold
        statDose = ft.getStatistics(self.imgDose)
        self.imgDose = sitk.Threshold(self.imgDose, lower=statDose.GetMaximum() * DCO, upper=1e5, outsideValue=np.nan)

        # determine if interactive is possible (only jupyter)
        if interactive and ft._checkJupyterMode():
            ipython = get_ipython()
            ipython.magic("matplotlib widget")
        elif not interactive and ft._checkJupyterMode():
            ipython = get_ipython()
            ipython.magic("matplotlib inline")
        else:
            interactive = False

        # prepare figure
        self.fig, self.axs = plt.subplots(ncols=3, nrows=1, figsize=figsize, gridspec_kw={"wspace": 0.2})
        if interactive:
            self.fig.canvas.toolbar_position = "bottom"
            self.fig.canvas.header_visible = False

        # determine colormap
        if isinstance(imgCmap, mpl.colors.LinearSegmentedColormap):
            self.imgCmap = imgCmap
        elif isinstance(imgCmap, str):
            self.imgCmap = mpl.cm.get_cmap(imgCmap)
        else:
            raise ValueError(f"Cannot recognise colormap {imgCmap}.")

        # make colorbar
        axCB = self.fig.add_axes(
            [self.axs[2].get_position().x1 + self.axs[2].get_position().width * 0.1, self.axs[2].get_position().y0, self.axs[2].get_position().width * 0.05, self.axs[2].get_position().height]
        )
        plCB = mpl.colorbar.ColorbarBase(axCB, cmap=self.imgCmap, norm=mpl.colors.Normalize(0, statDose.GetMaximum()))
        plCB.set_label("Dose [$Gy$ or $Gy(RBE)$]", labelpad=20, rotation=-90, fontsize=9)
        axCB.tick_params(labelsize=8)

        if interactive:
            # Call to select slice plane
            self.sliderX = ipyw.FloatSlider(
                value=self.point[0],
                min=self.imgCT.GetOrigin()[0],
                max=self.imgCT.GetOrigin()[0] + self.imgCT.GetSpacing()[0] * self.imgCT.GetSize()[0],
                step=self.imgCT.GetSpacing()[0],
                continuous_update=False,
                description="X [mm]:",
            )
            ipyw.interact(self.showSliceAX1, X=self.sliderX)
            self.sliderY = ipyw.FloatSlider(
                value=self.point[1],
                min=self.imgCT.GetOrigin()[1],
                max=self.imgCT.GetOrigin()[1] + self.imgCT.GetSpacing()[1] * self.imgCT.GetSize()[1],
                step=self.imgCT.GetSpacing()[1],
                continuous_update=False,
                description="Y [mm]:",
            )
            ipyw.interact(self.showSliceAX2, Y=self.sliderY)
            self.sliderZ = ipyw.FloatSlider(
                value=self.point[2],
                min=self.imgCT.GetOrigin()[2],
                max=self.imgCT.GetOrigin()[2] + self.imgCT.GetSpacing()[2] * self.imgCT.GetSize()[2],
                step=self.imgCT.GetSpacing()[2],
                continuous_update=False,
                description="Z [mm]:",
            )
            ipyw.interact(self.showSliceAX0, Z=self.sliderZ)

            # scroll event to move slices
            self.sliderX.observe(self.scrollEvent, "value")
            self.fig.canvas.mpl_connect("scroll_event", self.scrollEvent)
            self.fig.canvas.mpl_connect("key_press_event", self.scrollEventShiftPress)
            self.fig.canvas.mpl_connect("key_release_event", self.scrollEventShiftRelease)
            self.fig.canvas.mpl_connect("button_press_event", self.mouseButtomPressEvent)
            self.scrollEventShift = False
        else:
            self.showSliceAX1(X=self.point[0])
            self.showSliceAX2(Y=self.point[1])
            self.showSliceAX0(Z=self.point[2])

    def scrollEventShiftPress(self, event):
        if event.key == "shift":
            self.scrollEventShift = True

    def scrollEventShiftRelease(self, event):
        if event.key == "shift":
            self.scrollEventShift = False

    def mouseButtomPressEvent(self, event):
        if self.scrollEventShift:
            if event.inaxes == self.axs[0] and event.button == 1:
                self.sliderY.value = event.ydata
                self.sliderX.value = event.xdata
            if event.inaxes == self.axs[1] and event.button == 1:
                self.sliderZ.value = event.xdata
                self.sliderY.value = event.ydata
            if event.inaxes == self.axs[2] and event.button == 1:
                self.sliderZ.value = event.ydata
                self.sliderX.value = event.xdata

    def scrollEvent(self, event):
        if self.scrollEventShift:
            if event.inaxes == self.axs[1]:
                if event.button == "up":
                    self.sliderX.value = self.point[0] + self.imgCT.GetSpacing()[0]
                elif event.button == "down":
                    self.sliderX.value = self.point[0] - self.imgCT.GetSpacing()[0]
            if event.inaxes == self.axs[2]:
                if event.button == "up":
                    self.sliderY.value = self.point[1] + self.imgCT.GetSpacing()[1]
                elif event.button == "down":
                    self.sliderY.value = self.point[1] - self.imgCT.GetSpacing()[1]
            if event.inaxes == self.axs[0]:
                if event.button == "up":
                    self.sliderZ.value = self.point[2] + self.imgCT.GetSpacing()[2]
                elif event.button == "down":
                    self.sliderZ.value = self.point[2] - self.imgCT.GetSpacing()[2]

    def removeArtist(self, ax):
        for artist in ax.lines + ax.collections:
            artist.remove()
        ax.legend_ = None

    def replotPointLines(self):
        for artist in self.axs[0].lines + self.axs[1].lines + self.axs[2].lines:
            artist.remove()
        self.axs[0].axvline(self.point[0])
        self.axs[0].axhline(self.point[1])
        self.axs[1].axvline(self.point[2])
        self.axs[1].axhline(self.point[1])
        self.axs[2].axvline(self.point[0])
        self.axs[2].axhline(self.point[2])

    def showSliceAX0(self, Z):
        import fredtools as ft

        self.point[2] = Z
        self.removeArtist(self.axs[0])
        ft.showSlice(self.axs[0], plane="XY", point=self.point, imgA=self.imgCT, imgB=self.imgDose, imgROIs=self.imgROIs, showLegend=True, imgCmap=self.imgCmap, fontsize=8, raiseWarning=False)
        self.replotPointLines()

    def showSliceAX1(self, X):
        import fredtools as ft

        self.point[0] = X
        self.removeArtist(self.axs[1])
        ft.showSlice(self.axs[1], plane="ZY", point=self.point, imgA=self.imgCT, imgB=self.imgDose, imgROIs=self.imgROIs, showLegend=True, imgCmap=self.imgCmap, fontsize=8, raiseWarning=False)
        self.replotPointLines()

    def showSliceAX2(self, Y):
        import fredtools as ft

        self.point[1] = Y
        self.removeArtist(self.axs[2])
        ft.showSlice(self.axs[2], plane="X-Z", point=self.point, imgA=self.imgCT, imgB=self.imgDose, imgROIs=self.imgROIs, showLegend=True, imgCmap=self.imgCmap, fontsize=8, raiseWarning=False)
        self.replotPointLines()
