def showSlice(
    ax,
    imgBack=None,
    imgFront=None,
    plane="XY",
    point=None,
    cmapBack="bone",
    cmapFront="jet",
    alphaFront=0.7,
    imgROIs=None,
    vmaxBack=None,
    vmaxFront=None,
    showLegend=True,
    fontsize=8,
    raiseWarning=True,
):
    """Display image slice in front of an another image slice including contours.

    The function displays on `ax` a `plane` going through `point`
    of a 3D image describing front image overlapped on a slice of an another
    image describing the background. Basically, it forms a simple wrapper to
    matplotlib.pyplot.imshow allowing for a quick display of a slice from a
    3D image of singal (e.g. dose) and/or an another image (e.g CT).

    Usually `imgBack` describes the CT and `imgFront` describes the dose or
    other signal. At least one variable, `imgBack` or `imgFront` must be given.

    Parameters
    ----------
    ax : AxesSubplot
        Axis to plot the image on.
    imgBack : SimpleITK Image, optional
        Object of a SimpleITK 3D image describing the background
        image. (def. None)
    imgFront : SimpleITK Image, optional
        Object of a SimpleITK 3D image describing the foreground
        image. (def. None)
    plane : str, optional
        Plane to generate the 2D slice given as a string.
        See fredtools.getSlice for more details. (def. 'XY')
    point : array_like, optional
        3D point to generate the 2D slice through. If None
        then the centre of mass of the 3D foreground image (or background
        image if the foreground image is not given) will be used. (def. None)
    cmapBack : string or matplotlib.colors.Colormap, optional
        Colormap to display the background image slice. (def. 'bone')
    cmapFront : string or matplotlib.colors.Colormap, optional
        Colormap to display the foreground image slice. (def. 'jet')
    imgROIs : SimpleITK Image or list of SimpleITK Images, optional
        An Instance of a SimpleITK image or list of instances of
        SimpleITK image objects describing a 3D mask. (def. None)
    vmaxBack : scalar, optional
        Maximum value of the background image. If None then the
        maximum value of 3D background image will be used. (def. None)
    vmaxFront : scalar, optional
        Maximum value of the foreground dose map. If None then the
        maximum value of 3D foreground image will be used. (def. None)
    alphaFront : float, optional
        Alpha value pf the transparency of the foreground image. (def. 0.7)
    showLegend : bool, optional
        Show legend of the ROI contour names if they exist. (def. True)
    fontsize : scalar, optional
        Basic font size to be used for ticks, labels, legend, etc. (def. 8)
    raiseWarning : bool, optional
        Raise warnings. (def. True)

    Returns
    -------
    matplotlib.image.AxesImage
        Foreground image (or background image if the foreground image
        is not given) attached to the axis `ax`.

    See Also
    --------
        showSlices: show three projections of a 3D image, also interactively.
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
    if not (imgBack or imgFront):
        raise AttributeError(f"At least one image, imgBack or imgFront, must be given.")

    # determine point (def. mass centre of dose) if not given
    if point is None:
        if imgFront:
            point = ft.getMassCenter(imgFront)
        else:
            point = ft.getMassCenter(imgBack)

    # determine colormap
    if isinstance(cmapBack, mpl.colors.LinearSegmentedColormap):
        cmapBack = cmapBack
    elif isinstance(cmapBack, str):
        cmapBack = mpl.cm.get_cmap(cmapBack)
    else:
        raise ValueError(f"Cannot recognise cmapBack colormap {cmapBack}.")
    if isinstance(cmapFront, mpl.colors.LinearSegmentedColormap):
        cmapFront = cmapFront
    elif isinstance(cmapFront, str):
        cmapFront = mpl.cm.get_cmap(cmapFront)
    else:
        raise ValueError(f"Cannot recognise cmapFront colormap {cmapFront}.")

    # determine vmax
    if imgBack and not vmaxBack:
        vmaxBack = ft.getStatistics(imgBack).GetMaximum()
    if imgFront and not vmaxFront:
        vmaxFront = ft.getStatistics(imgFront).GetMaximum()

    # show back slice image
    if imgBack:
        # check if imgBack is a 3D SimpleITK image
        ft._isSITK_volume(imgBack)

        slBack = ft.getSlice(imgBack, point=point, plane=plane, raiseWarning=raiseWarning)
        axesImage = ax.imshow(ft.arr(slBack), cmap=cmapBack, extent=ft.getExtMpl(slBack), vmax=vmaxBack)

    # show front slice image
    if imgFront:
        # check if imgFront is a 3D SimpleITK image
        ft._isSITK_volume(imgFront)

        slFront = ft.getSlice(imgFront, point=point, plane=plane, raiseWarning=raiseWarning)
        axesImage = ax.imshow(ft.arr(slFront), cmap=cmapFront, extent=ft.getExtMpl(slFront), alpha=alphaFront, vmin=0, vmax=vmaxFront)

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

                # if "ROIName" in imgROI.GetMetaDataKeys():
                #     name = imgROI.GetMetaData("ROIName")
                # else:
                #     name = "unknown"
                ax.plot([], color=color, label=imgROI.GetMetaData("ROIName") if "ROIName" in imgROI.GetMetaDataKeys() else "unknown")
                # plROI.collections[0].set_label(name)
        if len(ax.get_legend_handles_labels()[0]) > 0 and showLegend:
            ax.legend(fontsize=fontsize)

    # set  x/y limits to CT
    if imgBack:
        ax.set_xlim(ft.getExtMpl(slBack)[0], ft.getExtMpl(slBack)[1])
        ax.set_ylim(ft.getExtMpl(slBack)[2], ft.getExtMpl(slBack)[3])
    else:
        ax.set_xlim(ft.getExtMpl(slFront)[0], ft.getExtMpl(slFront)[1])
        ax.set_ylim(ft.getExtMpl(slFront)[2], ft.getExtMpl(slFront)[3])

    # set axis labels
    planeSimple = re.sub("[-+]", "", plane)
    ax.set_xlabel(planeSimple[0] + " [$mm$]", fontsize=fontsize + 1)
    ax.set_ylabel(planeSimple[1] + " [$mm$]", fontsize=fontsize + 1)

    # set font sizes
    ax.tick_params(labelsize=fontsize)

    return axesImage


class showSlices:
    """Class to display three projections of 3D image slices on an another 3D image slices including contours.

    The class creates a figure with three axes and displays
    three projections (planes), 'XY', 'ZY' and 'X-Z' (reversed Z),
    going through `point`, of a 3D image describing foreground
    overlapped on a background image. The class can display in
    an interactive mode exploiting ipywidgets functionality,
    allowing to move slices with a mouse well or move slices to
    the point when the mouse button is pressed. All those interactive
    features work with Shift pressed.

    Usually `imgBack` describes the CT and `imgFront` describes the dose or
    other signal. At least one variable, `imgBack` or `imgFront` must be given.

    Parameters
    ----------
    imgBack : SimpleITK Image, optional
        Object of a 3D SimpleITK image describing the background
        image. (e.g. CT). (def. None)
    imgFront : SimpleITK Image, optional
        Object of a 3D SimpleITK image describing the foreground
        image. (e.g. dose or gamma index map). (def. None)
    imgROIs : SimpleITK Image or list of SimpleITK Images, optional
        An Instance of a SimpleITK image or list of instances of
        SimpleITK image objects describing a 3D mask. (def. None)
    point : array_like, optional
        3D point to generate the 2D slice through. If None
        then the centre of mass of the 3D foreground image (or the background
        image if the foreground is not given) will be used. (def. None)
    DCOFront : float, optional
        Dose cut-off. The fraction of the maximum value of the foreground
        image below which the data will not be displayed. (def. 0.1)
    cmapBack : string or matplotlib.colors.Colormap, optional
        Colormap to display the background image slice. (def. 'bone')
    cmapFront : string or matplotlib.colors.Colormap, optional
        Colormap to display the foreground image slice. (def. 'jet')
    figsize : 2-element list, optional
        Width and height of the figure in inches. (def. [15, 5])
    interactive : bool, optional
        Display in interactive mode using ipwidgets.
        Works only in jupyter. (def. True)

    Examples
    --------
    See `Jupyter notebook of Image Display Tutorial <https://github.com/jasqs/FREDtools/blob/main/examples/Image%20Display%20Tutorial.ipynb>`_.
    """

    def __init__(self, imgBack=None, imgFront=None, imgROIs=None, point=None, DCOFront=0.1, cmapBack="bone", cmapFront="jet", figsize=[15, 5], interactive=True):
        import ipywidgets as ipyw
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import fredtools as ft
        import SimpleITK as sitk
        import numpy as np
        from IPython import get_ipython

        self.imgBack = imgBack
        self.imgFront = imgFront
        self.imgROIs = imgROIs

        # check if any of the image is given
        if not (self.imgBack or self.imgFront):
            raise AttributeError(f"At least one image, imgBack or imgFront, must be given.")

        # determine image slider
        if self.imgBack:
            self.imgSlider = self.imgBack
        else:
            self.imgSlider = self.imgFront

        # determine point
        if point is None:
            if imgFront:
                self.point = list(ft.getMassCenter(self.imgFront))
            else:
                self.point = list(ft.getMassCenter(self.imgBack))
        else:
            self.point = list(point)

        # check if point is correct
        if len(self.point) != 3:
            raise ValueError(f"The `point` must be a 3-element vector and is {self.point}.")

        # set dose threshold
        if self.imgFront:
            statFront = ft.getStatistics(self.imgFront)
            self.imgFront = sitk.Threshold(self.imgFront, lower=statFront.GetMaximum() * DCOFront, upper=statFront.GetMaximum() * 10, outsideValue=np.nan)
        if self.imgBack:
            statBack = ft.getStatistics(self.imgBack)

        # determine if interactive is possible (only jupyter)
        if interactive and ft._checkJupyterMode():
            ipython = get_ipython()
            # ipython.run_line_magic("matplotlib widget")
            ipython.run_line_magic("matplotlib", "widget")
        elif not interactive and ft._checkJupyterMode():
            ipython = get_ipython()
            # ipython.run_line_magic("matplotlib inline")
            ipython.run_line_magic("matplotlib", "inline")
        else:
            interactive = False

        # prepare figure
        self.fig, self.axs = plt.subplots(ncols=3, nrows=1, figsize=figsize, gridspec_kw={"wspace": 0.2})
        if interactive:
            self.fig.canvas.toolbar_position = "bottom"
            self.fig.canvas.header_visible = False

        # determine colormap
        if isinstance(cmapBack, mpl.colors.LinearSegmentedColormap):
            self.cmapBack = cmapBack
        elif isinstance(cmapBack, str):
            self.cmapBack = mpl.cm.get_cmap(cmapBack)
        else:
            raise ValueError(f"Cannot recognise cmapBack colormap {self.cmapBack}.")
        if isinstance(cmapFront, mpl.colors.LinearSegmentedColormap):
            self.cmapFront = cmapFront
        elif isinstance(cmapFront, str):
            self.cmapFront = mpl.cm.get_cmap(cmapFront)
        else:
            raise ValueError(f"Cannot recognise cmapFront colormap {self.cmapFront}.")

        # make colorbar
        axCB = self.fig.add_axes(
            [self.axs[2].get_position().x1 + self.axs[2].get_position().width * 0.1, self.axs[2].get_position().y0, self.axs[2].get_position().width * 0.05, self.axs[2].get_position().height]
        )
        if self.imgFront:
            plCB = mpl.colorbar.ColorbarBase(axCB, cmap=self.cmapFront, norm=mpl.colors.Normalize(statFront.GetMinimum(), statFront.GetMaximum()))
            plCB.set_label("Front signal", labelpad=20, rotation=-90, fontsize=9)
        elif imgBack:
            plCB = mpl.colorbar.ColorbarBase(axCB, cmap=self.cmapBack, norm=mpl.colors.Normalize(statBack.GetMinimum(), statBack.GetMaximum()))
            plCB.set_label("Back signal", labelpad=20, rotation=-90, fontsize=9)
        axCB.tick_params(labelsize=8)

        if interactive:
            # Call to select slice plane
            self.sliderX = ipyw.FloatSlider(
                value=self.point[0],
                min=self.imgSlider.GetOrigin()[0],
                max=self.imgSlider.GetOrigin()[0] + self.imgSlider.GetSpacing()[0] * self.imgSlider.GetSize()[0],
                step=self.imgSlider.GetSpacing()[0],
                continuous_update=False,
                description="X [mm]:",
            )
            ipyw.interact(self.showSliceAX1, X=self.sliderX)
            self.sliderY = ipyw.FloatSlider(
                value=self.point[1],
                min=self.imgSlider.GetOrigin()[1],
                max=self.imgSlider.GetOrigin()[1] + self.imgSlider.GetSpacing()[1] * self.imgSlider.GetSize()[1],
                step=self.imgSlider.GetSpacing()[1],
                continuous_update=False,
                description="Y [mm]:",
            )
            ipyw.interact(self.showSliceAX2, Y=self.sliderY)
            self.sliderZ = ipyw.FloatSlider(
                value=self.point[2],
                min=self.imgSlider.GetOrigin()[2],
                max=self.imgSlider.GetOrigin()[2] + self.imgSlider.GetSpacing()[2] * self.imgSlider.GetSize()[2],
                step=self.imgSlider.GetSpacing()[2],
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
                    self.sliderX.value = self.point[0] + self.imgSlider.GetSpacing()[0]
                elif event.button == "down":
                    self.sliderX.value = self.point[0] - self.imgSlider.GetSpacing()[0]
            if event.inaxes == self.axs[2]:
                if event.button == "up":
                    self.sliderY.value = self.point[1] + self.imgSlider.GetSpacing()[1]
                elif event.button == "down":
                    self.sliderY.value = self.point[1] - self.imgSlider.GetSpacing()[1]
            if event.inaxes == self.axs[0]:
                if event.button == "up":
                    self.sliderZ.value = self.point[2] + self.imgSlider.GetSpacing()[2]
                elif event.button == "down":
                    self.sliderZ.value = self.point[2] - self.imgSlider.GetSpacing()[2]

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
        self.axs[0].clear()
        ft.showSlice(
            self.axs[0],
            plane="XY",
            point=self.point,
            imgBack=self.imgBack,
            imgFront=self.imgFront,
            imgROIs=self.imgROIs,
            showLegend=True,
            cmapFront=self.cmapFront,
            cmapBack=self.cmapBack,
            fontsize=8,
            raiseWarning=False,
        )
        self.replotPointLines()
        self.fig.canvas.draw()

    def showSliceAX1(self, X):
        import fredtools as ft

        self.point[0] = X
        self.removeArtist(self.axs[1])
        self.axs[1].clear()
        ft.showSlice(
            self.axs[1],
            plane="ZY",
            point=self.point,
            imgBack=self.imgBack,
            imgFront=self.imgFront,
            imgROIs=self.imgROIs,
            showLegend=True,
            cmapFront=self.cmapFront,
            cmapBack=self.cmapBack,
            fontsize=8,
            raiseWarning=False,
        )
        self.replotPointLines()
        self.fig.canvas.draw()

    def showSliceAX2(self, Y):
        import fredtools as ft

        self.point[1] = Y
        self.removeArtist(self.axs[2])
        self.axs[2].clear()
        ft.showSlice(
            self.axs[2],
            plane="X-Z",
            point=self.point,
            imgBack=self.imgBack,
            imgFront=self.imgFront,
            imgROIs=self.imgROIs,
            showLegend=True,
            cmapFront=self.cmapFront,
            cmapBack=self.cmapBack,
            fontsize=8,
            raiseWarning=False,
        )
        self.replotPointLines()
        self.fig.canvas.draw()