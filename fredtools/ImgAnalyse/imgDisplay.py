from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def showSlice(ax: Axes, imgBack: SITKImage | None = None, imgFront: SITKImage | None = None, plane: str = "XY", point: PointLike | None = None, cmapBack: str | Colormap = "bone", cmapFront: str | Colormap = "jet", alphaFront: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.7, imgROIs: Iterable[SITKImage] | None = None, vmaxBack: float | None = None, vmaxFront: float | None = None, showLegend: bool = True, fontsize: NonNegativeInt = 8) -> AxesImage:
    """Display image slice in front of another image slice including contours.

    The function displays on `ax` a `plane` going through `point`
    of a 3D image describing the front image overlapped on a slice of another
    image describing the background. It forms a simple wrapper to
    matplotlib.pyplot.imshow allowing for a quick display of a slice from a
    3D image of signal (e.g. dose) and/or another image (e.g. CT).

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
        maximum value of the 3D background image will be used. (def. None)
    vmaxFront : scalar, optional
        Maximum value of the foreground dose map. If None then the
        maximum value of the 3D foreground image will be used. (def. None)
    alphaFront : float, optional
        Alpha value pf the transparency of the foreground image. (def. 0.7)
    showLegend : bool, optional
        Show legend of the ROI contour names if they exist. (def. True)
    fontsize : scalar, optional
        Basic font size to be used for ticks, labels, legend, etc. (def. 8)

    Returns
    -------
    matplotlib.image.AxesImage
        Foreground image (or background image if the foreground image
        is not given) attached to the axis `ax`.

    See Also
    --------
        showSlices: show three projections of a 3D image, also interactively.
        getSlice: get a 2D image slice from SimpleITK Image.

    Examples
    --------
    See `Jupyter notebook of Image Display Tutorial <https://github.com/jasqs/FREDtools/blob/main/examples/Image%20Display%20Tutorial.ipynb>`_.
    """
    import fredtools as ft
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import re
    import SimpleITK as sitk

    # set background of the axis to black
    ax.set_facecolor("black")

    # check if any of the image is given
    if not (imgBack or imgFront):
        error = AttributeError(f"At least one image, imgBack or imgFront, must be given.")
        _logger.error(error)
        raise error

    # determine point (def. mass centre of dose) if not given
    if point is None:
        if imgFront:
            point = ft.getMassCenter(imgFront)
        elif imgBack:
            point = ft.getMassCenter(imgBack)
        else:
            error = AttributeError(f"At least one image, imgBack or imgFront, must be given.")
            _logger.error(error)
            raise error

    # determine colormap
    try:
        cmapBackColormap = plt.get_cmap(cmapBack)
    except ValueError as e:
        _logger.error(f"Cannot recognise cmapBack colormap {cmapBack}.")
        raise e
    try:
        cmapFrontColormap = plt.get_cmap(cmapFront)
    except ValueError as e:
        _logger.error(f"Cannot recognise cmapFront colormap {cmapFront}.")
        raise e

    # determine vmax
    if imgBack and not vmaxBack:
        vmaxBack = float(ft.getStatistics(imgBack).GetMaximum())
    if imgFront and not vmaxFront:
        vmaxFront = float(ft.getStatistics(imgFront).GetMaximum())

    axesImage = None
    slBack = None
    slFront = None

    # show back slice image
    if imgBack:
        # check if imgBack is a 3D SimpleITK image
        ft._imgTypeChecker.isSITK_volume(imgBack)

        slBack = ft.getSlice(imgBack, point=point, plane=plane)
        axesImage = ax.imshow(sitk.GetArrayViewFromImage(slBack).squeeze(), cmap=cmapBackColormap, extent=ft.getExtMpl(slBack), vmax=vmaxBack)

    # show front slice image
    if imgFront:
        # check if imgFront is a 3D SimpleITK image
        ft._imgTypeChecker.isSITK_volume(imgFront)

        slFront = ft.getSlice(imgFront, point=point, plane=plane)
        axesImage = ax.imshow(sitk.GetArrayViewFromImage(slFront).squeeze(), cmap=cmapFrontColormap, extent=ft.getExtMpl(slFront), alpha=alphaFront, vmin=0, vmax=vmaxFront)

    if not axesImage:
        error = AttributeError(f"Cannot display the image slice.")
        _logger.error(error)
        raise

    # show ROIs slice
    if imgROIs:
        for imgROI in imgROIs if isinstance(imgROIs, Iterable) else [imgROIs]:
            ft._imgTypeChecker.isSITK_mask(imgROI, raiseError=True)

            # convert the floating mask to binary if needed
            if ft._imgTypeChecker.isSITK_maskFloating(imgROI):
                imgROI = ft.floatingToBinaryMask(imgROI, threshold=0.5)
                _logger.debug(f"The floating mask for '" + (imgROI.GetMetaData("ROIName") if "ROIName" in imgROI.GetMetaDataKeys() else "unknown") + "' contour was converted to a binary mask with threshold=0.5.")

            slROI = ft.getSlice(imgROI, point=point, plane=plane)
            if "ROIColor" in imgROI.GetMetaDataKeys():
                color = np.array(re.findall(r"\d+", imgROI.GetMetaData("ROIColor")), dtype="int") / 255
            else:
                color = np.array([0, 0, 1])
            if ft.getStatistics(slROI).GetMaximum() > 0:
                ax.contour(sitk.GetArrayViewFromImage(slROI).squeeze(), extent=ft.getExtMpl(slROI), levels=[0.5], colors=[color], linewidths=2, origin="upper")
                ax.plot([], color=color, label=imgROI.GetMetaData("ROIName") if "ROIName" in imgROI.GetMetaDataKeys() else "unknown")
        if len(ax.get_legend_handles_labels()[0]) > 0 and showLegend:
            ax.legend(fontsize=fontsize)

    # set  x/y limits to CT
    if slBack:
        ax.set_xlim(ft.getExtMpl(slBack)[0], ft.getExtMpl(slBack)[1])
        ax.set_ylim(ft.getExtMpl(slBack)[2], ft.getExtMpl(slBack)[3])
    elif slFront:
        ax.set_xlim(ft.getExtMpl(slFront)[0], ft.getExtMpl(slFront)[1])
        ax.set_ylim(ft.getExtMpl(slFront)[2], ft.getExtMpl(slFront)[3])
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

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
t
    Examples
    --------
    See `Jupyter notebook of Image Display Tutorial <https://github.com/jasqs/FREDtools/blob/main/examples/Image%20Display%20Tutorial.ipynb>`_.
    """

    def __init__(self, imgBack: SITKImage | None = None, imgFront: SITKImage | None = None, imgROIs: Iterable[SITKImage] | None = None, point: PointLike | None = None, DCOFront: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.1, cmapBack: str | Colormap = "bone", cmapFront: str | Colormap = "jet", figsize: Iterable[Numberic] = (15, 5)):
        import ipywidgets as ipyw
        from matplotlib import colorbar, colors
        import matplotlib.pyplot as plt
        from matplotlib import get_backend
        import fredtools as ft
        import SimpleITK as sitk
        import numpy as np
        from IPython.core.getipython import get_ipython

        self.imgBack = imgBack
        self.imgFront = imgFront
        self.imgROIs = imgROIs

        # check if any of the image is given
        if not (self.imgBack or self.imgFront):
            error = AttributeError(f"At least one image, imgBack or imgFront, must be given.")
            _logger.error(error)
            raise error

        # determine image slider
        if self.imgBack:
            self.imgSlider = self.imgBack
        elif self.imgFront:
            self.imgSlider = self.imgFront
        else:
            error = AttributeError(f"At least one image, imgBack or imgFront, must be given.")
            _logger.error(error)
            raise error

        # determine point
        if point is None:
            if self.imgFront:
                self.point = list(ft.getMassCenter(self.imgFront))
            elif self.imgBack:
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
        else:
            statFront = sitk.StatisticsImageFilter()

        if self.imgBack:
            statBack = ft.getStatistics(self.imgBack)
        else:
            statBack = sitk.StatisticsImageFilter()

        # # determine if interactive is possible (only jupyter)
        if get_backend() in ["widget", "ipympl"]:
            interactive = True
        else:
            interactive = False

        # prepare figure
        self.fig, self.axs = plt.subplots(ncols=3, nrows=1, figsize=figsize, gridspec_kw={"wspace": 0.2})
        if interactive:
            self.fig.canvas.toolbar_position = "bottom"  # type: ignore
            self.fig.canvas.header_visible = False  # type: ignore

        # determine colormap
        try:
            self.cmapBack = plt.get_cmap(cmapBack)
        except ValueError as e:
            _logger.error(f"Cannot recognise cmapBack colormap {cmapBack}.")
            raise e
        try:
            self.cmapFront = plt.get_cmap(cmapFront)
        except ValueError as e:
            _logger.error(f"Cannot recognise cmapFront colormap {cmapFront}.")
            raise e

        # make colorbar
        axCB = self.fig.add_axes((self.axs[2].get_position().x1 + self.axs[2].get_position().width * 0.1, self.axs[2].get_position().y0, self.axs[2].get_position().width * 0.05, self.axs[2].get_position().height))
        if self.imgFront:
            plCB = colorbar.ColorbarBase(axCB, cmap=self.cmapFront, norm=colors.Normalize(statFront.GetMinimum(), statFront.GetMaximum()))
            plCB.set_label("Front signal", labelpad=20, rotation=-90, fontsize=9)
        elif imgBack:
            plCB = colorbar.ColorbarBase(axCB, cmap=self.cmapBack, norm=colors.Normalize(statBack.GetMinimum(), statBack.GetMaximum()))
            plCB.set_label("Back signal", labelpad=20, rotation=-90, fontsize=9)
        axCB.tick_params(labelsize=8)

        if interactive:
            # Call to select slice plane
            self.sliderX = ipyw.FloatSlider(value=self.point[0],
                                            min=self.imgSlider.GetOrigin()[0],
                                            max=self.imgSlider.GetOrigin()[0] + self.imgSlider.GetSpacing()[0] * self.imgSlider.GetSize()[0],
                                            step=self.imgSlider.GetSpacing()[0],
                                            continuous_update=False,
                                            description="X [mm]:")
            ipyw.interact(self.showSliceAX1, X=self.sliderX)
            self.sliderY = ipyw.FloatSlider(value=self.point[1],
                                            min=self.imgSlider.GetOrigin()[1],
                                            max=self.imgSlider.GetOrigin()[1] + self.imgSlider.GetSpacing()[1] * self.imgSlider.GetSize()[1],
                                            step=self.imgSlider.GetSpacing()[1],
                                            continuous_update=False,
                                            description="Y [mm]:")
            ipyw.interact(self.showSliceAX2, Y=self.sliderY)
            self.sliderZ = ipyw.FloatSlider(value=self.point[2],
                                            min=self.imgSlider.GetOrigin()[2],
                                            max=self.imgSlider.GetOrigin()[2] + self.imgSlider.GetSpacing()[2] * self.imgSlider.GetSize()[2],
                                            step=self.imgSlider.GetSpacing()[2],
                                            continuous_update=False,
                                            description="Z [mm]:")
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
        ft.showSlice(self.axs[0],
                     plane="XY",
                     point=self.point,
                     imgBack=self.imgBack,
                     imgFront=self.imgFront,
                     imgROIs=self.imgROIs,
                     showLegend=True,
                     cmapFront=self.cmapFront,
                     cmapBack=self.cmapBack,
                     fontsize=8)
        self.replotPointLines()
        self.fig.canvas.draw()

    def showSliceAX1(self, X):
        import fredtools as ft

        self.point[0] = X
        self.removeArtist(self.axs[1])
        self.axs[1].clear()
        ft.showSlice(self.axs[1],
                     plane="ZY",
                     point=self.point,
                     imgBack=self.imgBack,
                     imgFront=self.imgFront,
                     imgROIs=self.imgROIs,
                     showLegend=True,
                     cmapFront=self.cmapFront,
                     cmapBack=self.cmapBack,
                     fontsize=8)
        self.replotPointLines()
        self.fig.canvas.draw()

    def showSliceAX2(self, Y):
        import fredtools as ft

        self.point[1] = Y
        self.removeArtist(self.axs[2])
        self.axs[2].clear()
        ft.showSlice(self.axs[2],
                     plane="X-Z",
                     point=self.point,
                     imgBack=self.imgBack,
                     imgFront=self.imgFront,
                     imgROIs=self.imgROIs,
                     showLegend=True,
                     cmapFront=self.cmapFront,
                     cmapBack=self.cmapBack,
                     fontsize=8)
        self.replotPointLines()
        self.fig.canvas.draw()
