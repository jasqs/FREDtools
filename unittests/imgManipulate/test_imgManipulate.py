import unittest
import SimpleITK as sitk
import fredtools as ft
import numpy as np
import pandas as pd
import shapely as sph


def _higherIndexValue(img, point, outsideValue=np.nan):
    """Value of the voxel of `img` that the nearest-neighbour interpolation takes at `point`, with the rule written out:
    a position lying exactly halfway between two voxels takes the voxel with the higher index (`outsideValue` outside)."""
    index = np.floor(np.array(img.TransformPhysicalPointToContinuousIndex([float(x) for x in point])) + 0.5 + 1e-6).astype(int)
    if np.any(index < 0) or np.any(index >= np.array(img.GetSize())):
        return outsideValue
    return float(sitk.GetArrayViewFromImage(img)[tuple(index[::-1])])


def _countHalfway(img, imgRes, transform=None):
    """Number of voxels of `imgRes` whose centre lies exactly halfway between two voxels of `img` along any axis."""
    count = 0
    for index in np.ndindex(*imgRes.GetSize()):
        point = imgRes.TransformIndexToPhysicalPoint([int(i) for i in index])
        if transform is not None:
            point = transform.TransformPoint(point)
        continuousIndex = np.array(img.TransformPhysicalPointToContinuousIndex(point))
        count += bool(np.any(np.abs(continuousIndex - np.floor(continuousIndex) - 0.5) < 1e-6))
    return count


class test_mapStructToImg(unittest.TestCase):

    def setUp(self):
        self.img = ft.readMHD("unittests/testData/MHDImages/img3D.mhd")
        self.RSfileName = ft.sortDicoms("unittests/testData/TPSDicoms/TPSPlan/").RSfileNames

    def test_mapStructToImg(self):
        imgROI = ft.mapStructToImg(self.img, self.RSfileName, "testStuct_SphHoleDet", displayInfo=True)
        self.assertAlmostEqual(ft.getStatistics(imgROI).GetSum()*np.prod(imgROI.GetSpacing()), 130987, places=-1)

    def test_mapStructToImg_structureWithHole(self):
        """'External-body' is 'External' minus 'BODY' (a shell with a hole), so its mask must be the difference of their masks.

        On this 3 mm grid the earlier flood fill left about half of 'External-body' empty (1101 instead of 2130 cm3)."""
        imgROIs = {structName: ft.mapStructToImg(self.img, self.RSfileName, structName) for structName in ("External-body", "External", "BODY")}
        volumeDifference = ft.getStructVolume(imgROIs["External"]) - ft.getStructVolume(imgROIs["BODY"])
        self.assertAlmostEqual(ft.getStructVolume(imgROIs["External-body"]), volumeDifference, delta=0.005 * volumeDifference)
        arrDifference = ft.arr(imgROIs["External-body"]) - (ft.arr(imgROIs["External"]) - ft.arr(imgROIs["BODY"]))
        self.assertLess(np.abs(arrDifference).mean(), 2e-3)

    def test_mapStructToImg_invalidStruct(self):
        with self.assertRaises(ValueError):
            ft.mapStructToImg(self.img, self.RSfileName, "invalid", displayInfo=True)

    def test_mapStructToImg_emptyStruct(self):
        imgROI = ft.mapStructToImg(self.img, self.RSfileName, "NoContour", displayInfo=True)
        self.assertEqual(ft.getStatistics(imgROI).GetSum(), 0)

    def test_mapStructToImg_binaryMask(self):
        imgROI = ft.mapStructToImg(self.img, self.RSfileName, "testStuct_SphHoleDet", binaryMask=True, displayInfo=True)
        self.assertAlmostEqual(ft.getStatistics(imgROI).GetSum()*np.prod(imgROI.GetSpacing()), 125928, places=-2)

    def test_mapStructToImg_binaryMask_invalidFraction(self):
        with self.assertRaises(ValueError):
            ft.mapStructToImg(self.img, self.RSfileName, "testStuct_SphHoleDet", binaryMask=True, areaFraction=1.1, displayInfo=True)

    def test_mapStructToImg_singleSlice(self):
        """A structure contoured at a single depth is mapped as one image slice thick instead of raising."""
        import os
        import shutil
        import tempfile
        import pydicom as dicom
        import shapely as sph

        # build an RS in which 'PTV_sphere' keeps only its middle contour
        tempDir = tempfile.mkdtemp(prefix="test_mapStructToImg_singleSlice_", dir="unittests/imgManipulate")
        try:
            dicomTags = dicom.dcmread(self.RSfileName)
            ROINumber = next(int(ROI.ROINumber) for ROI in dicomTags.StructureSetROISequence if ROI.ROIName == "PTV_sphere")
            ROIContour = next(ROIContour for ROIContour in dicomTags.ROIContourSequence if int(ROIContour.ReferencedROINumber) == ROINumber)
            contour = ROIContour.ContourSequence[len(ROIContour.ContourSequence) // 2]
            ROIContour.ContourSequence = [contour]
            RSfileName = os.path.join(tempDir, "RS.singleSlice.dcm")
            dicomTags.save_as(RSfileName)

            contourPoints = np.array(contour.ContourData, dtype=float).reshape(-1, 3)
            contourArea = sph.Polygon(contourPoints[:, :2]).area
            contourDepth = contourPoints[0, 2]

            imgROI = ft.mapStructToImg(self.img, RSfileName, "PTV_sphere", displayInfo=True)
            self.assertEqual(imgROI.GetSize(), self.img.GetSize())
            # the mask is one image slice thick, spread over at most two slices by the resampling
            nonZeroSlices = np.unique(np.nonzero(ft.arr(imgROI))[0])
            self.assertLessEqual(len(nonZeroSlices), 2)
            sliceDepths = np.asarray(ft.getVoxelCentres(self.img)[2])[nonZeroSlices]
            self.assertTrue(np.all(np.abs(sliceDepths - contourDepth) <= self.img.GetSpacing()[2]))
            # the volume is the contour area times one image slice thickness
            self.assertAlmostEqual(ft.getStructVolume(imgROI), contourArea * self.img.GetSpacing()[2] / 1e3, delta=0.05 * contourArea * self.img.GetSpacing()[2] / 1e3)

            imgROIbinary = ft.mapStructToImg(self.img, RSfileName, "PTV_sphere", binaryMask=True, areaFraction=0.0)
            self.assertGreater(ft.getStatistics(imgROIbinary).GetSum(), 0)
        finally:
            shutil.rmtree(tempDir, ignore_errors=True)

    def test_mapStructToImg_reversedAxes(self):
        """An image with reversed axes (e.g. a prone CT with direction -1,-1,+1) gives the same mask as the identity-direction image."""
        imgROIref = ft.mapStructToImg(self.img, self.RSfileName, "testStuct_SphHoleDet")
        imgROIrefBinary = ft.mapStructToImg(self.img, self.RSfileName, "testStuct_SphHoleDet", binaryMask=True)
        for flipAxes in ([True, True, False], [False, True, False], [True, True, True]):
            with self.subTest(flipAxes=flipAxes):
                # the same image expressed with reversed axes: the voxel arrays are flipped, the physical content is unchanged
                imgFlipped = sitk.Flip(self.img, flipAxes, flipAboutOrigin=False)
                self.assertEqual(tuple(np.diag(np.reshape(imgFlipped.GetDirection(), (3, 3)))), tuple(-1.0 if flip else 1.0 for flip in flipAxes))

                imgROI = ft.mapStructToImg(imgFlipped, self.RSfileName, "testStuct_SphHoleDet", displayInfo=True)
                self.assertEqual(imgROI.GetSize(), imgFlipped.GetSize())
                self.assertEqual(imgROI.GetOrigin(), imgFlipped.GetOrigin())
                self.assertEqual(imgROI.GetDirection(), imgFlipped.GetDirection())
                # flipped back to the original frame, the mask must be the one of the identity-direction image
                imgROIback = sitk.Flip(imgROI, flipAxes, flipAboutOrigin=False)
                self.assertEqual(imgROIback.GetOrigin(), imgROIref.GetOrigin())
                np.testing.assert_allclose(ft.arr(imgROIback), ft.arr(imgROIref), atol=1e-6)
                self.assertAlmostEqual(ft.getStructVolume(imgROI), ft.getStructVolume(imgROIref), places=6)

                imgROIbinary = ft.mapStructToImg(imgFlipped, self.RSfileName, "testStuct_SphHoleDet", binaryMask=True)
                np.testing.assert_array_equal(ft.arr(sitk.Flip(imgROIbinary, flipAxes, flipAboutOrigin=False)), ft.arr(imgROIrefBinary))

    def test_mapStructToImg_obliqueDirection(self):
        """An image with an oblique (non-diagonal) direction cannot be mapped slice by slice and raises."""
        imgOblique = sitk.Image(self.img)
        angle = np.radians(10)
        imgOblique.SetDirection((np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle), 0, 0, 0, 1))
        with self.assertRaises(ValueError):
            ft.mapStructToImg(imgOblique, self.RSfileName, "testStuct_SphHoleDet")

    def test_mapStructToImg_irregularDepths(self):
        """Contours at depths which are not multiples of the smallest contour distance are mapped as slabs of the common step."""
        import os
        import shutil
        import tempfile
        import pydicom as dicom
        import shapely as sph

        # build an RS in which 'PTV_sphere' keeps three contours 18 and 12 mm apart: 12 is not a multiple of 18, the common step is 6 mm
        tempDir = tempfile.mkdtemp(prefix="test_mapStructToImg_irregularDepths_", dir="unittests/imgManipulate")
        try:
            dicomTags = dicom.dcmread(self.RSfileName)
            ROINumber = next(int(ROI.ROINumber) for ROI in dicomTags.StructureSetROISequence if ROI.ROIName == "PTV_sphere")
            ROIContour = next(ROIContour for ROIContour in dicomTags.ROIContourSequence if int(ROIContour.ReferencedROINumber) == ROINumber)
            contours = sorted(ROIContour.ContourSequence, key=lambda contour: float(contour.ContourData[2]))
            contours = [contours[20], contours[35], contours[45]]
            ROIContour.ContourSequence = contours
            RSfileName = os.path.join(tempDir, "RS.irregularDepths.dcm")
            dicomTags.save_as(RSfileName)

            contourPoints = [np.array(contour.ContourData, dtype=float).reshape(-1, 3) for contour in contours]
            contourDepths = np.array([points[0, 2] for points in contourPoints])
            contourAreas = np.array([sph.Polygon(points[:, :2]).area for points in contourPoints])
            contourDistances = np.diff(contourDepths)
            self.assertFalse(np.isclose(contourDistances[1] / contourDistances[0], np.round(contourDistances[1] / contourDistances[0])))
            step = np.gcd.reduce(np.round(contourDistances * 1000).astype(int)) / 1000

            imgROI = ft.mapStructToImg(self.img, RSfileName, "PTV_sphere", displayInfo=True)
            self.assertEqual(imgROI.GetSize(), self.img.GetSize())
            # each contour is a slab of the common step thickness at its own depth, with an empty gap between the first two
            sliceDepths = np.asarray(ft.getVoxelCentres(self.img)[2])
            sliceSums = ft.arr(imgROI).sum(axis=(1, 2))
            nonZeroDepths = sliceDepths[sliceSums > 0]
            self.assertGreater(len(nonZeroDepths), 0)
            self.assertTrue(np.all(np.min(np.abs(nonZeroDepths[:, np.newaxis] - contourDepths[np.newaxis, :]), axis=1) < step + 1e-6))
            gapSlices = (sliceDepths > contourDepths[0] + step) & (sliceDepths < contourDepths[1] - step)
            self.assertGreater(gapSlices.sum(), 0)
            self.assertTrue(np.all(sliceSums[gapSlices] == 0))
            # the volume is the sum of the contour areas times the common step
            expectedVolume = contourAreas.sum() * step / 1e3
            self.assertAlmostEqual(ft.getStructVolume(imgROI), expectedVolume, delta=0.05 * expectedVolume)

            imgROIbinary = ft.mapStructToImg(self.img, RSfileName, "PTV_sphere", binaryMask=True, areaFraction=0.0)
            self.assertGreater(ft.getStatistics(imgROIbinary).GetSum(), 0)

            # depths with no common step of at least 0.1 mm raise
            contours[2].ContourData = [round(float(value) + (0.001 if valueIdx % 3 == 2 else 0), 3) for valueIdx, value in enumerate(contours[2].ContourData)]
            dicomTags.save_as(RSfileName)
            with self.assertRaises(RuntimeError):
                ft.mapStructToImg(self.img, RSfileName, "PTV_sphere")
        finally:
            shutil.rmtree(tempDir, ignore_errors=True)

    def test_mapStructToImg_contourOutsideImage(self):
        """A contour lying completely outside the image extent is not mapped and does not change the mask."""
        import copy
        import os
        import shutil
        import tempfile
        import pydicom as dicom

        # build an RS in which 'PTV_sphere' has an additional small contour outside the image, at the depth of an existing contour
        tempDir = tempfile.mkdtemp(prefix="test_mapStructToImg_contourOutsideImage_", dir="unittests/imgManipulate")
        try:
            dicomTags = dicom.dcmread(self.RSfileName)
            ROINumber = next(int(ROI.ROINumber) for ROI in dicomTags.StructureSetROISequence if ROI.ROIName == "PTV_sphere")
            ROIContour = next(ROIContour for ROIContour in dicomTags.ROIContourSequence if int(ROIContour.ReferencedROINumber) == ROINumber)
            contour = ROIContour.ContourSequence[len(ROIContour.ContourSequence) // 2]
            contourOutside = copy.deepcopy(contour)
            x0, y0, depth = ft.getExtent(self.img)[0][1] + 5, ft.getExtent(self.img)[1][0] + 5, float(contour.ContourData[2])
            contourOutside.ContourData = [x0, y0, depth, x0 + 0.7, y0, depth, x0 + 0.7, y0 + 0.05, depth, x0, y0 + 0.05, depth]
            contourOutside.NumberOfContourPoints = 4
            ROIContour.ContourSequence.append(contourOutside)
            RSfileName = os.path.join(tempDir, "RS.contourOutsideImage.dcm")
            dicomTags.save_as(RSfileName)

            imgROIref = ft.mapStructToImg(self.img, self.RSfileName, "PTV_sphere")
            imgROI = ft.mapStructToImg(self.img, RSfileName, "PTV_sphere", displayInfo=True)
            np.testing.assert_allclose(ft.arr(imgROI), ft.arr(imgROIref), atol=1e-6)
        finally:
            shutil.rmtree(tempDir, ignore_errors=True)


class test_getStructureArray(unittest.TestCase):
    """_getStructureArray (used by mapStructToImg) maps a polygon given in pixel coordinates (continuous indices) to an array
    of the pixel occupancy: 1 for the pixels lying completely inside the polygon, the area inside the polygon for the pixels
    crossed by the contour and 0 outside, so that the sum of the array is the polygon area."""

    def assertOccupancy(self, polygon, arr):
        self.assertTrue(np.all((arr >= 0) & (arr <= 1 + 1e-12)))
        self.assertAlmostEqual(arr.sum(), polygon.area, places=9)

    def test_getStructureArray_square(self):
        """A square of 10x10 pixels from pixel edge to pixel edge fills exactly the 100 pixels inside."""
        polygon = sph.box(1.5, 2.5, 11.5, 12.5)
        arr = ft.ImgManipulate.imgManipulate._getStructureArray(polygon)
        self.assertOccupancy(polygon, arr)
        np.testing.assert_allclose(arr[3:13, 2:12], 1)
        self.assertEqual(np.count_nonzero(arr), 100)

    def test_getStructureArray_edgesThroughPixelCentres(self):
        """The pixels crossed by the contour in their centres are occupied by half (by a quarter in the corners)."""
        polygon = sph.box(2, 3, 12, 13)
        arr = ft.ImgManipulate.imgManipulate._getStructureArray(polygon)
        self.assertOccupancy(polygon, arr)
        np.testing.assert_allclose(arr[4:13, 3:12], 1)
        np.testing.assert_allclose(arr[3, 3:12], 0.5)
        np.testing.assert_allclose(arr[3, 2], 0.25)

    def test_getStructureArray_smallerThanPixel(self):
        """A polygon lying inside a single pixel occupies that pixel by its area."""
        polygon = sph.box(4.6, 4.6, 5.2, 5.1)
        arr = ft.ImgManipulate.imgManipulate._getStructureArray(polygon)
        self.assertOccupancy(polygon, arr)
        self.assertAlmostEqual(arr[5, 5], 0.3, places=12)
        self.assertEqual(np.count_nonzero(arr), 1)

    def test_getStructureArray_hole(self):
        """The pixels of a hole in the polygon stay empty."""
        polygon = sph.Polygon(sph.box(-0.5, -0.5, 19.5, 19.5).exterior.coords, [sph.box(5.5, 5.5, 13.5, 13.5).exterior.coords])
        arr = ft.ImgManipulate.imgManipulate._getStructureArray(polygon)
        self.assertOccupancy(polygon, arr)
        np.testing.assert_array_equal(arr[6:14, 6:14], 0)
        np.testing.assert_allclose(arr[0:5, 0:20], 1)

    def test_getStructureArray_representativePointOnContour(self):
        """A square with a thin stem above it, whose representative point lies in the stem, i.e. in a pixel crossed by the contour.

        The old flood fill started from that pixel and left the inside of the square empty."""
        polygon = sph.Polygon([(0.2, 0.2), (20.2, 0.2), (20.2, 20.2), (10.5, 20.2), (10.5, 40.5), (9.9, 40.5), (9.9, 20.2), (0.2, 20.2)])
        arr = ft.ImgManipulate.imgManipulate._getStructureArray(polygon)
        self.assertOccupancy(polygon, arr)
        np.testing.assert_allclose(arr[2:19, 2:19], 1)

    def test_getStructureArray_insideSplitByContour(self):
        """Two squares joined by a thin neck, whose pixels crossed by the contour split the inside into two regions.

        The old flood fill filled only the region with its starting point."""
        polygon = sph.Polygon([(0.2, 0.2), (10.2, 0.2), (10.2, 5.0), (20.2, 5.0), (20.2, 0.2), (30.2, 0.2), (30.2, 10.2),
                               (20.2, 10.2), (20.2, 5.4), (10.2, 5.4), (10.2, 10.2), (0.2, 10.2)])
        arr = ft.ImgManipulate.imgManipulate._getStructureArray(polygon)
        self.assertOccupancy(polygon, arr)
        np.testing.assert_allclose(arr[2:9, 2:9], 1)
        np.testing.assert_allclose(arr[2:9, 22:29], 1)

    def test_getStructureArray_randomPolygons(self):
        """Random star-shaped polygons (some thin, some with a hole) in the coordinates used by mapStructToImg (x, y >= -0.5)."""
        rng = np.random.default_rng(12345)
        for polygonIdx in range(300):
            radius = float(rng.choice([0.4, 1.5, 3, 8, 20])) * rng.uniform(0.7, 1.3)
            verticesNo = int(rng.integers(3, 40))
            angles = np.sort(rng.uniform(0, 2 * np.pi, verticesNo))
            radii = radius * (1 + float(rng.choice([0, 0.3, 0.8])) * rng.uniform(-1, 1, verticesNo))
            shell = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            if rng.random() < 0.2:  # thin polygon
                shell[:, int(rng.integers(0, 2))] *= rng.uniform(0.02, 0.2)
            shell = shell - shell.min(axis=0) - 0.5 + (0 if rng.random() < 0.3 else rng.uniform(0, 30, 2))  # some touch x=-0.5 or y=-0.5
            holes = [shell.mean(axis=0) + 0.35 * (shell - shell.mean(axis=0))] if rng.random() < 0.3 and radius > 3 else []
            polygon = sph.Polygon(shell, holes)
            if not polygon.is_valid or polygon.area == 0:
                continue
            with self.subTest(polygonIdx=polygonIdx):
                self.assertOccupancy(polygon, ft.ImgManipulate.imgManipulate._getStructureArray(polygon))


class test_floatingToBinaryMask(unittest.TestCase):

    def setUp(self):
        self.img = ft.readMHD("unittests/testData/MHDImages/img3D.mhd")
        self.RSfileName = ft.sortDicoms("unittests/testData/TPSDicoms/TPSPlan/").RSfileNames
        self.imgROI = ft.mapStructToImg(self.img, self.RSfileName, "testStuct_SphHoleDet", displayInfo=True)

    def test_floatingToBinaryMask(self):
        imgROIBinary = ft.floatingToBinaryMask(self.imgROI, threshold=0.0, thresholdEqual=False, displayInfo=True)
        self.assertAlmostEqual(ft.getStatistics(imgROIBinary).GetSum()*np.prod(imgROIBinary.GetSpacing()), 175149, places=-2)

    def test_floatingToBinaryMask_thresholdEqual(self):
        imgROIBinary = ft.floatingToBinaryMask(self.imgROI, threshold=0.5, thresholdEqual=True, displayInfo=True)
        self.assertAlmostEqual(ft.getStatistics(imgROIBinary).GetSum()*np.prod(imgROIBinary.GetSpacing()), 125928, places=-2)

    def test_floatingToBinaryMask_invalidThreshold(self):
        with self.subTest(thresholdEqual=False):
            with self.assertRaises(ValueError):
                ft.floatingToBinaryMask(self.imgROI, threshold=-0.1, thresholdEqual=False, displayInfo=True)
            with self.assertRaises(ValueError):
                ft.floatingToBinaryMask(self.imgROI, threshold=1.1, thresholdEqual=False, displayInfo=True)
        with self.subTest(thresholdEqual=True):
            with self.assertRaises(ValueError):
                ft.floatingToBinaryMask(self.imgROI, threshold=0, thresholdEqual=True, displayInfo=True)
            with self.assertRaises(ValueError):
                ft.floatingToBinaryMask(self.imgROI, threshold=1.1, thresholdEqual=True, displayInfo=True)


class test_cropImgToMask(unittest.TestCase):

    def setUp(self):
        self.img = ft.createImg([100, 100, 100], centred=True, fillRandom=True)
        self.radii = [30, 30, 30]
        self.imgMask = ft.createEllipseMask(self.img, [0, 0, 0], self.radii)
        self.imgMask = sitk.Cast(self.imgMask, sitk.sitkFloat32)

    def test_cropImgToMask(self):
        imgCrop = ft.cropImgToMask(self.img, self.imgMask, displayInfo=True)
        self.assertListEqual(list(ft.getSize(imgCrop)), list(np.array(self.radii)*2))


class test_setValueMask(unittest.TestCase):

    def setUp(self):
        self.img = ft.createImg([100, 100, 100], centred=True, fillRandom=False)
        self.img += 100
        self.radii = [30, 30, 30]
        self.imgMask = ft.createEllipseMask(self.img, [0, 0, 0], self.radii)
        self.imgMask = sitk.Cast(self.imgMask, sitk.sitkFloat32)

    def test_setValueMask(self):
        self.assertListEqual(list(np.unique(sitk.GetArrayViewFromImage(self.img))), [100])
        imgSetVal = ft.setValueMask(self.img, self.imgMask, value=-10, displayInfo=True)
        self.assertListEqual(list(np.unique(sitk.GetArrayViewFromImage(imgSetVal))), [-10, 100])


class test_resampleImg(unittest.TestCase):

    def setUp(self):
        self.img = ft.readMHD("unittests/testData/MHDImages/img3D.mhd")

    def test_resampleImg(self):
        with self.subTest(interpolation="nearest"):
            imgRef = ft.readMHD("unittests/testData/MHDImages/img3D_resampleNearest.mhd")
            imgRes = ft.resampleImg(self.img, spacing=[2, 1, 3], interpolation="nearest")
            self.assertTrue(ft.compareImg(imgRes, imgRef))
        with self.subTest(interpolation="linear"):
            imgRef = ft.readMHD("unittests/testData/MHDImages/img3D_resampleLinear.mhd")
            imgRes = ft.resampleImg(self.img, spacing=[2, 1, 3], interpolation="linear")
            self.assertTrue(ft.compareImg(imgRes, imgRef))
        for splineOrder in range(1, 6):
            with self.subTest(interpolation="spline", splineOrder=splineOrder):
                imgRef = ft.readMHD(f"unittests/testData/MHDImages/img3D_resampleSpline{splineOrder}.mhd")
                imgRes = ft.resampleImg(self.img, spacing=[2, 1, 3], interpolation="spline", splineOrder=splineOrder)
                self.assertTrue(ft.compareImg(imgRes, imgRef))

    def test_resampleImg_invalidSplineOrder(self):
        """The spline order must be in range 1-5."""
        for splineOrder in (0, 6):
            with self.subTest(splineOrder=splineOrder):
                with self.assertRaises(ValueError):
                    ft.resampleImg(self.img, spacing=[2, 1, 3], interpolation="spline", splineOrder=splineOrder)

    def test_resampleImg_nearestNoHalfway(self):
        """No output voxel lies halfway between two input voxels, so the nearest voxel is unique; the expected image is
        calculated in numpy from the voxel positions."""
        imgRes = ft.resampleImg(self.img, spacing=[2.5, 1, 3], interpolation="nearest")
        indices = []
        for axis in range(3):
            centres = imgRes.GetOrigin()[axis] + np.arange(imgRes.GetSize()[axis]) * imgRes.GetSpacing()[axis]
            continuousIndex = (centres - self.img.GetOrigin()[axis]) / self.img.GetSpacing()[axis]
            self.assertTrue(np.all(np.abs(continuousIndex - np.floor(continuousIndex) - 0.5) > 1e-6))
            indices.append(np.floor(continuousIndex + 0.5).astype(int))
        arrExpected = sitk.GetArrayFromImage(self.img)[np.ix_(indices[2], indices[1], indices[0])]
        np.testing.assert_array_equal(sitk.GetArrayFromImage(imgRes), arrExpected)

    def test_resampleImg_nearestHalfway(self):
        """Resampling from 3 mm to 2 mm puts every third output voxel exactly halfway between two input voxels; it takes
        the voxel with the higher index, also when the x axis of the image is flipped."""
        for direction in ((1, 0, 0, 0, 1, 0, 0, 0, 1), (-1, 0, 0, 0, 1, 0, 0, 0, 1)):
            with self.subTest(directionX=direction[0]):
                img = sitk.GetImageFromArray(np.arange(10, 10 + 4 * 3 * 2, dtype="float32").reshape(2, 3, 4))
                img.SetSpacing((3.0, 3.0, 3.0))
                img.SetOrigin((1.0, -2.0, 3.0))
                img.SetDirection(direction)
                imgRes = ft.resampleImg(img, spacing=[2, 3, 3], interpolation="nearest")
                self.assertGreater(_countHalfway(img, imgRes), 0)
                arrExpected = np.zeros(imgRes.GetSize()[::-1])
                for index in np.ndindex(*imgRes.GetSize()):
                    arrExpected[index[::-1]] = _higherIndexValue(img, imgRes.TransformIndexToPhysicalPoint([int(i) for i in index]))
                np.testing.assert_array_equal(sitk.GetArrayFromImage(imgRes), arrExpected)


class test_sumImg(unittest.TestCase):

    def setUp(self):
        self.fileNames = ["unittests/testData/MHDImages/img3D_resampleLinear.mhd",
                          "unittests/testData/MHDImages/img3D_resampleNearest.mhd",
                          "unittests/testData/MHDImages/img3D_resampleSpline3.mhd"]
        self.imgs = ft.readMHD(self.fileNames)

    def test_sumImg(self):
        imgSum = ft.sumImg(self.imgs, displayInfo=True)
        self.assertAlmostEqual(ft.getStatistics(imgSum).GetSum(), np.sum([ft.getStatistics(img).GetSum() for img in self.imgs]), places=0)

    def test_sumImg_emptyList(self):
        with self.assertRaises(ValueError):
            ft.sumImg([])

    def test_sumImg_wrongFoR(self):
        self.imgs = list(self.imgs)
        self.imgs.append(ft.readMHD("unittests/testData/MHDImages/img3D.mhd"))
        with self.assertRaises(ValueError):
            ft.sumImg(self.imgs, displayInfo=True)

    def test_sumImg_pandasSeries(self):
        dataFrame = pd.DataFrame()
        imgs = ft.readMHD(self.fileNames)
        dataFrame['img'] = imgs
        dataFrame.set_index(pd.Index([4, 6, 3]), inplace=True)
        imgSum = ft.sumImg(dataFrame['img'], displayInfo=True)
        self.assertIsInstance(imgSum, sitk.Image)


class test_divideImg(unittest.TestCase):

    def setUp(self):
        self.img = ft.readMHD("unittests/testData/MHDImages/img3D.mhd")

    def test_divideImg(self):
        imgDiv = ft.divideImg(self.img, self.img, displayInfo=True)
        self.assertListEqual(list(np.unique(sitk.GetArrayViewFromImage(imgDiv))), [0, 1])


class test_sumVectorImg(unittest.TestCase):

    def setUp(self):
        self.img = ft.readMHD("unittests/testData/MHDImages/img3DVec.mhd")
        self.point = ft.getImageCenter(self.img)

    def test_sumVectorImg(self):
        imgSumVec = ft.sumVectorImg(self.img, displayInfo=True)
        pointSum = ft.arr(ft.readMHD("unittests/testData/MHDImages/img3DVecPoint_resampleNearest.mhd")).sum()
        imgSumPoint = ft.getPoint(imgSumVec, point=self.point, interpolation="nearest")
        self.assertEqual(ft.arr(imgSumPoint), pointSum)


class test_expandDimsImg(unittest.TestCase):

    def setUp(self):
        self.img = ft.createImg(size=[20, 30], spacing=[0.5, 0.5], centred=True, fillRandom=True)

    def test_expandDimsImg_default(self):
        imgExpanded = ft.expandDimsImg(self.img, displayInfo=True)
        self.assertEqual(imgExpanded.GetDimension(), 3)
        self.assertEqual(imgExpanded.GetSize(), (*self.img.GetSize(), 1))
        self.assertEqual(imgExpanded.GetSpacing(), (*self.img.GetSpacing(), 1.0))
        self.assertEqual(imgExpanded.GetOrigin(), (*self.img.GetOrigin(), 0.0))
        np.testing.assert_array_equal(sitk.GetArrayViewFromImage(imgExpanded)[0], sitk.GetArrayViewFromImage(self.img))

    def test_expandDimsImg_originSpacing(self):
        imgExpanded = ft.expandDimsImg(self.img, origin=-5.0, spacing=2.0)
        self.assertEqual(imgExpanded.GetOrigin(), (*self.img.GetOrigin(), -5.0))
        self.assertEqual(imgExpanded.GetSpacing(), (*self.img.GetSpacing(), 2.0))

    def test_expandDimsImg_3D(self):
        img3D = ft.createImg(size=[10, 10, 10], spacing=[1, 1, 1], centred=True)
        imgExpanded = ft.expandDimsImg(img3D)
        self.assertEqual(imgExpanded.GetDimension(), 4)
        self.assertEqual(imgExpanded.GetSize(), (10, 10, 10, 1))

    def test_expandDimsImg_invalid_image(self):
        with self.assertRaises(TypeError):
            ft.expandDimsImg(np.zeros((10, 10)))  # type: ignore


class test_setNaNImg(unittest.TestCase):

    def setUp(self):
        self.img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        self.img += 1  # Set all values to 1
        self.img[5, 5, 5] = float('nan')  # Introduce a NaN value

    def test_setNaNImg_defaultValue(self):
        imgResult = ft.setNaNImg(self.img, displayInfo=True)
        self.assertEqual(sitk.GetArrayViewFromImage(imgResult)[5, 5, 5], 0)
        self.assertNotIn(float('nan'), sitk.GetArrayViewFromImage(imgResult))

    def test_setNaNImg_customValue(self):
        imgResult = ft.setNaNImg(self.img, value=-1, displayInfo=True)
        self.assertEqual(sitk.GetArrayViewFromImage(imgResult)[5, 5, 5], -1)
        self.assertNotIn(float('nan'), sitk.GetArrayViewFromImage(imgResult))

    def test_setNaNImg_noNaN(self):
        img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        img += 1  # Set all values to 1
        imgResult = ft.setNaNImg(img, value=-1, displayInfo=True)
        self.assertEqual(sitk.GetArrayViewFromImage(imgResult)[5, 5, 5], 1)
        self.assertNotIn(float('nan'), sitk.GetArrayViewFromImage(imgResult))


class test_getImgBEV(unittest.TestCase):

    @staticmethod
    def _transformBEV(isocentrePosition, gantryAngle, couchAngle):
        """The transform used by getImgBEV (from the BEV to the image space), rebuilt for the expected values."""
        translation = sitk.TranslationTransform(3)
        translation.SetOffset(isocentrePosition)
        gantry = sitk.Euler3DTransform()
        gantry.SetRotation(angleX=0, angleY=0, angleZ=np.deg2rad(gantryAngle))
        couch = sitk.Euler3DTransform()
        couch.SetRotation(angleX=0, angleY=np.deg2rad(couchAngle), angleZ=0)
        rotateBEV = sitk.Euler3DTransform()
        rotateBEV.SetRotation(angleX=np.deg2rad(-90), angleY=0, angleZ=0)
        transformBEV = sitk.CompositeTransform(3)
        transformBEV.AddTransform(rotateBEV)
        transformBEV.AddTransform(sitk.ScaleTransform(3, (1, -1, 1)))
        transform = sitk.CompositeTransform(3)
        for transformSingle in (translation, couch, gantry, transformBEV):
            transform.AddTransform(transformSingle)
        return transform

    def test_getImgBEV_nearestHalfway(self):
        """With different spacings along the axes, the BEV voxels can lie exactly halfway between two image voxels;
        they take the voxel with the higher index."""
        img = sitk.GetImageFromArray(np.arange(10, 10 + 4 * 5 * 6, dtype="float32").reshape(6, 5, 4))
        img.SetSpacing((1.0, 3.0, 2.0))
        isocentrePosition, gantryAngle, couchAngle = [2.0, 6.0, 5.0], 0, 0
        imgBEV = ft.getImgBEV(img, isocentrePosition=isocentrePosition, gantryAngle=gantryAngle, couchAngle=couchAngle, defaultPixelValue=-1, interpolation="nearest")
        transform = self._transformBEV(isocentrePosition, gantryAngle, couchAngle)
        self.assertGreater(_countHalfway(img, imgBEV, transform), 0)
        arrExpected = np.zeros(imgBEV.GetSize()[::-1])
        for index in np.ndindex(*imgBEV.GetSize()):
            point = transform.TransformPoint(imgBEV.TransformIndexToPhysicalPoint([int(i) for i in index]))
            arrExpected[index[::-1]] = _higherIndexValue(img, point, outsideValue=-1)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(imgBEV), arrExpected)


if __name__ == '__main__':
    unittest.main()
