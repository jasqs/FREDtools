import unittest
import SimpleITK as sitk
import fredtools as ft
import numpy as np
import pandas as pd


class test_mapStructToImg(unittest.TestCase):

    def setUp(self):
        self.img = ft.readMHD("unittests/testData/MHDImages/img3D.mhd")
        self.RSfileName = ft.sortDicoms("unittests/testData/TPSDicoms/TPSPlan/").RSfileNames

    def test_mapStructToImg(self):
        imgROI = ft.mapStructToImg(self.img, self.RSfileName, "testStuct_SphHoleDet", displayInfo=True)
        self.assertAlmostEqual(ft.getStatistics(imgROI).GetSum()*np.prod(imgROI.GetSpacing()), 130987, places=-1)

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


class test_sumImg(unittest.TestCase):

    def setUp(self):
        self.fileNames = ["unittests/testData/MHDImages/img3D_resampleLinear.mhd",
                          "unittests/testData/MHDImages/img3D_resampleNearest.mhd",
                          "unittests/testData/MHDImages/img3D_resampleSpline0.mhd"]
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


if __name__ == '__main__':
    unittest.main()
