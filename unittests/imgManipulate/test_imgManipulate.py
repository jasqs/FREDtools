import unittest
import SimpleITK as sitk
import fredtools as ft
import numpy as np


class test_mapStructToImg(unittest.TestCase):

    def setUp(self):
        self.img = ft.readMHD("unittests/testData/MHDImages/img3D.mhd")
        self.RSfileName = "unittests/testData/TPSDicoms/TPSPlan/RS.KHTOG7e2vQYwpyyzHqOJ3R7o5.CT Horizontal OW.dcm"

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


class test_floatingToBinaryMask(unittest.TestCase):

    def setUp(self):
        self.img = ft.readMHD("unittests/testData/MHDImages/img3D.mhd")
        self.RSfileName = "unittests/testData/TPSDicoms/TPSPlan/RS.KHTOG7e2vQYwpyyzHqOJ3R7o5.CT Horizontal OW.dcm"
        self.imgROI = ft.mapStructToImg(self.img, self.RSfileName, "testStuct_SphHoleDet", displayInfo=True)

    def test_floatingToBinaryMask(self):
        imgROIBinary = ft.floatingToBinaryMask(self.imgROI, threshold=0.0, thresholdEqual=False, displayInfo=True)
        self.assertAlmostEqual(ft.getStatistics(imgROIBinary).GetSum()*np.prod(imgROIBinary.GetSpacing()), 175149, places=-2)

    def test_floatingToBinaryMask_thresholdEqual(self):
        imgROIBinary = ft.floatingToBinaryMask(self.imgROI, threshold=0.5, thresholdEqual=True, displayInfo=True)
        self.assertAlmostEqual(ft.getStatistics(imgROIBinary).GetSum()*np.prod(imgROIBinary.GetSpacing()), 125928, places=-2)

    def test_floatingToBinaryMask_invalidThreshold(self):
        with self.subTest("thresholdEqual=False"):
            with self.assertRaises(ValueError):
                ft.floatingToBinaryMask(self.imgROI, threshold=-0.1, thresholdEqual=False, displayInfo=True)
            with self.assertRaises(ValueError):
                ft.floatingToBinaryMask(self.imgROI, threshold=1.1, thresholdEqual=False, displayInfo=True)
        with self.subTest("thresholdEqual=True"):
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
        result = ft.resampleImg(self.img, spacing=[2.0, 2.0, 2.0])
        ft.resampleImg(self.img, [2, 1, 3], interpolation="spline", splineOrder=5)  # todo
        self.assertIsInstance(result, sitk.Image)


class test_sumImg(unittest.TestCase):

    def setUp(self):
        self.img = sitk.Image([10, 10, 10], sitk.sitkFloat32)

    def test_sumImg(self):
        result = ft.sumImg([self.img, self.img])
        self.assertIsInstance(result, sitk.Image)


class test_imgDivide(unittest.TestCase):

    def setUp(self):
        self.img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        self.imgMask = sitk.Image([10, 10, 10], sitk.sitkUInt8)
        self.imgMask += 1  # Set all values to 1

    def test_imgDivide(self):
        result = ft.divideImg(self.img, self.imgMask)
        self.assertIsInstance(result, sitk.Image)


class test_sumVectorImg(unittest.TestCase):

    def setUp(self):
        self.img = sitk.Image([10, 10, 10], sitk.sitkFloat32)

    def test_sumVectorImg(self):
        vector_img = sitk.Compose([self.img, self.img])
        result = ft.sumVectorImg(vector_img)
        self.assertIsInstance(result, sitk.Image)


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
