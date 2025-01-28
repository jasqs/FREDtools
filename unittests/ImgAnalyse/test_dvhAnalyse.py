import unittest
import numpy as np
from scipy.stats import norm
from dicompylercore import dvh
import fredtools as ft
import pydvh


class test_DVH(unittest.TestCase):
    def setUp(self):
        self.doseBins = np.arange(0, 140.01, 0.01)
        self.doseBinsCenter = 0.5*(self.doseBins[1:] + self.doseBins[:-1])
        self.dosePrescribed = 70
        self.doseSigma = 7
        self.countsDiff = norm.pdf(self.doseBinsCenter, loc=self.dosePrescribed, scale=self.doseSigma)
        self.countsDiff[self.doseBinsCenter < (self.dosePrescribed-(self.doseBinsCenter.max()-self.dosePrescribed))] = 0
        self.countsCum = self.countsDiff[::-1].cumsum()[::-1]

    def test_DVHInit(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.countsDiff, self.doseBins, type="differential", dose_units="Gy", volume_units="cm3", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertIsInstance(dvhTest, ft.ImgAnalyse.dvhAnalyse.DVH)
            self.assertEqual(dvhTest.type, "differential")
            self.assertTrue(np.allclose(dvhTest.cumulative.counts, self.countsCum))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.countsCum, self.doseBins, type="cumulative", dose_units="Gy", volume_units="cm3", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertIsInstance(dvhTest, ft.ImgAnalyse.dvhAnalyse.DVH)
            self.assertEqual(dvhTest.type, "cumulative")
            self.assertTrue(np.allclose(dvhTest.differential.counts, self.countsDiff))

        # self.assertEqual(dvhTest.volume, self.countsDiff.sum())
        # self.assertEqual(dvhTest.mean, np.sum(self.countsDiff*self.doseBinsCenter)/dvhTest.volume)
        # self.assertEqual(dvhTest.max, self.doseBinsCenter[np.argmax(self.countsDiff)])
        # self.assertEqual(dvhTest.min, self.doseBinsCenter[np.argmin(self.countsDiff)])
        # self.assertEqual(dvhTest.statistic('D98').value, 98)
        # self.assertEqual(dvhTest.statistic('D98').dose, self.doseBinsCenter[np.where(np.cumsum(self.countsDiff) >= 0.98*dvhTest.volume)[0][0]])
        # self.assertEqual(dvhTest.statistic('D2').value, 2)
        # self.assertEqual(dvhTest.statistic('D2').dose, self.doseBinsCenter[np.where(np.cumsum(self.countsDiff) >= 0.02*dvhTest.volume)[0][0]])
        # self.assertEqual(dvhTest.statistic('D50').value, 50)
        # self.assertEqual(dvhTest.statistic('D50').dose, self.doseBinsCenter[np.where(np.cumsum(self.countsDiff) >= 0.50*dvhTest.volume)[0][0]])


class test_getDVHStruct(unittest.TestCase):

    def setUp(self):
        self.dicomFiles = ft.sortDicoms("unittests/testData/TPSDicoms/TPSPlan/", recursive=True)
        planInfo = ft.getRNInfo(self.dicomFiles.RNfileNames)
        self.dosePrescribed = planInfo.dosePrescribed
        self.imgDose = ft.sumImg(ft.getRD(fileNames=self.dicomFiles.RDfileNames))

        self.dvhfileTPS = pydvh.DVHFile.from_file_eclipse("unittests/testData/TPSDicoms/TPSPlan/DVHexportFromTPS.txt")
        self.structName = "testStuct_SphHoleDet"

    def test_getDVHStruct(self):
        dvhTPS = self.dvhfileTPS.get_dvh_by_name(self.structName)
        dvhTPS = dvh.DVH(np.round(dvhTPS.volume_array, 4), np.append(dvhTPS.dose_array, dvhTPS.max_dose), rx_dose=self.dosePrescribed, name=self.structName)  # type: ignore

        dvhStruct = ft.getDVHStruct(self.imgDose, self.dicomFiles.RSfileNames, self.structName, self.dosePrescribed, doseLevelStep=0.001, resampleImg=0.5, displayInfo=True)

        self.assertLessEqual(float(np.abs((dvhStruct.volume/dvhTPS.volume-1)*100)), 1)
        self.assertLessEqual(float(np.abs((dvhStruct.mean/dvhTPS.mean-1)*100)), 0.5)
        self.assertLessEqual(float(np.abs((dvhStruct.statistic('D98').value/dvhTPS.statistic('D98').value-1)*100)), 1)  # type: ignore

    def test_getDVHStruct_invalidInput(self):
        with self.subTest("Invalid resampleImg"):
            with self.assertRaises(ValueError):
                ft.getDVHStruct(self.imgDose, self.dicomFiles.RSfileNames, self.structName, self.dosePrescribed, doseLevelStep=0.001, resampleImg=[1, 1], displayInfo=True)
        with self.subTest("Invalid struct name"):
            with self.assertRaises(ValueError):
                ft.getDVHStruct(self.imgDose, self.dicomFiles.RSfileNames, "invalid", self.dosePrescribed, doseLevelStep=0.001, resampleImg=0.5, displayInfo=True)
        with self.subTest("Invalid RS file"):
            with self.assertRaises(ValueError):
                ft.getDVHStruct(self.imgDose, self.dicomFiles.RNfileNames, self.structName, self.dosePrescribed, doseLevelStep=0.001, resampleImg=0.5, displayInfo=True)


class test_getDVHMask(unittest.TestCase):

    def setUp(self):
        self.dicomFiles = ft.sortDicoms("unittests/testData/TPSDicoms/TPSPlan/", recursive=True)
        planInfo = ft.getRNInfo(self.dicomFiles.RNfileNames)
        self.dosePrescribed = planInfo.dosePrescribed
        self.imgDose = ft.sumImg(ft.getRD(fileNames=self.dicomFiles.RDfileNames))

        self.dvhfileTPS = pydvh.DVHFile.from_file_eclipse("unittests/testData/TPSDicoms/TPSPlan/DVHexportFromTPS.txt")
        self.structName = "testStuct_SphHoleDet"

    def test_getDVHMask(self):
        dvhTPS = self.dvhfileTPS.get_dvh_by_name(self.structName)
        dvhTPS = dvh.DVH(np.round(dvhTPS.volume_array, 4), np.append(dvhTPS.dose_array, dvhTPS.max_dose), rx_dose=self.dosePrescribed, name=self.structName)  # type: ignore

        imgDose = ft.resampleImg(self.imgDose, [0.5, 0.5, 0.5])
        imgMask = ft.mapStructToImg(imgDose, self.dicomFiles.RSfileNames, self.structName)
        dvhStruct = ft.getDVHMask(imgDose, imgMask, self.dosePrescribed, doseLevelStep=0.001, displayInfo=True)

        self.assertLessEqual(float(np.abs((dvhStruct.volume/dvhTPS.volume-1)*100)), 1)
        self.assertLessEqual(float(np.abs((dvhStruct.mean/dvhTPS.mean-1)*100)), 0.5)
        self.assertLessEqual(float(np.abs((dvhStruct.statistic('D98').value/dvhTPS.statistic('D98').value-1)*100)), 1)  # type: ignore

    def test_getDVHMask_invalidInput(self):
        with self.subTest("Not a mask"):
            with self.assertRaises(TypeError):
                ft.getDVHMask(self.imgDose, self.imgDose, self.dosePrescribed, doseLevelStep=0.001, displayInfo=True)
        with self.subTest("FoR mismatch"):
            imgMask = ft.mapStructToImg(self.imgDose, self.dicomFiles.RSfileNames, self.structName)
            imgMask = ft.resampleImg(imgMask, [3, 3, 3])
            with self.assertRaises(TypeError):
                ft.getDVHMask(self.imgDose, imgMask, self.dosePrescribed, doseLevelStep=0.001, displayInfo=True)


if __name__ == '__main__':
    unittest.main()
