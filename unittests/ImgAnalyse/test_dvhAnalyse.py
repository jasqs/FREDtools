import unittest
import numpy as np
from scipy.stats import norm
import fredtools as ft
import pydvh


class test_DVH(unittest.TestCase):
    def setUp(self):
        # prepare DVH data based on a normal distribution
        self.doseBinSize = 0.01
        self.doseDiffEdges = np.arange(0, 140.0 + self.doseBinSize, self.doseBinSize)
        self.doseDiffCenters = 0.5*(self.doseDiffEdges[1:] + self.doseDiffEdges[:-1])
        self.doseCum = self.doseDiffEdges[:-1]
        self.dosePrescribed = 70
        self.doseSigma = 7
        self.volume = np.random.rand()*1000
        self.volumeDiff = norm.pdf(self.doseDiffCenters, loc=self.dosePrescribed, scale=self.doseSigma)
        self.volumeDiff[self.doseDiffCenters < (self.dosePrescribed-(self.doseDiffCenters.max()-self.dosePrescribed))] = 0  # make the differential counts symetric around the prescribed dose
        self.volumeDiff = np.round(self.volumeDiff, 5)
        self.volumeDiff *= self.volume/self.volumeDiff.sum()
        self.volumeCum = np.cumsum(self.volumeDiff[::-1])[::-1]

        # prepare example DVH data based on TPS export
        self.TPS_dvhs = pydvh.DVHFile.from_file_eclipse("unittests/testData/TPSDicoms/TPSPlan/DVHexportFromTPS.txt")
        self.TPS_structNames = self.TPS_dvhs._structure_names
        # self.TPS_structNames = ["testStuct_SphHoleDet"]
        dicomFiles = ft.sortDicoms("unittests/testData/TPSDicoms/TPSPlan/", recursive=True)
        planInfo = ft.getRNInfo(dicomFiles.RNfileNames)
        self.TPS_dosePrescribed = planInfo.dosePrescribed

    def test_DVHInit(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertIsInstance(dvhTest, ft.ImgAnalyse.dvhAnalyse.DVH)
            self.assertTrue(np.allclose(dvhTest.volumeCumAbs, self.volumeCum))
            self.assertTrue(np.allclose(dvhTest.volumeCumRel, self.volumeCum/self.volumeDiff.sum()*100))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertIsInstance(dvhTest, ft.ImgAnalyse.dvhAnalyse.DVH)
            self.assertTrue(np.allclose(dvhTest.volumeDiffAbs, self.volumeDiff))
            self.assertTrue(np.allclose(dvhTest.volumeDiffRel, self.volumeDiff/self.volumeDiff.sum()*100))

        with self.subTest("Differential DVH with no dosePrescribed"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", name="testDVH", color="r")
            self.assertIsInstance(dvhTest, ft.ImgAnalyse.dvhAnalyse.DVH)
            self.assertAlmostEqual(dvhTest.dosePrescribed, self.dosePrescribed, places=7)

        with self.subTest("Cumulative DVH with no dosePrescribed"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", name="testDVH", color="r")
            self.assertIsInstance(dvhTest, ft.ImgAnalyse.dvhAnalyse.DVH)
            self.assertAlmostEqual(dvhTest.dosePrescribed, self.dosePrescribed, places=7)

        with self.subTest("DVH with no name"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, color="r")
            self.assertIsInstance(dvhTest, ft.ImgAnalyse.dvhAnalyse.DVH)
            self.assertEqual(dvhTest.name, "unknown")

        with self.subTest("DVH with no color"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH")
            self.assertIsInstance(dvhTest, ft.ImgAnalyse.dvhAnalyse.DVH)
            self.assertEqual(dvhTest.color, "b")

        with self.subTest("Invalid type"):
            with self.assertRaises(AttributeError):
                ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="invalid", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")  # type: ignore

        with self.subTest("Invalid dose"):
            with self.assertRaises(AttributeError):
                ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges[:-1], type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            with self.assertRaises(AttributeError):
                doseEdges = self.doseDiffEdges.copy()
                np.random.shuffle(doseEdges)
                ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, doseEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            with self.assertRaises(AttributeError):
                ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseDiffEdges, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            with self.assertRaises(AttributeError):
                ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseCum, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")

        with self.subTest("Invalid volume"):
            with self.assertRaises(AttributeError):
                ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff[:-1], self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            with self.assertRaises(AttributeError):
                invalidVolume = np.array([0, 0, 0])
                ft.ImgAnalyse.dvhAnalyse.DVH(invalidVolume, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            with self.assertRaises(AttributeError):
                negativeVolume = np.array([-1, -2, -3])
                ft.ImgAnalyse.dvhAnalyse.DVH(negativeVolume, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            with self.assertRaises(AttributeError):
                emptyVolume = np.array([])
                ft.ImgAnalyse.dvhAnalyse.DVH(emptyVolume, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")

    def test_repr(self):
        dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
        self.assertIsInstance(repr(dvhTest), str)

    def test_compareEquality(self):
        dvhTest1 = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH1", color="r")
        dvhTest2 = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH2", color="r")
        self.assertFalse(dvhTest1 == "invalid")
        self.assertTrue(dvhTest1 == dvhTest2)

    def test_doseLevels(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.doseLevels, self.doseCum))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.doseLevels, self.doseCum))

    def test_doseDiffCenters(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.doseDiffCenters, self.doseDiffCenters))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.doseDiffCenters, self.doseDiffCenters))

    def test_doseDiffEdges(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.doseDiffEdges, self.doseDiffEdges))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.doseDiffEdges, self.doseDiffEdges))

    def test_volumeDiffAbs(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.volumeDiffAbs, self.volumeDiff))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.volumeDiffAbs, self.volumeDiff))

    def test_volumeDiffRel(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.volumeDiffRel, self.volumeDiff/self.volumeDiff.sum()*100))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.volumeDiffRel, self.volumeDiff/self.volumeDiff.sum()*100))

    def test_volumeCumAbs(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.volumeCumAbs, self.volumeCum))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.volumeCumAbs, self.volumeCum))

    def test_volumeCumRel(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.volumeCumRel, self.volumeCum/self.volumeDiff.sum()*100))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertTrue(np.allclose(dvhTest.volumeCumRel, self.volumeCum/self.volumeDiff.sum()*100))

    def test_volume(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.volume, self.volume)

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.volume, self.volume)

    def test_mean(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.mean, np.sum(self.volumeDiff*self.doseDiffCenters)/dvhTest.volume)

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.mean, np.sum(self.volumeDiff*self.doseDiffCenters)/dvhTest.volume)

    def test_stdDev(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.stdDev, np.sqrt(np.sum(self.volumeDiff*(self.doseDiffCenters-dvhTest.mean)**2)/dvhTest.volume))

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.stdDev, np.sqrt(np.sum(self.volumeDiff*(self.doseDiffCenters-dvhTest.mean)**2)/dvhTest.volume))

    def test_median(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.median, self.dosePrescribed)

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.median, self.dosePrescribed)

    def test_max(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertEqual(dvhTest.max, self.doseDiffEdges[np.nonzero(self.volumeDiff)[0][-1]+1])

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertEqual(dvhTest.max, self.doseDiffEdges[np.nonzero(self.volumeDiff)[0][-1]+1])

    def test_min(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertEqual(dvhTest.min, self.doseDiffEdges[np.nonzero(self.volumeDiff)[0]][0])

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertEqual(dvhTest.min, self.doseDiffEdges[np.nonzero(self.volumeDiff)[0]][0])

    def test_volumeConstraint(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.volumeConstraint(0), self.volume, places=7)
            self.assertAlmostEqual(dvhTest.volumeConstraint(100), self.volume/2, places=7)

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum,  type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.volumeConstraint(0), self.volume, places=7)
            self.assertAlmostEqual(dvhTest.volumeConstraint(100), self.volume/2, places=7)

        with self.subTest("Invalid input"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum,  type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            with self.assertRaises(AttributeError):
                dvhTest.volumeConstraint(-1)

    def test_doseConstraint(self):
        with self.subTest("Differential DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.doseConstraint(0), dvhTest.max+self.doseBinSize/2, places=7)
            self.assertAlmostEqual(dvhTest.doseConstraint(50), self.dosePrescribed, places=7)
            self.assertAlmostEqual(dvhTest.doseConstraint(100), dvhTest.min-self.doseBinSize/2, places=7)

        with self.subTest("Cumulative DVH"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            self.assertAlmostEqual(dvhTest.doseConstraint(0), dvhTest.max+self.doseBinSize/2, places=7)
            self.assertAlmostEqual(dvhTest.doseConstraint(50), self.dosePrescribed, places=7)
            self.assertAlmostEqual(dvhTest.doseConstraint(100), dvhTest.min-self.doseBinSize/2, places=7)

        with self.subTest("Invalid input"):
            dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeCum, self.doseCum, type="cumulative", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
            with self.assertRaises(AttributeError):
                dvhTest.doseConstraint(-1)

    def test_statistic(self):
        dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
        with self.subTest("Dose constraints"):
            self.assertAlmostEqual(dvhTest.statistic(f'D50'), self.dosePrescribed, places=7, msg=f"D50:\n\t Eval: {dvhTest.statistic(f'D50')} \n\t Ref: {self.dosePrescribed}")
            self.assertAlmostEqual(dvhTest.statistic(f'D{self.volume/2}cc'), self.dosePrescribed, places=7, msg=f"D{self.volume/2}cc:\n\t Eval: {dvhTest.statistic(f'D{self.volume/2}cc')} \n\t Ref: {self.dosePrescribed}")

            self.assertAlmostEqual(dvhTest.statistic(f'D0'), dvhTest.max+self.doseBinSize/2, places=7, msg=f"D0:\n\t Eval: {dvhTest.statistic(f'D0')}\n\t Ref: {dvhTest.max+self.doseBinSize/2}")
            self.assertAlmostEqual(dvhTest.statistic(f'D0cc'), dvhTest.max+self.doseBinSize/2, places=7, msg=f"D0cc:\n\t Eval: {dvhTest.statistic(f'D0cc')} \n\t Ref: {dvhTest.max+self.doseBinSize/2}")

            self.assertAlmostEqual(dvhTest.statistic(f'D100'), dvhTest.min-self.doseBinSize/2, places=7, msg=f"D100:\n\t Eval: {dvhTest.statistic(f'D100')} \n\t Ref: {dvhTest.min-self.doseBinSize/2}")
            self.assertAlmostEqual(dvhTest.statistic(f'D{self.volume}cc'), dvhTest.min-self.doseBinSize/2, places=7, msg=f"D{self.volume}cc:\n\t Eval: {dvhTest.statistic(f'D{self.volume}cc')} \n\t Ref: {dvhTest.min-self.doseBinSize/2}")

        with self.subTest("Volume constraints"):
            self.assertAlmostEqual(dvhTest.statistic(f'V100'), self.volume/2, places=7)
            self.assertAlmostEqual(dvhTest.statistic(f'V{self.dosePrescribed}Gy'), self.volume/2, places=7)

            self.assertAlmostEqual(dvhTest.statistic(f'V0'), self.volume, places=7)
            self.assertAlmostEqual(dvhTest.statistic(f'V0cc'), self.volume, places=7)

            self.assertAlmostEqual(dvhTest.statistic(f'V{self.doseDiffEdges[np.nonzero(self.volumeDiff)[0][-1]+2]/self.dosePrescribed*100}'), 0, places=7)
            self.assertAlmostEqual(dvhTest.statistic(f'V{self.doseDiffEdges[np.nonzero(self.volumeDiff)[0][-1]+2]}Gy'), 0, places=7)

        with self.subTest("Invalid input"):
            with self.assertRaises(AttributeError):
                dvhTest.statistic('invalid')
                dvhTest.statistic('x70')

    def test_compare(self):
        dvhTest1 = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH1", color="r")
        dvhTest2 = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH2", color="r")
        self.assertIsNone(dvhTest1.compare(dvhTest2))

    def test_displayInfo(self):
        dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
        self.assertIsNone(dvhTest.displayInfo())

    def test_plot(self):
        dvhTest = ft.ImgAnalyse.dvhAnalyse.DVH(self.volumeDiff, self.doseDiffEdges, type="differential", dosePrescribed=self.dosePrescribed, name="testDVH", color="r")
        self.assertIsNone(dvhTest.plot())

    def test_TPSExportBased(self):
        for structName in self.TPS_structNames:  # type: ignore
            with self.subTest(f"TPS DVH for {structName}"):
                dvhTPSRef = self.TPS_dvhs.get_dvh_by_name(structName)
                counts = dvhTPSRef.volume_array  # type: ignore
                bins = dvhTPSRef.dose_array  # type: ignore
                dvhTPSEval = ft.ImgAnalyse.dvhAnalyse.DVH(counts, bins, type="cumulative", dosePrescribed=self.TPS_dosePrescribed, name=structName)
                # volume agree within 0.1% or 0.05 cm3
                self.assertTrue(np.allclose(dvhTPSEval.volume, dvhTPSRef.total_volume, rtol=0.001, atol=0.05), msg=f"Volume\n\tRef:  {dvhTPSRef.total_volume}\n\tEval: {dvhTPSEval.volume}")  # type: ignore
                # mean dose agree within 0.01% or 0.01 Gy
                self.assertTrue(np.allclose(dvhTPSEval.mean, dvhTPSRef.mean_dose, rtol=0.0001, atol=0.01), msg=f"Mean dose\n\tRef:  {dvhTPSRef.mean_dose}\n\tEval: {dvhTPSEval.mean}")  # type: ignore
                # median dose agree within 0.01% or 0.01 Gy
                self.assertTrue(np.allclose(dvhTPSEval.median, dvhTPSRef.median_dose, rtol=0.0001, atol=0.01), msg=f"Median dose\n\tRef:  {dvhTPSRef.median_dose}\n\tEval: {dvhTPSEval.median}")  # type: ignore
                # minimum dose agree within 0.1% or 0.01 Gy
                self.assertTrue(np.allclose(dvhTPSEval.min, dvhTPSRef.min_dose, rtol=0.001, atol=0.1), msg=f"Min dose\n\tRef:  {dvhTPSRef.min_dose}\n\tEval: {dvhTPSEval.min}")  # type: ignore
                # maximum dose agree within 0.01% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.max, dvhTPSRef.max_dose, rtol=0.001, atol=0.1), msg=f"Max dose\n\tRef:  {dvhTPSRef.max_dose}\n\tEval: {dvhTPSEval.max}")  # type: ignore


class test_getDVHMask(unittest.TestCase):
    def setUp(self):
        self.dicomFiles = ft.sortDicoms("unittests/testData/TPSDicoms/TPSPlan/", recursive=True)
        planInfo = ft.getRNInfo(self.dicomFiles.RNfileNames)
        self.dosePrescribed = planInfo.dosePrescribed
        self.imgDose = ft.sumImg(ft.getRD(fileNames=self.dicomFiles.RDfileNames))

        self.dvhfileTPS = pydvh.DVHFile.from_file_eclipse("unittests/testData/TPSDicoms/TPSPlan/DVHexportFromTPS.txt")
        # self.structNames = ["PTV_sphere",  "testStuct_SphHoleDet", "BODY", "Body-SpherePTV"]
        self.structNames = ["PTV_sphere",  "testStuct_SphHoleDet", "BODY"]
        # self.structNames = ["PTV_sphere"]
        # self.structNames = ["testStruct_small"]

    def test_getDVHMask(self):
        for structName in self.structNames:  # type: ignore
            with self.subTest(f"DVH for {structName}"):

                dvhTPSRef = self.dvhfileTPS.get_dvh_by_name(structName)
                counts = dvhTPSRef.volume_array  # type: ignore
                bins = dvhTPSRef.dose_array  # type: ignore
                dvhTPSRef = ft.ImgAnalyse.dvhAnalyse.DVH(counts, bins, type="cumulative", dosePrescribed=self.dosePrescribed, name=structName)

                # imgDose = ft.resampleImg(self.imgDose, [1.5, 1.5, 1.2])
                imgDose = self.imgDose
                imgMask = ft.mapStructToImg(imgDose, self.dicomFiles.RSfileNames, structName)
                dvhTPSEval = ft.getDVHMask(imgDose, imgMask, self.dosePrescribed, displayInfo=True)

                # volume agree within 2% or 0.5 cm3
                self.assertTrue(np.allclose(dvhTPSEval.volume, dvhTPSRef.volume, rtol=0.02, atol=0.5), msg=f"Volume\n\tRef:  {dvhTPSRef.volume}\n\tEval: {dvhTPSEval.volume}")
                # mean dose agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.mean, dvhTPSRef.mean, rtol=0.001, atol=0.1), msg=f"Mean dose\n\tRef:  {dvhTPSRef.mean}\n\tEval: {dvhTPSEval.mean}")
                # Standard deviation agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.stdDev, dvhTPSRef.stdDev, rtol=0.001, atol=0.1), msg=f"Standard Deviation\n\tRef:  {dvhTPSRef.stdDev}\n\tEval: {dvhTPSEval.stdDev}")
                # Median dose agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.median, dvhTPSRef.median, rtol=0.001, atol=0.1), msg=f"Median dose\n\tRef:  {dvhTPSRef.median}\n\tEval: {dvhTPSEval.median}")
                # D98 agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.statistic('D98'), dvhTPSRef.statistic('D98'), rtol=0.001, atol=0.1), msg=f"D98\n\tRef:  {dvhTPSRef.statistic('D98')}\n\tEval: {dvhTPSEval.statistic('D98')}")
                # D2 agree within 0.1%  or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.statistic('D2'), dvhTPSRef.statistic('D2'), rtol=0.001, atol=0.1), msg=f"D02\n\tRef:  {dvhTPSRef.statistic('D2')}\n\tEval: {dvhTPSEval.statistic('D2')}")
                # # Maximum dose agree within 0.1% or 0.1 Gy
                # self.assertTrue(np.allclose(dvhTPSEval.max, dvhTPSRef.max, rtol=0.001, atol=0.1), msg=f"Max dose\n\tRef:  {dvhTPSRef.max}\n\tEval: {dvhTPSEval.max}")
                # # Minimum dose agree within 0.1% or 0.1 Gy
                # self.assertTrue(np.allclose(dvhTPSEval.min, dvhTPSRef.min, rtol=0.001, atol=0.1), msg=f"Min dose\n\tRef:  {dvhTPSRef.min}\n\tEval: {dvhTPSEval.min}")
                # D50 agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.statistic('D50'), dvhTPSRef.statistic('D50'), rtol=0.001, atol=0.1), msg=f"D50\n\tRef:  {dvhTPSRef.statistic('D50')}\n\tEval: {dvhTPSEval.statistic('D50')}")
                # V100 agree within 1% or 0.1 cm3
                self.assertTrue(np.allclose(dvhTPSEval.statistic('V100'), dvhTPSRef.statistic('V100'), rtol=0.01, atol=0.1), msg=f"V100\n\tRef:  {dvhTPSRef.statistic('V100')}\n\tEval: {dvhTPSEval.statistic('V100')}")

    def test_getDVHMask_invalidInput(self):
        with self.subTest("Not a mask"):
            with self.assertRaises(TypeError):
                ft.getDVHMask(self.imgDose, self.imgDose, self.dosePrescribed, displayInfo=True)
        with self.subTest("FoR mismatch"):
            imgMask = ft.mapStructToImg(self.imgDose, self.dicomFiles.RSfileNames, self.structNames[0])
            imgMask = ft.resampleImg(imgMask, [3, 3, 3])
            with self.assertRaises(TypeError):
                ft.getDVHMask(self.imgDose, imgMask, self.dosePrescribed, displayInfo=True)


class test_getDVHStruct(unittest.TestCase):
    def setUp(self):
        self.dicomFiles = ft.sortDicoms("unittests/testData/TPSDicoms/TPSPlan/", recursive=True)
        planInfo = ft.getRNInfo(self.dicomFiles.RNfileNames)
        self.dosePrescribed = planInfo.dosePrescribed
        self.imgDose = ft.sumImg(ft.getRD(fileNames=self.dicomFiles.RDfileNames))

        self.dvhfileTPS = pydvh.DVHFile.from_file_eclipse("unittests/testData/TPSDicoms/TPSPlan/DVHexportFromTPS.txt")
        # self.structNames = ["PTV_sphere",  "testStuct_SphHoleDet", "BODY", "Body-SpherePTV"]
        self.structNames = ["PTV_sphere",  "testStuct_SphHoleDet", "BODY"]
        # self.structNames = ["PTV_sphere"]
        # self.structNames = ["testStruct_small"]

    def test_getDVHStruct(self):
        for structName in self.structNames:  # type: ignore
            with self.subTest(f"DVH for {structName}"):

                dvhTPSRef = self.dvhfileTPS.get_dvh_by_name(structName)
                counts = dvhTPSRef.volume_array  # type: ignore
                bins = dvhTPSRef.dose_array  # type: ignore
                dvhTPSRef = ft.ImgAnalyse.dvhAnalyse.DVH(counts, bins, type="cumulative", dosePrescribed=self.dosePrescribed, name=structName)

                # imgDose = ft.resampleImg(self.imgDose, [1.5, 1.5, 1.2])
                imgDose = self.imgDose
                dvhTPSEval = ft.getDVHStruct(imgDose, self.dicomFiles.RSfileNames, structName=structName, dosePrescribed=self.dosePrescribed, displayInfo=True)

                # volume agree within 2% or 0.5 cm3
                self.assertTrue(np.allclose(dvhTPSEval.volume, dvhTPSRef.volume, rtol=0.02, atol=0.5), msg=f"Volume\n\tRef:  {dvhTPSRef.volume}\n\tEval: {dvhTPSEval.volume}")
                # mean dose agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.mean, dvhTPSRef.mean, rtol=0.001, atol=0.1), msg=f"Mean dose\n\tRef:  {dvhTPSRef.mean}\n\tEval: {dvhTPSEval.mean}")
                # Standard deviation agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.stdDev, dvhTPSRef.stdDev, rtol=0.001, atol=0.1), msg=f"Standard Deviation\n\tRef:  {dvhTPSRef.stdDev}\n\tEval: {dvhTPSEval.stdDev}")
                # Median dose agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.median, dvhTPSRef.median, rtol=0.001, atol=0.1), msg=f"Median dose\n\tRef:  {dvhTPSRef.median}\n\tEval: {dvhTPSEval.median}")
                # D98 agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.statistic('D98'), dvhTPSRef.statistic('D98'), rtol=0.001, atol=0.1), msg=f"D98\n\tRef:  {dvhTPSRef.statistic('D98')}\n\tEval: {dvhTPSEval.statistic('D98')}")
                # D2 agree within 0.1%  or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.statistic('D2'), dvhTPSRef.statistic('D2'), rtol=0.001, atol=0.1), msg=f"D02\n\tRef:  {dvhTPSRef.statistic('D2')}\n\tEval: {dvhTPSEval.statistic('D2')}")
                # # Maximum dose agree within 0.1% or 0.1 Gy
                # self.assertTrue(np.allclose(dvhTPSEval.max, dvhTPSRef.max, rtol=0.001, atol=0.1), msg=f"Max dose\n\tRef:  {dvhTPSRef.max}\n\tEval: {dvhTPSEval.max}")
                # # Minimum dose agree within 0.1% or 0.1 Gy
                # self.assertTrue(np.allclose(dvhTPSEval.min, dvhTPSRef.min, rtol=0.001, atol=0.1), msg=f"Min dose\n\tRef:  {dvhTPSRef.min}\n\tEval: {dvhTPSEval.min}")
                # D50 agree within 0.1% or 0.1 Gy
                self.assertTrue(np.allclose(dvhTPSEval.statistic('D50'), dvhTPSRef.statistic('D50'), rtol=0.001, atol=0.1), msg=f"D50\n\tRef:  {dvhTPSRef.statistic('D50')}\n\tEval: {dvhTPSEval.statistic('D50')}")
                # V100 agree within 1% or 0.1 cm3
                self.assertTrue(np.allclose(dvhTPSEval.statistic('V100'), dvhTPSRef.statistic('V100'), rtol=0.01, atol=0.1), msg=f"V100\n\tRef:  {dvhTPSRef.statistic('V100')}\n\tEval: {dvhTPSEval.statistic('V100')}")

    def test_getDVHStruct_invalidInput(self):
        with self.subTest("Invalid structName"):
            with self.assertRaises(AttributeError):
                ft.getDVHStruct(self.imgDose, self.dicomFiles.RSfileNames, structName="invalid", dosePrescribed=self.dosePrescribed, displayInfo=True)
        with self.subTest("Invalid Dicom with structures"):
            with self.assertRaises(TypeError):
                ft.getDVHStruct(self.imgDose, self.dicomFiles.RNfileNames, structName=self.structNames[0], dosePrescribed=self.dosePrescribed, displayInfo=True)
        with self.subTest("Invalid resampling"):
            with self.assertRaises(ValueError):
                ft.getDVHStruct(self.imgDose, self.dicomFiles.RSfileNames, structName=self.structNames[0], dosePrescribed=self.dosePrescribed, resampleImg=[2, 2])


if __name__ == '__main__':
    unittest.main()
