import unittest
import numpy as np
import SimpleITK as sitk
import fredtools as ft

GIImagesPath = "unittests/testData/MHDImages/GIImages"


def _createGaussImg(shift=(0, 0, 0), size=41, amplitude=10):
    """Create a 3D Gaussian dose distribution (sigma 8 mm, `amplitude` [Gy] maximum, `size` voxels of 1 mm along each axis, centred at 0) shifted by `shift` [mm]."""
    img = ft.createImg([size, size, size], spacing=[1, 1, 1], origin=[-(size // 2)] * 3)
    X, Y, Z = np.meshgrid(*ft.getVoxelCentres(img), indexing="ij")
    arr = amplitude * np.exp(-((X - shift[0])**2 + (Y - shift[1])**2 + (Z - shift[2])**2) / (2 * 8**2))
    imgGauss = sitk.GetImageFromArray(np.moveaxis(arr, [0, 1, 2], [2, 1, 0]))
    imgGauss.CopyInformation(img)
    return imgGauss


class test_calcGammaIndex(unittest.TestCase):

    def setUp(self):
        self.imgRef = ft.readMHD(f"{GIImagesPath}/GI_doseRef.mhd")
        self.imgEval = ft.readMHD(f"{GIImagesPath}/GI_doseEval.mhd")
        self.imgGauss = _createGaussImg()
        self.imgGaussShifted = _createGaussImg(shift=(1, 0, 0))
        self.imgUniform = ft.createImg([21, 21, 21], spacing=[1, 1, 1], origin=[0, 0, 0]) + 10.0
        self.CPUNo = ft.CPUNO

    def tearDown(self):
        ft.CPUNO = self.CPUNo

    def _assertPassRateMap(self, imgRefGI, imgPR):
        """Check a pass-rate map against the stored gamma map of the same configuration.

        Both modes analyse the same voxels (all voxels above the dose cutoff), so the masks of the analysed
        voxels must be identical and the pass/fail outcome is compared on these voxels.
        """
        arrRefGI = ft.arr(imgRefGI)
        arrPR = ft.arr(imgPR)
        analysed = arrRefGI >= 0
        analysedPR = arrPR >= 0
        self.assertTrue(ft.compareImgFoR(imgRefGI, imgPR))
        self.assertEqual(imgPR.GetPixelIDTypeAsString(), "8-bit signed integer")
        self.assertTrue(set(np.unique(arrPR)).issubset({-1, 0, 1}))
        self.assertTrue(np.array_equal(analysedPR, analysed))
        self.assertGreaterEqual(np.mean((arrPR[analysed] == 1) == (arrRefGI[analysed] <= 1)), 0.999)
        self.assertAlmostEqual(float(imgPR.GetMetaData("GIPR")), ft.getGIstat(imgRefGI).passRate, delta=1)
        self.assertAlmostEqual(ft.getGIstat(imgPR).passRate, ft.getGIstat(imgRefGI).passRate, delta=1)

    def test_calcGammaIndex_DD2_DTA2_global_DCO5(self):
        imgRefGI = ft.readMHD(f"{GIImagesPath}/GI_DD2_DTA2_global_DCO5.mhd")
        imgGI = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="global", displayInfo=True)
        self.assertTrue(ft.compareImgFoR(imgRefGI, imgGI))
        self.assertTrue(ft.compareImg(imgRefGI, imgGI, decimal=3))
        self.assertEqual(imgGI.GetPixelIDTypeAsString(), "32-bit float")
        self.assertAlmostEqual(float(imgGI.GetMetaData("GIPR")), float(imgRefGI.GetMetaData("GIPR")), delta=1)
        self.assertAlmostEqual(ft.getGIstat(imgGI).passRate, ft.getGIstat(imgRefGI).passRate, delta=1)
        imgPR = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="global", mode="pass-rate")
        self._assertPassRateMap(imgRefGI, imgPR)

    def test_calcGammaIndex_DD2_DTA2_local_DCO5(self):
        imgRefGI = ft.readMHD(f"{GIImagesPath}/GI_DD2_DTA2_local_DCO5.mhd")
        imgGI = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="local")
        self.assertTrue(ft.compareImgFoR(imgRefGI, imgGI))
        self.assertTrue(ft.compareImg(imgRefGI, imgGI, decimal=3))
        self.assertEqual(imgGI.GetPixelIDTypeAsString(), "32-bit float")
        self.assertAlmostEqual(float(imgGI.GetMetaData("GIPR")), float(imgRefGI.GetMetaData("GIPR")), delta=1)
        self.assertAlmostEqual(ft.getGIstat(imgGI).passRate, ft.getGIstat(imgRefGI).passRate, delta=1)
        imgPR = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="local", mode="pass-rate")
        self._assertPassRateMap(imgRefGI, imgPR)

    def test_calcGammaIndex_DD1_DTA2_local_DCO5(self):
        imgRefGI = ft.readMHD(f"{GIImagesPath}/GI_DD1_DTA2_local_DCO5.mhd")
        imgGI = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=1, DTA=2, DCO=0.05, DDType="local")
        self.assertTrue(ft.compareImgFoR(imgRefGI, imgGI))
        self.assertTrue(ft.compareImg(imgRefGI, imgGI, decimal=3))
        self.assertEqual(imgGI.GetPixelIDTypeAsString(), "32-bit float")
        self.assertAlmostEqual(float(imgGI.GetMetaData("GIPR")), float(imgRefGI.GetMetaData("GIPR")), delta=1)
        self.assertAlmostEqual(ft.getGIstat(imgGI).passRate, ft.getGIstat(imgRefGI).passRate, delta=1)
        imgPR = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=1, DTA=2, DCO=0.05, DDType="local", mode="pass-rate")
        self._assertPassRateMap(imgRefGI, imgPR)

    def test_calcGammaIndex_DD2_DTA2_global_DCO5_globalNorm50(self):
        imgRefGI = ft.readMHD(f"{GIImagesPath}/GI_DD2_DTA2_global_DCO5_globalNorm50.mhd")
        imgGI = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="global", globalNorm=50)
        self.assertTrue(ft.compareImgFoR(imgRefGI, imgGI))
        self.assertTrue(ft.compareImg(imgRefGI, imgGI, decimal=3))
        self.assertEqual(imgGI.GetPixelIDTypeAsString(), "32-bit float")
        self.assertAlmostEqual(float(imgGI.GetMetaData("GIPR")), float(imgRefGI.GetMetaData("GIPR")), delta=1)
        self.assertAlmostEqual(ft.getGIstat(imgGI).passRate, ft.getGIstat(imgRefGI).passRate, delta=1)
        imgPR = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="global", globalNorm=50, mode="pass-rate")
        self._assertPassRateMap(imgRefGI, imgPR)

    def test_calcGammaIndex_DD2_DTA2_global_DCO5_stepSize05abs(self):
        imgRefGI = ft.readMHD(f"{GIImagesPath}/GI_DD2_DTA2_global_DCO5_stepSize0.5abs.mhd")
        imgGI = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="global", stepSize=0.5, fractionalStepSize=False)
        self.assertTrue(ft.compareImgFoR(imgRefGI, imgGI))
        self.assertTrue(ft.compareImg(imgRefGI, imgGI, decimal=3))
        self.assertEqual(imgGI.GetPixelIDTypeAsString(), "32-bit float")
        self.assertAlmostEqual(float(imgGI.GetMetaData("GIPR")), float(imgRefGI.GetMetaData("GIPR")), delta=1)
        self.assertAlmostEqual(ft.getGIstat(imgGI).passRate, ft.getGIstat(imgRefGI).passRate, delta=1)
        imgPR = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="global", stepSize=0.5, fractionalStepSize=False, mode="pass-rate")
        self._assertPassRateMap(imgRefGI, imgPR)

    def test_calcGammaIndex_DD2_DTA2_local_DCO5_stepSize03abs(self):
        imgRefGI = ft.readMHD(f"{GIImagesPath}/GI_DD2_DTA2_local_DCO5_stepSize0.3abs.mhd")
        imgGI = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="local", stepSize=0.3, fractionalStepSize=False)
        self.assertTrue(ft.compareImgFoR(imgRefGI, imgGI))
        self.assertTrue(ft.compareImg(imgRefGI, imgGI, decimal=3))
        self.assertEqual(imgGI.GetPixelIDTypeAsString(), "32-bit float")
        self.assertAlmostEqual(float(imgGI.GetMetaData("GIPR")), float(imgRefGI.GetMetaData("GIPR")), delta=1)
        self.assertAlmostEqual(ft.getGIstat(imgGI).passRate, ft.getGIstat(imgRefGI).passRate, delta=1)
        imgPR = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="local", stepSize=0.3, fractionalStepSize=False, mode="pass-rate")
        self._assertPassRateMap(imgRefGI, imgPR)

    def test_calcGammaIndex_DD2_DTA2_global_DCO10(self):
        imgRefGI = ft.readMHD(f"{GIImagesPath}/GI_DD2_DTA2_global_DCO10.mhd")
        imgGI = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.10, DDType="global")
        self.assertTrue(ft.compareImgFoR(imgRefGI, imgGI))
        self.assertTrue(ft.compareImg(imgRefGI, imgGI, decimal=3))
        self.assertEqual(imgGI.GetPixelIDTypeAsString(), "32-bit float")
        self.assertAlmostEqual(float(imgGI.GetMetaData("GIPR")), float(imgRefGI.GetMetaData("GIPR")), delta=1)
        self.assertAlmostEqual(ft.getGIstat(imgGI).passRate, ft.getGIstat(imgRefGI).passRate, delta=1)
        imgPR = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.10, DDType="global", mode="pass-rate")
        self._assertPassRateMap(imgRefGI, imgPR)

    def test_calcGammaIndex_metadata(self):
        imgGI = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="global")
        for key in ["GIVersion", "DD", "DTA", "DDType", "DCO", "stepSize", "mode", "GIPR"]:
            with self.subTest(key=key):
                self.assertTrue(imgGI.HasMetaDataKey(key))
        self.assertEqual(imgGI.GetMetaData("GIVersion"), "4.5")
        self.assertNotIn("\0", imgGI.GetMetaData("GIVersion"))
        self.assertEqual(imgGI.GetMetaData("DD"), "2")
        self.assertEqual(imgGI.GetMetaData("DTA"), "2")
        self.assertEqual(imgGI.GetMetaData("DDType"), "global")
        self.assertEqual(imgGI.GetMetaData("DCO"), "0.05")
        self.assertEqual(float(imgGI.GetMetaData("stepSize")), 0.2)
        self.assertEqual(imgGI.GetMetaData("mode"), "gamma")
        self.assertAlmostEqual(float(imgGI.GetMetaData("GIPR")), ft.getGIstat(imgGI).passRate, delta=0.01)

    def test_calcGammaIndex_deterministic(self):
        imgGI1 = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        imgGI2 = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        self.assertTrue(ft.compareImg(imgGI1, imgGI2, decimal=7))

    def test_calcGammaIndex_interrupted(self):
        from unittest import mock
        imgRefGI = ft.readMHD(f"{GIImagesPath}/GI_DD2_DTA2_global_DCO5.mhd")
        with self.assertLogs(ft.GammaIndex.gammaIndex._logger, level="WARNING"):
            with self.assertRaises(KeyboardInterrupt):
                with mock.patch("time.sleep", side_effect=KeyboardInterrupt):
                    ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="global")
        imgGI = ft.calcGammaIndex(self.imgRef, self.imgEval, DD=2, DTA=2, DCO=0.05, DDType="global")
        self.assertTrue(ft.compareImg(imgRefGI, imgGI, decimal=3))

    def test_calcGammaIndex_CPUNo(self):
        ft.CPUNO = "auto"
        imgGIAuto = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        ft.CPUNO = 1
        imgGISingle = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        self.assertTrue(ft.compareImg(imgGIAuto, imgGISingle, decimal=7))
        self.assertEqual(imgGIAuto.GetMetaData("GIPR"), imgGISingle.GetMetaData("GIPR"))

    def test_calcGammaIndex_globalNormDefault(self):
        maxRef = ft.getStatistics(self.imgGauss).GetMaximum()
        imgGIDefault = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        imgGIMax = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global", globalNorm=maxRef)
        self.assertTrue(ft.compareImg(imgGIDefault, imgGIMax, decimal=7))

    def test_calcGammaIndex_stepSizeEquivalence(self):
        imgGIFractional = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global", stepSize=10, fractionalStepSize=True)
        imgGIAbsolute = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global", stepSize=0.2, fractionalStepSize=False)
        self.assertTrue(ft.compareImg(imgGIFractional, imgGIAbsolute, decimal=7))
        self.assertEqual(float(imgGIFractional.GetMetaData("stepSize")), 0.2)
        self.assertEqual(float(imgGIAbsolute.GetMetaData("stepSize")), 0.2)

    def test_calcGammaIndex_DDTypeAbbreviation(self):
        imgGIGlobal = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        imgGILocal = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="local")
        for DDTypeAbbreviation in ["G", "g"]:
            with self.subTest(DDTypeAbbreviation=DDTypeAbbreviation):
                imgGIAbbreviation = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType=DDTypeAbbreviation)  # type: ignore
                self.assertTrue(ft.compareImg(imgGIGlobal, imgGIAbbreviation, decimal=7))
                self.assertEqual(imgGIAbbreviation.GetMetaData("DDType"), "global")
        for DDTypeAbbreviation in ["L", "l"]:
            with self.subTest(DDTypeAbbreviation=DDTypeAbbreviation):
                imgGIAbbreviation = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType=DDTypeAbbreviation)  # type: ignore
                self.assertTrue(ft.compareImg(imgGILocal, imgGIAbbreviation, decimal=7))
                self.assertEqual(imgGIAbbreviation.GetMetaData("DDType"), "local")

    def test_calcGammaIndex_modeAbbreviation(self):
        imgGIGamma = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global", mode="gamma")
        imgGIPassRate = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global", mode="pass-rate")
        for modeAbbreviation in ["g", "Gamma"]:
            with self.subTest(modeAbbreviation=modeAbbreviation):
                imgGIAbbreviation = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global", mode=modeAbbreviation)  # type: ignore
                self.assertTrue(ft.compareImg(imgGIGamma, imgGIAbbreviation, decimal=7))
                self.assertEqual(imgGIAbbreviation.GetMetaData("mode"), "gamma")
        for modeAbbreviation in ["pr", "p", "Pass-Rate"]:
            with self.subTest(modeAbbreviation=modeAbbreviation):
                imgGIAbbreviation = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global", mode=modeAbbreviation)  # type: ignore
                self.assertTrue(ft.compareImg(imgGIPassRate, imgGIAbbreviation, decimal=7))
                self.assertEqual(imgGIAbbreviation.GetMetaData("mode"), "pass-rate")

    def test_calcGammaIndex_identical(self):
        imgGI = ft.calcGammaIndex(self.imgGauss, self.imgGauss, DD=2, DTA=2, DCO=0.05, DDType="global")
        arrGI = ft.arr(imgGI)
        arrRef = ft.arr(sitk.Cast(self.imgGauss, sitk.sitkFloat32))
        self.assertTrue(np.any(arrGI < 0))
        self.assertTrue(np.array_equal(arrGI < 0, arrRef < np.float32(0.05 * arrRef.max())))
        self.assertEqual(arrGI[arrGI >= 0].max(), 0)
        self.assertEqual(float(imgGI.GetMetaData("GIPR")), 100)
        self.assertEqual(ft.getGIstat(imgGI).passRate, 100)

    def test_calcGammaIndex_uniformLocal(self):
        imgGI = ft.calcGammaIndex(self.imgUniform, self.imgUniform * 1.01, DD=2, DTA=2, DCO=0.05, DDType="local")
        statGI = ft.getGIstat(imgGI)
        self.assertAlmostEqual(statGI.min, 0.5, delta=1E-3)
        self.assertAlmostEqual(statGI.max, 0.5, delta=1E-3)
        self.assertEqual(statGI.passRate, 100)
        imgGI = ft.calcGammaIndex(self.imgUniform, self.imgUniform * 1.03, DD=2, DTA=2, DCO=0.05, DDType="local")
        statGI = ft.getGIstat(imgGI)
        self.assertAlmostEqual(statGI.min, 1.5, delta=1E-3)
        self.assertAlmostEqual(statGI.max, 1.5, delta=1E-3)
        self.assertEqual(statGI.passRate, 0)

    def test_calcGammaIndex_uniformGlobalNorm(self):
        imgGI = ft.calcGammaIndex(self.imgUniform, self.imgUniform * 1.01, DD=2, DTA=2, DCO=0.05, DDType="global", globalNorm=20)
        statGI = ft.getGIstat(imgGI)
        self.assertAlmostEqual(statGI.min, 0.25, delta=1E-3)
        self.assertAlmostEqual(statGI.max, 0.25, delta=1E-3)
        self.assertEqual(statGI.passRate, 100)

    def test_calcGammaIndex_gaussianShift(self):
        imgGI = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        statGI = ft.getGIstat(imgGI)
        self.assertAlmostEqual(statGI.max, 0.5, delta=0.02)
        self.assertEqual(statGI.passRate, 100)

    def test_calcGammaIndex_largeGamma(self):
        # gamma*DTA (about 99 mm) exceeds the reach of the 9 mm evaluation grid, so the search must end at the grid
        imgGI = ft.calcGammaIndex(_createGaussImg(size=9), _createGaussImg(size=9, amplitude=0.1), DD=2, DTA=2, DCO=0.05, DDType="global", stepSize=0.5, fractionalStepSize=False)
        statGI = ft.getGIstat(imgGI)
        self.assertAlmostEqual(statGI.max, (10 - 0.1) / 0.2, delta=1E-3)
        self.assertEqual(statGI.passRate, 0)

    def test_calcGammaIndex_flippedDirection(self):
        imgGI = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        for axis in range(3):
            with self.subTest(axis=axis):
                flipAxes = [axisFlip == axis for axisFlip in range(3)]
                imgRefFlipped = sitk.Flip(self.imgGauss, flipAxes)
                imgEvalFlipped = sitk.Flip(self.imgGaussShifted, flipAxes)
                imgGIFlipped = ft.calcGammaIndex(imgRefFlipped, imgEvalFlipped, DD=2, DTA=2, DCO=0.05, DDType="global")
                self.assertTrue(ft.compareImgFoR(imgRefFlipped, imgGIFlipped))
                self.assertTrue(ft.compareImg(imgGI, sitk.DICOMOrient(imgGIFlipped, "LPS"), decimal=5))
                self.assertEqual(imgGIFlipped.GetMetaData("GIPR"), imgGI.GetMetaData("GIPR"))

    def test_calcGammaIndex_permutedDirection(self):
        imgGI = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        imgRefPermuted = sitk.PermuteAxes(self.imgGauss, [1, 0, 2])
        imgEvalPermuted = sitk.PermuteAxes(self.imgGaussShifted, [1, 0, 2])
        imgGIPermuted = ft.calcGammaIndex(imgRefPermuted, imgEvalPermuted, DD=2, DTA=2, DCO=0.05, DDType="global")
        self.assertTrue(ft.compareImgFoR(imgRefPermuted, imgGIPermuted))
        self.assertTrue(ft.compareImg(imgGI, sitk.DICOMOrient(imgGIPermuted, "LPS"), decimal=5))

    def test_calcGammaIndex_obliqueDirection(self):
        imgGI = ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="global")
        direction = (np.cos(0.17), -np.sin(0.17), 0, np.sin(0.17), np.cos(0.17), 0, 0, 0, 1)
        imgRefOblique = sitk.Image(self.imgGauss)
        imgRefOblique.SetDirection(direction)
        imgEvalOblique = sitk.Image(self.imgGaussShifted)
        imgEvalOblique.SetDirection(direction)
        imgGIOblique = ft.calcGammaIndex(imgRefOblique, imgEvalOblique, DD=2, DTA=2, DCO=0.05, DDType="global")
        self.assertTrue(ft.compareImgFoR(imgRefOblique, imgGIOblique))
        self.assertTrue(np.allclose(ft.arr(imgGI), ft.arr(imgGIOblique), atol=1E-5))

    def test_calcGammaIndex_invalid_imgRef(self):
        with self.assertRaises(TypeError):
            ft.calcGammaIndex(ft.arr(self.imgGauss), self.imgGaussShifted, DD=2, DTA=2, DCO=0.05)  # type: ignore

    def test_calcGammaIndex_invalid_imgEval(self):
        with self.assertRaises(TypeError):
            ft.calcGammaIndex(self.imgGauss, ft.arr(self.imgGaussShifted), DD=2, DTA=2, DCO=0.05)  # type: ignore

    def test_calcGammaIndex_invalid_vectorImg(self):
        imgVector = sitk.Compose(self.imgGauss, self.imgGauss)
        with self.assertRaises(TypeError):
            ft.calcGammaIndex(imgVector, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05)
        with self.assertRaises(TypeError):
            ft.calcGammaIndex(self.imgGauss, imgVector, DD=2, DTA=2, DCO=0.05)

    def test_calcGammaIndex_invalid_2D(self):
        img2D = sitk.GetImageFromArray(np.ones((20, 20), dtype=np.float32))
        with self.assertRaises(NotImplementedError):
            ft.calcGammaIndex(img2D, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05)
        with self.assertRaises(NotImplementedError):
            ft.calcGammaIndex(self.imgGauss, img2D, DD=2, DTA=2, DCO=0.05)

    def test_calcGammaIndex_invalid_size1Axis(self):
        imgSlice = sitk.GetImageFromArray(np.ones((1, 20, 20), dtype=np.float32))
        with self.assertRaises(NotImplementedError):
            ft.calcGammaIndex(imgSlice, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05)
        with self.assertRaises(NotImplementedError):
            ft.calcGammaIndex(self.imgGauss, imgSlice, DD=2, DTA=2, DCO=0.05)

    def test_calcGammaIndex_invalid_directionMismatch(self):
        imgEvalFlipped = sitk.Flip(self.imgGaussShifted, [True, False, False])
        with self.assertRaises(ValueError):
            ft.calcGammaIndex(self.imgGauss, imgEvalFlipped, DD=2, DTA=2, DCO=0.05)
        with self.assertRaises(ValueError):
            ft.calcGammaIndex(imgEvalFlipped, self.imgGauss, DD=2, DTA=2, DCO=0.05)

    def test_calcGammaIndex_invalid_noOverlap(self):
        imgEvalDetached = sitk.Image(self.imgGaussShifted)
        for shiftX in [100, 41]:  # detached and touching (the images are 41 mm wide)
            imgEvalDetached.SetOrigin([-20 + shiftX, -20, -20])
            with self.subTest(shiftX=shiftX):
                with self.assertRaises(ValueError):
                    ft.calcGammaIndex(self.imgGauss, imgEvalDetached, DD=2, DTA=2, DCO=0.05)
                with self.assertRaises(ValueError):
                    ft.calcGammaIndex(imgEvalDetached, self.imgGauss, DD=2, DTA=2, DCO=0.05)

    def test_calcGammaIndex_invalid_DD(self):
        for DD in [0, 100, -1, "2"]:
            with self.subTest(DD=DD):
                with self.assertRaises(ValueError):
                    ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=DD, DTA=2, DCO=0.05)

    def test_calcGammaIndex_invalid_DTA(self):
        for DTA in [0, -1, "2"]:
            with self.subTest(DTA=DTA):
                with self.assertRaises(ValueError):
                    ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=DTA, DCO=0.05)

    def test_calcGammaIndex_invalid_DCO(self):
        for DCO in [0, 1, -0.1, "0.05"]:
            with self.subTest(DCO=DCO):
                with self.assertRaises(ValueError):
                    ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=DCO)

    def test_calcGammaIndex_invalid_globalNorm(self):
        for globalNorm in [0, -1, "10"]:
            with self.subTest(globalNorm=globalNorm):
                with self.assertRaises(ValueError):
                    ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, globalNorm=globalNorm)

    def test_calcGammaIndex_invalid_stepSize(self):
        for stepSize in [0, -1, "10"]:
            with self.subTest(stepSize=stepSize):
                with self.assertRaises(ValueError):
                    ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, stepSize=stepSize)

    def test_calcGammaIndex_invalid_DDType(self):
        with self.assertRaises(ValueError):
            ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, DDType="x")  # type: ignore

    def test_calcGammaIndex_invalid_mode(self):
        with self.assertRaises(ValueError):
            ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05, mode="x")  # type: ignore

    def test_calcGammaIndex_invalid_nonFinite(self):
        for value in [np.nan, np.inf]:
            arr = ft.arr(self.imgGauss)
            arr[0, 0, 0] = value
            imgNonFinite = sitk.GetImageFromArray(arr)
            imgNonFinite.CopyInformation(self.imgGauss)
            with self.subTest(value=value, image="reference"):
                with self.assertRaises(ValueError):
                    ft.calcGammaIndex(imgNonFinite, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05)
            with self.subTest(value=value, image="evaluation"):
                with self.assertRaises(ValueError):
                    ft.calcGammaIndex(self.imgGauss, imgNonFinite, DD=2, DTA=2, DCO=0.05)

    def test_calcGammaIndex_invalid_zeroReference(self):
        with self.assertRaises(ValueError):
            ft.calcGammaIndex(self.imgGauss * 0, self.imgGaussShifted, DD=2, DTA=2, DCO=0.05)

    def test_calcGammaIndex_invalid_cutoffAboveMax(self):
        maxRef = ft.getStatistics(self.imgGauss).GetMaximum()
        with self.assertRaises(ValueError):
            ft.calcGammaIndex(self.imgGauss, self.imgGaussShifted, DD=2, DTA=2, DCO=0.5, globalNorm=3 * maxRef)


class test_getGIstat(unittest.TestCase):

    def setUp(self):
        self.imgGamma = sitk.GetImageFromArray(np.array([[[-1, 0, 0.5, 1, 1.5, 2]]], dtype=np.float32))
        self.imgGammaNaN = sitk.GetImageFromArray(np.array([[[np.nan, 0, 0.5, 1, 1.5, 2]]], dtype=np.float32))
        self.imgPassRate = sitk.GetImageFromArray(np.array([[[-1, 0, 1, 1, 1, -1]]], dtype=np.int8))

    def test_getGIstat_gamma(self):
        statGI = ft.getGIstat(self.imgGamma, displayInfo=True)
        self.assertAlmostEqual(statGI.passRate, 60, places=5)
        self.assertAlmostEqual(statGI.mean, 1.0, places=5)
        self.assertAlmostEqual(statGI.std, np.sqrt(0.5), places=5)
        self.assertAlmostEqual(statGI.min, 0, places=5)
        self.assertAlmostEqual(statGI.max, 2, places=5)

    def test_getGIstat_gammaNaN(self):
        statGI = ft.getGIstat(self.imgGammaNaN)
        self.assertAlmostEqual(statGI.passRate, 60, places=5)
        self.assertAlmostEqual(statGI.mean, 1.0, places=5)
        self.assertAlmostEqual(statGI.std, np.sqrt(0.5), places=5)
        self.assertAlmostEqual(statGI.min, 0, places=5)
        self.assertAlmostEqual(statGI.max, 2, places=5)

    def test_getGIstat_gammaFloat64(self):
        statGI = ft.getGIstat(sitk.Cast(self.imgGamma, sitk.sitkFloat64))
        self.assertAlmostEqual(statGI.passRate, 60, places=5)
        self.assertAlmostEqual(statGI.mean, 1.0, places=5)

    def test_getGIstat_referenceMap(self):
        imgRefGI = ft.readMHD(f"{GIImagesPath}/GI_DD2_DTA2_global_DCO5.mhd")
        statGI = ft.getGIstat(imgRefGI)
        self.assertAlmostEqual(statGI.passRate, float(imgRefGI.GetMetaData("GIPR")), delta=0.01)
        self.assertGreaterEqual(statGI.min, 0)
        self.assertGreater(statGI.max, 1)

    def test_getGIstat_passRate(self):
        statGI = ft.getGIstat(self.imgPassRate, displayInfo=True)
        self.assertAlmostEqual(statGI.passRate, 75, places=5)
        self.assertTrue(np.isnan(statGI.mean))
        self.assertTrue(np.isnan(statGI.std))
        self.assertTrue(np.isnan(statGI.min))
        self.assertTrue(np.isnan(statGI.max))

    def test_getGIstat_allExcluded(self):
        statGI = ft.getGIstat(sitk.GetImageFromArray(np.full((2, 2, 2), -1, dtype=np.float32)))
        self.assertTrue(np.isnan(statGI.passRate))
        statGI = ft.getGIstat(sitk.GetImageFromArray(np.full((2, 2, 2), -1, dtype=np.int8)))
        self.assertTrue(np.isnan(statGI.passRate))

    def test_getGIstat_returnType(self):
        statGI = ft.getGIstat(self.imgGamma)
        self.assertIsInstance(statGI, dict)
        self.assertEqual(set(statGI.keys()), {"passRate", "mean", "std", "min", "max"})
        self.assertEqual(statGI.passRate, statGI["passRate"])

    def test_getGIstat_inputUnchanged(self):
        arrGamma = ft.arr(self.imgGamma)
        ft.getGIstat(self.imgGamma)
        self.assertTrue(np.array_equal(ft.arr(self.imgGamma), arrGamma))

    def test_getGIstat_invalid_values(self):
        with self.assertRaises(ValueError):
            ft.getGIstat(sitk.GetImageFromArray(np.array([[[-1, 0, 1, 2]]], dtype=np.int8)))

    def test_getGIstat_invalid_type(self):
        with self.assertRaises(TypeError):
            ft.getGIstat(sitk.GetImageFromArray(np.array([[[0, 1]]], dtype=np.complex64)))

    def test_getGIstat_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getGIstat(ft.arr(self.imgGamma))  # type: ignore


class test_getGIcmap(unittest.TestCase):

    def test_getGIcmap_type(self):
        from matplotlib.colors import LinearSegmentedColormap
        cmapGI = ft.getGIcmap(3)
        self.assertIsInstance(cmapGI, LinearSegmentedColormap)
        self.assertEqual(cmapGI.name, "GIcmap")
        self.assertEqual(cmapGI.N, 256)

    def test_getGIcmap_N(self):
        self.assertEqual(ft.getGIcmap(3, N=64).N, 64)

    def test_getGIcmap_colours(self):
        cmapGI = ft.getGIcmap(3)
        self.assertTrue(np.allclose(cmapGI(0.0)[:3], np.array([1, 0, 128]) / 255, atol=0.02))
        self.assertTrue(np.all(np.array(cmapGI(1 / 3 - 0.01)[:3]) > 0.9))
        self.assertTrue(np.allclose(cmapGI(1 / 3 + 0.01)[:3], np.array([254, 193, 192]) / 255, atol=0.02))
        self.assertTrue(np.allclose(cmapGI(1.0)[:3], np.array([255, 67, 66]) / 255, atol=0.02))

    def test_getGIcmap_maxGIClamp(self):
        cmapGI = ft.getGIcmap(1)
        cmapGIClamped = ft.getGIcmap(0.5)
        values = np.linspace(0, 1, 50)
        self.assertTrue(np.allclose(cmapGI(values), cmapGIClamped(values)))

    def test_getGIcmap_invalid_N(self):
        for N in [0, 1, -5, 2.5]:
            with self.subTest(N=N):
                with self.assertRaises(ValueError):
                    ft.getGIcmap(3, N=N)


class test__libFredGIFileName(unittest.TestCase):
    """The library file of each platform (sys.platform, platform.machine and the Python word size are patched)."""

    def _fileName(self, sysPlatform, machine, maxsize=2**63 - 1):
        from unittest import mock
        with mock.patch("sys.platform", sysPlatform), mock.patch("platform.machine", return_value=machine), mock.patch("sys.maxsize", maxsize):
            return ft.GammaIndex.gammaIndex._libFredGIFileName()

    def test__libFredGIFileName_supported(self):
        for sysPlatform, machine, fileName in (("linux", "x86_64", "libFredGI.so"),
                                               ("win32", "AMD64", "libFredGI.dll"),
                                               ("darwin", "arm64", "libFredGI.dylib"),
                                               ("darwin", "x86_64", "libFredGI.dylib")):
            with self.subTest(platform=sysPlatform, machine=machine):
                self.assertEqual(self._fileName(sysPlatform, machine), fileName)

    def test__libFredGIFileName_unsupported(self):
        for sysPlatform, machine, maxsize in (("linux", "aarch64", 2**63 - 1),
                                              ("win32", "ARM64", 2**63 - 1),
                                              ("freebsd14", "amd64", 2**63 - 1),
                                              ("win32", "AMD64", 2**31 - 1)):  # 32-bit Python
            with self.subTest(platform=sysPlatform, machine=machine, maxsize=maxsize):
                with self.assertLogs(ft.GammaIndex.gammaIndex._logger, level="ERROR"):
                    with self.assertRaises(OSError):
                        self._fileName(sysPlatform, machine, maxsize)

    def test__libFredGIFileName_shipped(self):
        # the library of the running platform is shipped with FREDtools
        import os
        fileName = ft.GammaIndex.gammaIndex._libFredGIFileName()
        self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(ft.GammaIndex.gammaIndex.__file__), fileName)))

    def test_calcGammaIndex_unsupportedPlatform(self):
        from unittest import mock
        imgGauss = _createGaussImg()
        with mock.patch("sys.platform", "freebsd14"):
            with self.assertLogs(ft.GammaIndex.gammaIndex._logger, level="ERROR"):
                with self.assertRaises(OSError):
                    ft.calcGammaIndex(imgGauss, imgGauss, DD=2, DTA=2, DCO=0.05, DDType="global")


if __name__ == '__main__':
    unittest.main()
