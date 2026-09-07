import unittest
import numpy as np
import SimpleITK as sitk
from lmfit.model import ModelResult
import fredtools as ft


def _createSpotImg(amplitude=10.0, centre=(2.0, -3.0), sigma=(3.0, 5.0), rotation=0.0, noise=0.0):
    """Create a 2D image (101x81 voxels of 0.5 mm) of a rotated anisotropic Gaussian spot, optionally with Gaussian noise."""
    img = ft.createImg([101, 81], spacing=[0.5, 0.5], origin=[-25.0, -20.0])
    X, Y = np.meshgrid(*ft.getVoxelCentres(img), indexing="ij")
    rotationRad = np.deg2rad(rotation)
    u = (X - centre[0]) * np.cos(rotationRad) + (Y - centre[1]) * np.sin(rotationRad)
    v = (X - centre[0]) * np.sin(rotationRad) - (Y - centre[1]) * np.cos(rotationRad)
    arr = amplitude * np.exp(-u**2 / (2 * sigma[0]**2) - v**2 / (2 * sigma[1]**2))
    if noise:
        arr += np.random.default_rng(0).normal(0, noise, arr.shape)
    imgSpot = sitk.GetImageFromArray(arr.T)
    imgSpot.CopyInformation(img)
    return imgSpot


def _createSpotsImg(spots):
    """Create a 2D image (200x160 voxels of 0.5 mm) with isotropic Gaussian spots given as a list of (x, y, amplitude, sigma)."""
    img = ft.createImg([200, 160], spacing=[0.5, 0.5], origin=[-50.0, -40.0])
    X, Y = np.meshgrid(*ft.getVoxelCentres(img), indexing="ij")
    arr = np.zeros_like(X)
    for x, y, amplitude, sigma in spots:
        arr += amplitude * np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    imgSpots = sitk.GetImageFromArray(arr.T)
    imgSpots.CopyInformation(img)
    return imgSpots


def _labelSizes(imgLabel):
    """Get a dictionary with the number of voxels of each label of a label image."""
    arrLabel = ft.arr(imgLabel)
    return {int(label): int((arrLabel == label).sum()) for label in np.unique(arrLabel[arrLabel > 0])}


class test_findSpots(unittest.TestCase):

    def setUp(self):
        self.spots = [(-20.0, -10.0, 10.0, 3.0), (20.0, 15.0, 8.0, 5.0), (0.0, -25.0, 6.0, 2.0)]
        self.imgSpots = _createSpotsImg(self.spots)
        self.imgSingleSpot = _createSpotsImg([(0.0, 0.0, 5.0, 4.0)])
        self.imgOverlappingSpots = _createSpotsImg([(0.0, 0.0, 5.0, 4.0), (6.0, 0.0, 5.0, 4.0)])
        self.imgOmniPro = ft.readOPG("unittests/testData/OmniPro/image.opg")[:, :, 0]

    def _getLabelAtPoint(self, imgLabel, point):
        return imgLabel[imgLabel.TransformPhysicalPointToIndex([float(coordinate) for coordinate in point])]

    def test_findSpots(self):
        imgLabel = ft.findSpots(self.imgSpots, displayInfo=True)
        self.assertEqual(imgLabel.GetPixelIDTypeAsString(), "8-bit unsigned integer")
        self.assertTrue(ft.compareImgFoR(self.imgSpots, imgLabel))
        self.assertEqual(sorted(_labelSizes(imgLabel).keys()), [1, 2, 3])

    def test_findSpots_sorted_by_size(self):
        imgLabel = ft.findSpots(self.imgSpots)
        labelSizes = _labelSizes(imgLabel)
        self.assertGreater(labelSizes[1], labelSizes[2])
        self.assertGreater(labelSizes[2], labelSizes[3])
        self.assertEqual(self._getLabelAtPoint(imgLabel, (20.0, 15.0)), 1)
        self.assertEqual(self._getLabelAtPoint(imgLabel, (-20.0, -10.0)), 2)
        self.assertEqual(self._getLabelAtPoint(imgLabel, (0.0, -25.0)), 3)

    def test_findSpots_bounding_box(self):
        arrLabel = ft.arr(ft.findSpots(self.imgSpots))
        for label, size in _labelSizes(ft.findSpots(self.imgSpots)).items():
            with self.subTest(label=label):
                labelIdx = np.argwhere(arrLabel == label)
                boundingBoxArea = np.prod(labelIdx.max(axis=0) - labelIdx.min(axis=0) + 1)
                self.assertEqual(size, boundingBoxArea)

    def test_findSpots_covers_spot_region(self):
        arrLabel = ft.arr(ft.findSpots(self.imgSpots, DCO=0.1))
        arrSpots = ft.arr(self.imgSpots)
        self.assertTrue(np.all(arrLabel[arrSpots >= 0.1 * arrSpots.max()] > 0))

    def test_findSpots_margin(self):
        labelSizesDefault = _labelSizes(ft.findSpots(self.imgSpots, margin=3))
        labelSizesSmall = _labelSizes(ft.findSpots(self.imgSpots, margin=0.1))
        labelSizesLarge = _labelSizes(ft.findSpots(self.imgSpots, margin=6))
        labelSizesIterable = _labelSizes(ft.findSpots(self.imgSpots, margin=[6, 1]))
        for label in [1, 2, 3]:
            with self.subTest(label=label):
                self.assertLess(labelSizesSmall[label], labelSizesDefault[label])
                self.assertLess(labelSizesDefault[label], labelSizesLarge[label])
                self.assertLess(labelSizesIterable[label], labelSizesLarge[label])

    def test_findSpots_DCO(self):
        self.assertEqual(len(_labelSizes(ft.findSpots(self.imgSpots, DCO=0.1))), 3)
        self.assertEqual(len(_labelSizes(ft.findSpots(self.imgSpots, DCO=0.5))), 2)
        self.assertEqual(len(_labelSizes(ft.findSpots(self.imgSpots, DCO=0.9))), 0)

    def test_findSpots_single_spot(self):
        imgLabel = ft.findSpots(self.imgSingleSpot)
        self.assertEqual(list(_labelSizes(imgLabel).keys()), [1])
        self.assertEqual(self._getLabelAtPoint(imgLabel, (0.0, 0.0)), 1)

    def test_findSpots_overlapping_spots(self):
        imgLabel = ft.findSpots(self.imgOverlappingSpots)
        self.assertEqual(list(_labelSizes(imgLabel).keys()), [1])
        self.assertEqual(self._getLabelAtPoint(imgLabel, (0.0, 0.0)), 1)
        self.assertEqual(self._getLabelAtPoint(imgLabel, (6.0, 0.0)), 1)

    def test_findSpots_OmniPro(self):
        imgLabel = ft.findSpots(self.imgOmniPro, DCO=0.5, margin=8, displayInfo=True)
        self.assertEqual(list(_labelSizes(imgLabel).keys()), [1])
        self.assertEqual(self._getLabelAtPoint(imgLabel, ft.getMaxPosition(self.imgOmniPro)), 1)

    def test_findSpots_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.findSpots(ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0]))
        with self.assertRaises(TypeError):
            ft.findSpots(ft.arr(self.imgSpots))  # type: ignore

    def test_findSpots_invalid_DCO(self):
        for DCO in [0, 1, -0.1, 1.5, "0.1"]:
            with self.subTest(DCO=DCO):
                with self.assertRaises(ValueError):
                    ft.findSpots(self.imgSpots, DCO=DCO)  # type: ignore

    def test_findSpots_invalid_margin(self):
        with self.assertRaises(TypeError):
            ft.findSpots(self.imgSpots, margin=None)  # type: ignore


class test_fitSpotProfile(unittest.TestCase):

    def setUp(self):
        self.amplitude = 7.0
        self.centre = 1.5
        self.sigma = 2.5
        self.pos = np.linspace(-20.0, 20.0, 81)
        self.vec = self.amplitude * np.exp(-(self.pos - self.centre)**2 / (2 * self.sigma**2))
        self.vecCentred = self.amplitude * np.exp(-self.pos**2 / (2 * self.sigma**2))

    def test_fitSpotProfile(self):
        result = ft.fitSpotProfile(self.pos, self.vec)
        self.assertIsInstance(result, ModelResult)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.best_values["amplitude"], self.amplitude, places=4)
        self.assertAlmostEqual(result.best_values["centre"], self.centre, places=4)
        self.assertAlmostEqual(result.best_values["sigma"], self.sigma, places=4)
        self.assertTrue(np.allclose(result.best_fit, self.vec, atol=1E-6))

    def test_fitSpotProfile_lists(self):
        result = ft.fitSpotProfile(list(self.pos), list(self.vec))
        self.assertAlmostEqual(result.best_values["amplitude"], self.amplitude, places=4)
        self.assertAlmostEqual(result.best_values["centre"], self.centre, places=4)
        self.assertAlmostEqual(result.best_values["sigma"], self.sigma, places=4)

    def test_fitSpotProfile_cutLevel(self):
        result = ft.fitSpotProfile(self.pos, self.vec, cutLevel=0.5)
        self.assertEqual(result.ndata, int((self.vec >= 0.5 * self.vec.max()).sum()))
        self.assertAlmostEqual(result.best_values["amplitude"], self.amplitude, places=4)
        self.assertAlmostEqual(result.best_values["centre"], self.centre, places=4)
        self.assertAlmostEqual(result.best_values["sigma"], self.sigma, places=4)

    def test_fitSpotProfile_fixAmplitude(self):
        result = ft.fitSpotProfile(self.pos, self.vec, fixAmplitude=True)
        self.assertFalse(result.params["amplitude"].vary)
        self.assertEqual(result.best_values["amplitude"], self.vec.max())
        self.assertAlmostEqual(result.best_values["centre"], self.centre, places=4)
        self.assertAlmostEqual(result.best_values["sigma"], self.sigma, places=4)

    def test_fitSpotProfile_fixCentreToZero(self):
        result = ft.fitSpotProfile(self.pos, self.vecCentred, fixCentreToZero=True)
        self.assertFalse(result.params["centre"].vary)
        self.assertEqual(result.best_values["centre"], 0)
        self.assertAlmostEqual(result.best_values["amplitude"], self.amplitude, places=4)
        self.assertAlmostEqual(result.best_values["sigma"], self.sigma, places=4)

    def test_fitSpotProfile_noise(self):
        vecNoise = self.vec + np.random.default_rng(1).normal(0, 0.1, self.vec.shape)
        result = ft.fitSpotProfile(self.pos, vecNoise)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.best_values["amplitude"], self.amplitude, delta=0.2)
        self.assertAlmostEqual(result.best_values["centre"], self.centre, delta=0.1)
        self.assertAlmostEqual(result.best_values["sigma"], self.sigma, delta=0.1)

    def test_fitSpotProfile_method_case(self):
        result = ft.fitSpotProfile(self.pos, self.vec, method="SINGLEGAUSS")  # type: ignore
        self.assertAlmostEqual(result.best_values["sigma"], self.sigma, places=4)

    def test_fitSpotProfile_invalid_method(self):
        with self.assertRaises(ValueError):
            ft.fitSpotProfile(self.pos, self.vec, method="doubleGauss")  # type: ignore

    def test_fitSpotProfile_invalid_input(self):
        for pos, vec in [(5.0, self.vec), (self.pos, 5.0), (np.zeros((2, 3)), np.zeros((2, 3))), (self.pos[:-1], self.vec)]:
            with self.subTest(pos=np.shape(pos), vec=np.shape(vec)):
                with self.assertRaises(TypeError):
                    ft.fitSpotProfile(pos, vec)  # type: ignore


class test_fitSpotImg(unittest.TestCase):

    def setUp(self):
        self.amplitude = 10.0
        self.centre = (2.0, -3.0)
        self.sigma = (3.0, 5.0)
        self.imgSpot = _createSpotImg(self.amplitude, self.centre, self.sigma)
        self.imgSpotCentred = _createSpotImg(self.amplitude, (0.0, 0.0), self.sigma)

    def _assertSpotFit(self, result, amplitude, centre, sigma, rotation, delta=1E-3):
        """Check the fitted parameters, accounting for the equivalence of swapping the sigmas with a rotation by 90 degrees."""
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.best_values["amplitude"], amplitude, delta=delta * amplitude)
        self.assertAlmostEqual(result.best_values["centerX"], centre[0], delta=delta)
        self.assertAlmostEqual(result.best_values["centerY"], centre[1], delta=delta)
        if abs(result.best_values["sigmaX"] - sigma[0]) < abs(result.best_values["sigmaX"] - sigma[1]):
            expectedSigma, expectedRotation = sigma, rotation
        else:
            expectedSigma, expectedRotation = sigma[::-1], rotation + 90
        self.assertAlmostEqual(result.best_values["sigmaX"], expectedSigma[0], delta=delta)
        self.assertAlmostEqual(result.best_values["sigmaY"], expectedSigma[1], delta=delta)
        rotationDifference = (result.best_values["rotation"] - expectedRotation) % 180
        self.assertLess(min(rotationDifference, 180 - rotationDifference), delta * 100)

    def test_fitSpotImg(self):
        result = ft.fitSpotImg(self.imgSpot)
        self.assertIsInstance(result, ModelResult)
        self._assertSpotFit(result, self.amplitude, self.centre, self.sigma, 0.0)

    def test_fitSpotImg_best_fit(self):
        result = ft.fitSpotImg(self.imgSpot)
        self.assertEqual(result.best_fit.shape, sitk.GetArrayFromImage(self.imgSpot).shape)
        self.assertTrue(np.allclose(result.best_fit, sitk.GetArrayFromImage(self.imgSpot), atol=1E-6))

    def test_fitSpotImg_rotated(self):
        for rotation in [20.0, 45.0, 70.0, 90.0, 135.0]:
            with self.subTest(rotation=rotation):
                result = ft.fitSpotImg(_createSpotImg(self.amplitude, self.centre, self.sigma, rotation=rotation))
                self._assertSpotFit(result, self.amplitude, self.centre, self.sigma, rotation)

    def test_fitSpotImg_cutLevel(self):
        maxValue = ft.getStatistics(self.imgSpot).GetMaximum()
        imgSpotCut = sitk.Threshold(self.imgSpot, lower=0.5 * maxValue, upper=1.1 * maxValue, outsideValue=0)
        result = ft.fitSpotImg(self.imgSpot, cutLevel=0.5)
        resultCut = ft.fitSpotImg(imgSpotCut)
        for parameter in result.best_values:
            with self.subTest(parameter=parameter):
                self.assertAlmostEqual(result.best_values[parameter], resultCut.best_values[parameter], places=6)

    def test_fitSpotImg_fixAmplitude(self):
        result = ft.fitSpotImg(self.imgSpot, fixAmplitude=True)
        self.assertFalse(result.params["amplitude"].vary)
        self.assertEqual(result.best_values["amplitude"], ft.getStatistics(self.imgSpot).GetMaximum())
        self._assertSpotFit(result, ft.getStatistics(self.imgSpot).GetMaximum(), self.centre, self.sigma, 0.0)

    def test_fitSpotImg_fixCentreToZero(self):
        result = ft.fitSpotImg(self.imgSpotCentred, fixCentreToZero=True)
        self.assertFalse(result.params["centerX"].vary)
        self.assertFalse(result.params["centerY"].vary)
        self.assertEqual(result.best_values["centerX"], 0)
        self.assertEqual(result.best_values["centerY"], 0)
        self._assertSpotFit(result, self.amplitude, (0.0, 0.0), self.sigma, 0.0)

    def test_fitSpotImg_noise(self):
        result = ft.fitSpotImg(_createSpotImg(self.amplitude, self.centre, self.sigma, rotation=30.0, noise=0.2))
        self._assertSpotFit(result, self.amplitude, self.centre, self.sigma, 30.0, delta=0.05)

    def test_fitSpotImg_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.fitSpotImg(ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0]))
        with self.assertRaises(TypeError):
            ft.fitSpotImg(ft.arr(self.imgSpot))  # type: ignore

    def test_fitSpotImg_invalid_method(self):
        with self.assertRaises(ValueError):
            ft.fitSpotImg(self.imgSpot, method="doubleGauss")  # type: ignore


class test_fitSigmaSquaredModel(unittest.TestCase):

    def setUp(self):
        self.a = 4.0
        self.b = 0.01
        self.c = 0.0002
        self.pos = np.linspace(-100.0, 100.0, 21)
        self.beamSize = np.sqrt(self.a + self.b * self.pos + self.c * self.pos**2)

    def test_fitSigmaSquaredModel(self):
        result = ft.fitSigmaSquaredModel(self.pos, self.beamSize)
        self.assertIsInstance(result, ModelResult)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.best_values["a"], self.a, places=6)
        self.assertAlmostEqual(result.best_values["b"], self.b, places=6)
        self.assertAlmostEqual(result.best_values["c"], self.c, places=6)

    def test_fitSigmaSquaredModel_lists(self):
        result = ft.fitSigmaSquaredModel(list(self.pos), list(self.beamSize))
        self.assertAlmostEqual(result.best_values["a"], self.a, places=6)
        self.assertAlmostEqual(result.best_values["b"], self.b, places=6)
        self.assertAlmostEqual(result.best_values["c"], self.c, places=6)

    def test_fitSigmaSquaredModel_best_fit(self):
        result = ft.fitSigmaSquaredModel(self.pos, self.beamSize)
        self.assertTrue(np.allclose(result.best_fit, self.beamSize**2, atol=1E-6))

    def test_fitSigmaSquaredModel_noise(self):
        beamSizeNoise = self.beamSize * (1 + np.random.default_rng(2).normal(0, 0.005, self.beamSize.shape))
        result = ft.fitSigmaSquaredModel(self.pos, beamSizeNoise)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.best_values["a"], self.a, delta=0.1)
        self.assertAlmostEqual(result.best_values["b"], self.b, delta=0.001)
        self.assertAlmostEqual(result.best_values["c"], self.c, delta=0.00002)

    def test_fitSigmaSquaredModel_positive_c(self):
        beamSizeNegativeCurvature = np.sqrt(9.0 - 0.0003 * self.pos**2)
        result = ft.fitSigmaSquaredModel(self.pos, beamSizeNegativeCurvature)
        self.assertEqual(result.params["c"].min, 0.0)
        self.assertAlmostEqual(result.best_values["c"], 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
