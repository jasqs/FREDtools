import unittest
import fredtools as ft
import numpy as np
import SimpleITK as sitk


def _halfwayImg(direction=None):
    """Small 3D image with distinct values (size 5x4x3, spacing 2x3x4 mm) for the tests of the nearest-neighbour
    interpolation at positions lying exactly halfway between two voxels."""
    arr = np.arange(10, 10 + 5 * 4 * 3, dtype="float32").reshape(3, 4, 5)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 3.0, 4.0))
    img.SetOrigin((1.0, -2.0, 3.0))
    if direction is not None:
        img.SetDirection(direction)
    return img


def _higherIndexValue(img, point):
    """Value of the voxel of `img` that the nearest-neighbour interpolation takes at `point`, with the rule written out:
    a position lying exactly halfway between two voxels takes the voxel with the higher index (numpy.nan outside)."""
    index = np.floor(np.array(img.TransformPhysicalPointToContinuousIndex([float(x) for x in point])) + 0.5 + 1e-6).astype(int)
    if np.any(index < 0) or np.any(index >= np.array(img.GetSize())):
        return np.nan
    return float(sitk.GetArrayViewFromImage(img)[tuple(index[::-1])])


class test_getSlice(unittest.TestCase):

    def setUp(self):
        self.img3D = ft.readMHD('unittests/testData/MHDImages/img3D.mhd')
        self.point = ft.getImageCenter(self.img3D)
        self.plane = "Y-Z"

    def test_getSlice_linear(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DSliceY-Z_resampleLinear.mhd")
        imgEval = ft.getSlice(self.img3D, point=self.point, plane=self.plane, interpolation="linear", displayInfo=True)
        self.assertTrue(ft.compareImg(imgRef, imgEval, decimal=7))

    def test_getSlice_nearest(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DSliceY-Z_resampleNearest.mhd")
        imgEval = ft.getSlice(self.img3D, point=self.point, plane=self.plane, interpolation='nearest', displayInfo=True)
        self.assertTrue(ft.compareImg(imgRef, imgEval, decimal=7))

    def test_getSlice_spline(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DSliceY-Z_resampleSpline3.mhd")
        imgEval = ft.getSlice(self.img3D, point=self.point, plane=self.plane, interpolation='spline', splineOrder=3, displayInfo=True)
        self.assertTrue(ft.compareImg(imgRef, imgEval, decimal=7))

    def test_getSlice_invalid_plane(self):
        with self.assertRaises(AttributeError):
            ft.getSlice(self.img3D, self.point, plane='CC')

    def test_getSlice_invalid_point(self):
        with self.assertRaises(AttributeError):
            ft.getSlice(self.img3D, [32, 32], plane='XY')


class test_getProfile(unittest.TestCase):

    def setUp(self):
        self.img3D = ft.readMHD('unittests/testData/MHDImages/img3D.mhd')
        self.point = ft.getImageCenter(self.img3D)
        self.axis = 'X'

    def test_getProfile_linear(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DProfileX_resampleLinear.mhd")
        imgEval = ft.getProfile(self.img3D, point=self.point, axis=self.axis, interpolation='linear', displayInfo=True)
        self.assertTrue(ft.compareImg(imgRef, imgEval, decimal=7))

    def test_getProfile_nearest(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DProfileX_resampleNearest.mhd")
        imgEval = ft.getProfile(self.img3D, point=self.point, axis=self.axis, interpolation='nearest', displayInfo=True)
        self.assertTrue(ft.compareImg(imgRef, imgEval, decimal=7))

    def test_getProfile_spline(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DProfileX_resampleSpline3.mhd")
        imgEval = ft.getProfile(self.img3D, point=self.point, axis=self.axis, interpolation='spline', splineOrder=3, displayInfo=True)
        self.assertTrue(ft.compareImg(imgRef, imgEval, decimal=7))

    def test_getProfile_invalid_axis(self):
        with self.assertRaises(AttributeError):
            ft.getProfile(self.img3D, self.point, axis='C')

    def test_getProfile_invalid_point(self):
        with self.assertRaises(AttributeError):
            ft.getProfile(self.img3D, [32, 32], axis='X')


class test_getPoint(unittest.TestCase):

    def setUp(self):
        self.img3D = ft.readMHD('unittests/testData/MHDImages/img3D.mhd')
        self.point = ft.getImageCenter(self.img3D)
        self.img3DVec = ft.readMHD('unittests/testData/MHDImages/img3DVec.mhd')
        self.pointImgVec = ft.getImageCenter(self.img3DVec)

    def test_getPoint_linear(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DPoint_resampleLinear.mhd")
        imgEval = ft.getPoint(self.img3D, self.point, interpolation='linear', displayInfo=True)
        self.assertEqual(imgRef.GetSize(), (1, 1, 1))
        self.assertAlmostEqual(ft.arr(imgRef), ft.arr(imgEval))

    def test_getPoint_nearest(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DPoint_resampleNearest.mhd")
        imgEval = ft.getPoint(self.img3D, self.point, interpolation='nearest', displayInfo=True)
        self.assertEqual(imgRef.GetSize(), (1, 1, 1))
        self.assertAlmostEqual(ft.arr(imgRef), ft.arr(imgEval))

    def test_getPoint_spline(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DPoint_resampleSpline3.mhd")
        imgEval = ft.getPoint(self.img3D, self.point, interpolation='spline', splineOrder=3, displayInfo=True)
        self.assertEqual(imgRef.GetSize(), (1, 1, 1))
        self.assertAlmostEqual(ft.arr(imgRef), ft.arr(imgEval))

    def test_getPoint_vectorImg(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DVecPoint_resampleNearest.mhd")
        imgEval = ft.getPoint(self.img3DVec, self.pointImgVec, interpolation='nearest', displayInfo=True)
        self.assertEqual(imgRef.GetSize(), (1, 1, 1))
        self.assertTrue(np.all(np.isclose(ft.arr(imgRef).tolist(), ft.arr(imgEval).tolist())))

    def test_getPoint_invalid_point(self):
        with self.assertRaises(AttributeError):
            ft.getPoint(self.img3D, [32, 32], displayInfo=True)


class test_getInteg(unittest.TestCase):

    def setUp(self):
        self.img3D = ft.readMHD('unittests/testData/MHDImages/img3D.mhd')
        self.axis = 'Y'

    def test_getInteg(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DIntegY.mhd")
        imgEval = ft.getInteg(self.img3D, axis=self.axis, displayInfo=True)
        self.assertTrue(ft.compareImg(imgRef, imgEval, decimal=7))

    def test_getInteg_invalid_axis(self):
        with self.assertRaises(AttributeError):
            ft.getInteg(self.img3D, axis='C')


class test_getCumSum(unittest.TestCase):

    def setUp(self):
        self.img3D = ft.readMHD('unittests/testData/MHDImages/img3D.mhd')
        self.axis = 'Y'

    def test_getCumSum(self):
        imgRef = ft.readMHD("unittests/testData/MHDImages/img3DCumSumY.mhd")
        imgEval = ft.getCumSum(self.img3D, axis=self.axis, displayInfo=True)
        self.assertTrue(ft.compareImg(imgRef, imgEval, decimal=7))

    def test_getCumSum_invalid_axis(self):
        with self.assertRaises(AttributeError):
            ft.getCumSum(self.img3D, axis='C')


class test_getHalfwayToHigherIndexTransform(unittest.TestCase):
    """The helper moves the sampling positions of the nearest-neighbour interpolation by a tiny fraction of a voxel
    towards the higher indices of the image, and leaves the other interpolations unchanged."""

    def setUp(self):
        self.img = _halfwayImg()
        self.point = (4.25, -1.5, 7.0)
        self.rotation = sitk.Euler3DTransform()
        self.rotation.SetRotation(0, 0, np.deg2rad(90))

    def test_getHalfwayToHigherIndexTransform_otherInterpolators(self):
        for interpolation in ("linear", "spline"):
            with self.subTest(interpolation=interpolation):
                interpolator = ft._helper.setSITKInterpolator(interpolation=interpolation)
                transform = ft._helper.getHalfwayToHigherIndexTransform(self.img, interpolator)
                np.testing.assert_array_equal(transform.TransformPoint(self.point), self.point)
                transform = ft._helper.getHalfwayToHigherIndexTransform(self.img, interpolator, self.rotation)
                np.testing.assert_array_equal(transform.TransformPoint(self.point), self.rotation.TransformPoint(self.point))

    def test_getHalfwayToHigherIndexTransform_nearest(self):
        transform = ft._helper.getHalfwayToHigherIndexTransform(self.img, sitk.sitkNearestNeighbor)
        np.testing.assert_allclose(np.subtract(transform.TransformPoint(self.point), self.point), np.array([2.0, 3.0, 4.0]) * 1e-9, rtol=1e-5)

    def test_getHalfwayToHigherIndexTransform_flippedAxis(self):
        # the shift goes towards the higher indices of the image, i.e. towards -x when the x axis is flipped
        img = _halfwayImg(direction=(-1, 0, 0, 0, 1, 0, 0, 0, 1))
        transform = ft._helper.getHalfwayToHigherIndexTransform(img, sitk.sitkNearestNeighbor)
        np.testing.assert_allclose(np.subtract(transform.TransformPoint(self.point), self.point), np.array([-2.0, 3.0, 4.0]) * 1e-9, rtol=1e-5)

    def test_getHalfwayToHigherIndexTransform_withTransform(self):
        # the given transform is applied first and the shift in the image space after it
        transform = ft._helper.getHalfwayToHigherIndexTransform(self.img, sitk.sitkNearestNeighbor, self.rotation)
        np.testing.assert_allclose(np.subtract(transform.TransformPoint(self.point), self.rotation.TransformPoint(self.point)), np.array([2.0, 3.0, 4.0]) * 1e-9, rtol=1e-5)

    def test_getHalfwayToHigherIndexTransform_shiftVoxelFraction(self):
        transform = ft._helper.getHalfwayToHigherIndexTransform(self.img, sitk.sitkNearestNeighbor, shiftVoxelFraction=1e-3)
        np.testing.assert_allclose(np.subtract(transform.TransformPoint(self.point), self.point), np.array([2.0, 3.0, 4.0]) * 1e-3, rtol=1e-9)


class test_nearestHalfway(unittest.TestCase):
    """With the nearest-neighbour interpolation, a position lying exactly halfway between two voxels takes the voxel
    with the higher index, also when the position is moved by 1e-12 mm (as by the floating-point rounding of
    another platform). The positions are given as continuous indices of the image (x.5 = halfway)."""

    def setUp(self):
        self.img = _halfwayImg()
        self.arr = sitk.GetArrayFromImage(self.img)  # [z, y, x]
        self.noises = ([0, 0, 0], [1e-12, 1e-12, 1e-12], [-1e-12, -1e-12, -1e-12])

    def point(self, continuousIndex, noise=(0, 0, 0)):
        return np.array(self.img.TransformContinuousIndexToPhysicalPoint(continuousIndex)) + noise

    def test_getSlice(self):
        # halfway between the slices z=0 and z=1 -> the slice z=1
        for noise in self.noises:
            with self.subTest(noise=noise):
                sl = ft.getSlice(self.img, point=self.point((2, 1, 0.5), noise), plane="XY", interpolation="nearest")
                np.testing.assert_array_equal(sitk.GetArrayFromImage(sl)[0], self.arr[1])

    def test_getProfile(self):
        # halfway in y (1|2) and z (0|1) -> the row y=2, z=1
        for noise in self.noises:
            with self.subTest(noise=noise):
                prof = ft.getProfile(self.img, point=self.point((2, 1.5, 0.5), noise), axis="X", interpolation="nearest")
                np.testing.assert_array_equal(sitk.GetArrayFromImage(prof).ravel(), self.arr[1, 2, :])

    def test_getPoint(self):
        # halfway in all three axes -> the voxel x=2, y=3, z=1
        for noise in self.noises:
            with self.subTest(noise=noise):
                value = ft.getPoint(self.img, self.point((1.5, 2.5, 0.5), noise), interpolation="nearest")
                self.assertEqual(float(ft.arr(value)), self.arr[1, 3, 2])

    def test_getPoint_flippedAxis(self):
        # the higher index of the image, not the higher physical coordinate
        img = _halfwayImg(direction=(-1, 0, 0, 0, 1, 0, 0, 0, 1))
        for noise in self.noises:
            with self.subTest(noise=noise):
                point = np.array(img.TransformContinuousIndexToPhysicalPoint((1.5, 2.5, 0.5))) + noise
                value = ft.getPoint(img, point, interpolation="nearest")
                self.assertEqual(float(ft.arr(value)), self.arr[1, 3, 2])

    def test_getProfilePoints(self):
        # along x through y halfway (1|2) and z halfway (0|1) -> the row y=2, z=1
        for noise in self.noises:
            with self.subTest(noise=noise):
                _, values = ft.getProfilePoints(self.img, self.point((0, 1.5, 0.5), noise), self.point((4, 1.5, 0.5), noise), spacing=2, interpolation="nearest")
                np.testing.assert_array_equal(values, self.arr[1, 2, :])

    def test_getProfilePoints_upperBorder(self):
        # points halfway along x (0.5, 1.5, ..., 4.5); the last one lies exactly on the upper border of the image
        pointA, pointB = self.point((0.5, 1, 1)), self.point((4.5, 1, 1))
        points = [self.point((x, 1, 1)) for x in (0.5, 1.5, 2.5, 3.5, 4.5)]
        with self.subTest(interpolation="nearest"):
            # the border point is outside for the nearest-neighbour interpolation (like in getPoint, getSlice, getProfile)
            _, values = ft.getProfilePoints(self.img, pointA, pointB, spacing=2, interpolation="nearest")
            np.testing.assert_array_equal(values, [_higherIndexValue(self.img, point) for point in points])
            self.assertTrue(np.isnan(values[-1]))
        with self.subTest(interpolation="linear"):
            # the other interpolations are not changed
            _, values = ft.getProfilePoints(self.img, pointA, pointB, spacing=2, interpolation="linear")
            np.testing.assert_allclose(values, [self.img.EvaluateAtPhysicalPoint([float(x) for x in point], sitk.sitkLinear) for point in points])


class test_splineInterpolation(unittest.TestCase):
    """The spline interpolation is a B-spline interpolation: it passes through the voxel values and between the
    voxel centres equals the B-spline interpolation of SimpleITK (sitkBSpline1..5)."""

    def setUp(self):
        rng = np.random.default_rng(9)
        self.arr = rng.uniform(0, 100, (5, 6, 9))  # [z, y, x]
        self.img = sitk.GetImageFromArray(self.arr)
        self.img.SetSpacing((2.0, 3.0, 4.0))
        self.img.SetOrigin((1.0, -2.0, 3.0))

    def point(self, img, continuousIndex):
        return np.array(img.TransformContinuousIndexToPhysicalPoint(continuousIndex))

    def test_getPoint_singleVoxel(self):
        # a single bright voxel keeps its value (a smoothing would lower it, e.g. to 29.6 for order 3)
        arr = np.zeros((7, 7, 7))
        arr[3, 3, 3] = 100
        img = sitk.GetImageFromArray(arr)
        for splineOrder in range(1, 6):
            with self.subTest(splineOrder=splineOrder):
                value = ft.getPoint(img, self.point(img, (3, 3, 3)), interpolation="spline", splineOrder=splineOrder)
                self.assertAlmostEqual(float(ft.arr(value)), 100, delta=1e-6)

    def test_getProfilePoints_voxelCentres(self):
        # along x through the voxel centres y=2, z=1 -> the voxel values
        for splineOrder in range(1, 6):
            with self.subTest(splineOrder=splineOrder):
                _, values = ft.getProfilePoints(self.img, self.point(self.img, (0, 2, 1)), self.point(self.img, (8, 2, 1)), spacing=2, interpolation="spline", splineOrder=splineOrder)
                np.testing.assert_allclose(values, self.arr[1, 2, :], rtol=0, atol=1e-6)

    def test_getProfilePoints_betweenVoxelCentres(self):
        pointA, pointB = self.point(self.img, (0.4, 2.3, 1.6)), self.point(self.img, (7.7, 3.4, 2.9))
        for splineOrder in range(1, 6):
            with self.subTest(splineOrder=splineOrder):
                _, values = ft.getProfilePoints(self.img, pointA, pointB, spacing=1.7, interpolation="spline", splineOrder=splineOrder)
                points = ft.ImgManipulate.imgGetSubimg._calcPointsAlongLine(pointA, pointB, 1.7)
                valuesRef = [self.img.EvaluateAtPhysicalPoint([float(x) for x in point], getattr(sitk, f"sitkBSpline{splineOrder}")) if ft.isPointInside(self.img, point) else np.nan for point in points]
                np.testing.assert_allclose(values, valuesRef, rtol=1e-12, atol=1e-9)

    def test_getProfilePoints_vector(self):
        # each component of a vector image is interpolated like the scalar image of that component
        arrs = [self.arr, self.arr[::-1].copy(), 2 * self.arr + 5]
        components = []
        for arr in arrs:
            component = sitk.GetImageFromArray(arr)
            component.CopyInformation(self.img)
            components.append(component)
        imgVec = sitk.Compose(components)
        pointA, pointB = self.point(self.img, (0.4, 2.3, 1.6)), self.point(self.img, (7.7, 3.4, 2.9))
        _, values = ft.getProfilePoints(imgVec, pointA, pointB, spacing=1.7, interpolation="spline", splineOrder=3)
        for idx, component in enumerate(components):
            with self.subTest(component=idx):
                _, valuesComponent = ft.getProfilePoints(component, pointA, pointB, spacing=1.7, interpolation="spline", splineOrder=3)
                np.testing.assert_allclose([value[idx] for value in values], valuesComponent, rtol=1e-12, atol=1e-9)


class test_getImgPointInterpolator(unittest.TestCase):
    """The image point interpolator gives the same values as SimpleITK's EvaluateAtPhysicalPoint, with the B-spline
    coefficients computed only once for the spline interpolation."""

    def setUp(self):
        self.img = sitk.GetImageFromArray(np.random.default_rng(10).uniform(0, 100, (5, 6, 9)).astype("float32"))
        self.img.SetSpacing((2.0, 3.0, 4.0))
        self.points = [self.img.TransformContinuousIndexToPhysicalPoint(ci) for ci in ((0.4, 2.3, 1.6), (4.0, 3.0, 2.0), (7.7, 3.4, 2.9))]

    def test_getImgPointInterpolator(self):
        for interpolator in (sitk.sitkNearestNeighbor, sitk.sitkLinear, sitk.sitkBSpline1, sitk.sitkBSpline2, sitk.sitkBSpline3, sitk.sitkBSpline4, sitk.sitkBSpline5):
            with self.subTest(interpolator=interpolator):
                interpolate = ft._helper.getImgPointInterpolator(self.img, interpolator)
                np.testing.assert_allclose([interpolate(point) for point in self.points], [self.img.EvaluateAtPhysicalPoint(point, interpolator) for point in self.points], rtol=1e-12, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
