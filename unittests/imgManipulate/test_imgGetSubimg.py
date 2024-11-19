import unittest
import SimpleITK as sitk
import numpy as np
import fredtools as ft


class test_GetSlice(unittest.TestCase):

    def setUp(self):
        self.img3D = sitk.GaussianSource(size=[64, 64, 64], mean=[32, 32, 32], sigma=[10, 10, 10])
        self.point = [32, 32, 32]

    def test_getSlice(self):
        slice_img = ft.getSlice(self.img3D, self.point, plane='-YX', displayInfo=True)
        self.assertEqual(slice_img.GetDimension(), 3)
        self.assertEqual(slice_img.GetSize()[2], 1)

    def test_getSlice_invalid_plane(self):
        with self.assertRaises(AttributeError):
            ft.getSlice(self.img3D, self.point, plane='CC')

    def test_getSlice_invalid_point(self):
        with self.assertRaises(AttributeError):
            ft.getSlice(self.img3D, [32, 32], plane='XY')


class test_GetProfile(unittest.TestCase):

    def setUp(self):
        self.img3D = sitk.GaussianSource(size=[64, 64, 64], mean=[32, 32, 32], sigma=[10, 10, 10])
        self.point = [32, 32, 32]

    def test_getProfile(self):
        profile_img = ft.getProfile(self.img3D, self.point, axis='-X', displayInfo=True)
        self.assertEqual(profile_img.GetDimension(), 3)
        self.assertEqual(profile_img.GetSize()[0], 64)

    def test_getProfile_invalid_axis(self):
        with self.assertRaises(AttributeError):
            ft.getProfile(self.img3D, self.point, axis='C')

    def test_getProfile_invalid_point(self):
        with self.assertRaises(AttributeError):
            ft.getProfile(self.img3D, [32, 32], axis='X')


class test_GetPoint(unittest.TestCase):

    def setUp(self):
        self.img3D = sitk.GaussianSource(size=[64, 64, 64], mean=[32, 32, 32], sigma=[10, 10, 10])
        self.point = [32, 32, 32]

    def test_getPoint(self):
        point_img = ft.getPoint(self.img3D, self.point, displayInfo=True)
        self.assertEqual(point_img.GetDimension(), 3)
        self.assertEqual(point_img.GetSize(), (1, 1, 1))

    def test_getPoint_invalid_point(self):
        with self.assertRaises(AttributeError):
            ft.getPoint(self.img3D, [32, 32], displayInfo=True)


class test_GetInteg(unittest.TestCase):

    def setUp(self):
        self.img3D = sitk.GaussianSource(size=[64, 64, 64], mean=[32, 32, 32], sigma=[10, 10, 10])

    def test_getInteg(self):
        integ_img = ft.getInteg(self.img3D, axis='X', displayInfo=True)
        self.assertEqual(integ_img.GetDimension(), 3)
        self.assertEqual(integ_img.GetSize()[0], 64)

    def test_getInteg_invalid_axis(self):
        with self.assertRaises(AttributeError):
            ft.getInteg(self.img3D, axis='C')


class test_GetCumSum(unittest.TestCase):

    def setUp(self):
        self.img3D = sitk.GaussianSource(size=[64, 64, 64], mean=[32, 32, 32], sigma=[10, 10, 10])

    def test_getCumSum(self):
        cumsum_img = ft.getCumSum(self.img3D, axis='X', displayInfo=True)
        self.assertEqual(cumsum_img.GetDimension(), 3)
        self.assertEqual(cumsum_img.GetSize(), self.img3D.GetSize())

    def test_getCumSum_invalid_axis(self):
        with self.assertRaises(AttributeError):
            ft.getCumSum(self.img3D, axis='C')


if __name__ == '__main__':
    unittest.main()
