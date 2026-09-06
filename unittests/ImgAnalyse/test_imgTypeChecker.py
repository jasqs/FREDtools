import unittest
import numpy as np
import SimpleITK as sitk
import itk
import fredtools as ft


class test_isITK(unittest.TestCase):

    def setUp(self):
        self.imgITK2D = ft.SITK2ITK(ft.createImg([10, 10], spacing=[1, 1]))
        self.imgITK3D = ft.SITK2ITK(ft.createImg([10, 10, 10]))
        self.imgITK4D = itk.image_from_array(np.zeros((2, 3, 4, 5), dtype=np.float32))
        self.imgSITK3D = ft.createImg([10, 10, 10])
        self.invalidInputs = [np.zeros((10, 10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isITK_2D(self):
        self.assertTrue(ft._imgTypeChecker.isITK(self.imgITK2D))

    def test_isITK_3D(self):
        self.assertTrue(ft._imgTypeChecker.isITK(self.imgITK3D))

    def test_isITK_4D(self):
        self.assertTrue(ft._imgTypeChecker.isITK(self.imgITK4D))

    def test_isITK_SITK(self):
        self.assertFalse(ft._imgTypeChecker.isITK(self.imgSITK3D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK(self.imgSITK3D, raiseError=True)

    def test_isITK_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isITK(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isITK(invalidInput, raiseError=True)


class test_isITK2D(unittest.TestCase):

    def setUp(self):
        self.imgITK2D = ft.SITK2ITK(ft.createImg([10, 10], spacing=[1, 1]))
        self.imgITK3D = ft.SITK2ITK(ft.createImg([10, 10, 10]))
        self.imgITK4D = itk.image_from_array(np.zeros((2, 3, 4, 5), dtype=np.float32))
        self.imgSITK2D = ft.createImg([10, 10], spacing=[1, 1])
        self.invalidInputs = [np.zeros((10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isITK2D_2D(self):
        self.assertTrue(ft._imgTypeChecker.isITK2D(self.imgITK2D))

    def test_isITK2D_3D(self):
        self.assertFalse(ft._imgTypeChecker.isITK2D(self.imgITK3D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK2D(self.imgITK3D, raiseError=True)

    def test_isITK2D_4D(self):
        self.assertFalse(ft._imgTypeChecker.isITK2D(self.imgITK4D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK2D(self.imgITK4D, raiseError=True)

    def test_isITK2D_SITK(self):
        self.assertFalse(ft._imgTypeChecker.isITK2D(self.imgSITK2D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK2D(self.imgSITK2D, raiseError=True)

    def test_isITK2D_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isITK2D(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isITK2D(invalidInput, raiseError=True)


class test_isITK3D(unittest.TestCase):

    def setUp(self):
        self.imgITK2D = ft.SITK2ITK(ft.createImg([10, 10], spacing=[1, 1]))
        self.imgITK3D = ft.SITK2ITK(ft.createImg([10, 10, 10]))
        self.imgITK4D = itk.image_from_array(np.zeros((2, 3, 4, 5), dtype=np.float32))
        self.imgSITK3D = ft.createImg([10, 10, 10])
        self.invalidInputs = [np.zeros((10, 10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isITK3D_3D(self):
        self.assertTrue(ft._imgTypeChecker.isITK3D(self.imgITK3D))

    def test_isITK3D_2D(self):
        self.assertFalse(ft._imgTypeChecker.isITK3D(self.imgITK2D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK3D(self.imgITK2D, raiseError=True)

    def test_isITK3D_4D(self):
        self.assertFalse(ft._imgTypeChecker.isITK3D(self.imgITK4D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK3D(self.imgITK4D, raiseError=True)

    def test_isITK3D_SITK(self):
        self.assertFalse(ft._imgTypeChecker.isITK3D(self.imgSITK3D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK3D(self.imgSITK3D, raiseError=True)

    def test_isITK3D_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isITK3D(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isITK3D(invalidInput, raiseError=True)


class test_isITK4D(unittest.TestCase):

    def setUp(self):
        self.imgITK2D = ft.SITK2ITK(ft.createImg([10, 10], spacing=[1, 1]))
        self.imgITK3D = ft.SITK2ITK(ft.createImg([10, 10, 10]))
        self.imgITK4D = itk.image_from_array(np.zeros((2, 3, 4, 5), dtype=np.float32))
        self.imgSITK4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)
        self.invalidInputs = [np.zeros((2, 3, 4, 5)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isITK4D_4D(self):
        self.assertTrue(ft._imgTypeChecker.isITK4D(self.imgITK4D))

    def test_isITK4D_2D(self):
        self.assertFalse(ft._imgTypeChecker.isITK4D(self.imgITK2D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK4D(self.imgITK2D, raiseError=True)

    def test_isITK4D_3D(self):
        self.assertFalse(ft._imgTypeChecker.isITK4D(self.imgITK3D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK4D(self.imgITK3D, raiseError=True)

    def test_isITK4D_SITK(self):
        self.assertFalse(ft._imgTypeChecker.isITK4D(self.imgSITK4D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isITK4D(self.imgSITK4D, raiseError=True)

    def test_isITK4D_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isITK4D(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isITK4D(invalidInput, raiseError=True)


class test_isSITK(unittest.TestCase):

    def setUp(self):
        self.img2D = ft.createImg([10, 10], spacing=[1, 1])
        self.img3D = ft.createImg([10, 10, 10])
        self.img4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)
        self.img3DVector = ft.createImg([10, 10, 10], components=3)
        self.imgITK3D = ft.SITK2ITK(self.img3D)
        self.invalidInputs = [np.zeros((10, 10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_2D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK(self.img2D))

    def test_isSITK_3D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK(self.img3D))

    def test_isSITK_4D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK(self.img4D))

    def test_isSITK_vector(self):
        self.assertTrue(ft._imgTypeChecker.isSITK(self.img3DVector))

    def test_isSITK_ITK(self):
        self.assertFalse(ft._imgTypeChecker.isSITK(self.imgITK3D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK(self.imgITK3D, raiseError=True)

    def test_isSITK_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK(invalidInput, raiseError=True)


class test_isSITK2D(unittest.TestCase):

    def setUp(self):
        self.img2D = ft.createImg([10, 10], spacing=[1, 1])
        self.img3D = ft.createImg([10, 10, 10])
        self.img4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)
        self.imgITK2D = ft.SITK2ITK(self.img2D)
        self.invalidInputs = [np.zeros((10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK2D_2D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK2D(self.img2D))

    def test_isSITK2D_3D(self):
        self.assertFalse(ft._imgTypeChecker.isSITK2D(self.img3D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK2D(self.img3D, raiseError=True)

    def test_isSITK2D_4D(self):
        self.assertFalse(ft._imgTypeChecker.isSITK2D(self.img4D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK2D(self.img4D, raiseError=True)

    def test_isSITK2D_ITK(self):
        self.assertFalse(ft._imgTypeChecker.isSITK2D(self.imgITK2D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK2D(self.imgITK2D, raiseError=True)

    def test_isSITK2D_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK2D(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK2D(invalidInput, raiseError=True)


class test_isSITK3D(unittest.TestCase):

    def setUp(self):
        self.img2D = ft.createImg([10, 10], spacing=[1, 1])
        self.img3D = ft.createImg([10, 10, 10])
        self.img4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)
        self.imgITK3D = ft.SITK2ITK(self.img3D)
        self.invalidInputs = [np.zeros((10, 10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK3D_3D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK3D(self.img3D))

    def test_isSITK3D_2D(self):
        self.assertFalse(ft._imgTypeChecker.isSITK3D(self.img2D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK3D(self.img2D, raiseError=True)

    def test_isSITK3D_4D(self):
        self.assertFalse(ft._imgTypeChecker.isSITK3D(self.img4D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK3D(self.img4D, raiseError=True)

    def test_isSITK3D_ITK(self):
        self.assertFalse(ft._imgTypeChecker.isSITK3D(self.imgITK3D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK3D(self.imgITK3D, raiseError=True)

    def test_isSITK3D_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK3D(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK3D(invalidInput, raiseError=True)


class test_isSITK4D(unittest.TestCase):

    def setUp(self):
        self.img2D = ft.createImg([10, 10], spacing=[1, 1])
        self.img3D = ft.createImg([10, 10, 10])
        self.img4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)
        self.imgITK4D = itk.image_from_array(np.zeros((2, 3, 4, 5), dtype=np.float32))
        self.invalidInputs = [np.zeros((2, 3, 4, 5)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK4D_4D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK4D(self.img4D))

    def test_isSITK4D_2D(self):
        self.assertFalse(ft._imgTypeChecker.isSITK4D(self.img2D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK4D(self.img2D, raiseError=True)

    def test_isSITK4D_3D(self):
        self.assertFalse(ft._imgTypeChecker.isSITK4D(self.img3D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK4D(self.img3D, raiseError=True)

    def test_isSITK4D_ITK(self):
        self.assertFalse(ft._imgTypeChecker.isSITK4D(self.imgITK4D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK4D(self.imgITK4D, raiseError=True)

    def test_isSITK4D_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK4D(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK4D(invalidInput, raiseError=True)


class test_isSITK_point(unittest.TestCase):

    def setUp(self):
        self.imgPoint3D = ft.createImg([1, 1, 1])
        self.imgPoint2D = ft.createImg([1, 1], spacing=[1, 1])
        self.imgProfile = ft.createImg([10, 1, 1])
        self.imgSlice = ft.createImg([10, 10, 1])
        self.imgVolume = ft.createImg([10, 10, 10])
        self.invalidInputs = [np.zeros((1, 1, 1)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_point_3D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_point(self.imgPoint3D))

    def test_isSITK_point_2D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_point(self.imgPoint2D))

    def test_isSITK_point_profile(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_point(self.imgProfile))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_point(self.imgProfile, raiseError=True)

    def test_isSITK_point_slice(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_point(self.imgSlice))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_point(self.imgSlice, raiseError=True)

    def test_isSITK_point_volume(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_point(self.imgVolume))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_point(self.imgVolume, raiseError=True)

    def test_isSITK_point_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_point(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_point(invalidInput, raiseError=True)


class test_isSITK_profile(unittest.TestCase):

    def setUp(self):
        self.imgProfile3D = ft.createImg([10, 1, 1])
        self.imgProfile2D = ft.createImg([10, 1], spacing=[1, 1])
        self.imgPoint = ft.createImg([1, 1, 1])
        self.imgSlice = ft.createImg([10, 10, 1])
        self.imgVolume = ft.createImg([10, 10, 10])
        self.invalidInputs = [np.zeros((10, 1, 1)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_profile_3D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_profile(self.imgProfile3D))

    def test_isSITK_profile_2D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_profile(self.imgProfile2D))

    def test_isSITK_profile_point(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_profile(self.imgPoint))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_profile(self.imgPoint, raiseError=True)

    def test_isSITK_profile_slice(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_profile(self.imgSlice))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_profile(self.imgSlice, raiseError=True)

    def test_isSITK_profile_volume(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_profile(self.imgVolume))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_profile(self.imgVolume, raiseError=True)

    def test_isSITK_profile_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_profile(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_profile(invalidInput, raiseError=True)


class test_isSITK_slice(unittest.TestCase):

    def setUp(self):
        self.imgSlice3D = ft.createImg([10, 10, 1])
        self.imgSlice2D = ft.createImg([10, 10], spacing=[1, 1])
        self.imgPoint = ft.createImg([1, 1, 1])
        self.imgProfile = ft.createImg([10, 1, 1])
        self.imgVolume = ft.createImg([10, 10, 10])
        self.invalidInputs = [np.zeros((10, 10, 1)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_slice_3D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_slice(self.imgSlice3D))

    def test_isSITK_slice_2D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_slice(self.imgSlice2D))

    def test_isSITK_slice_point(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_slice(self.imgPoint))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_slice(self.imgPoint, raiseError=True)

    def test_isSITK_slice_profile(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_slice(self.imgProfile))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_slice(self.imgProfile, raiseError=True)

    def test_isSITK_slice_volume(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_slice(self.imgVolume))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_slice(self.imgVolume, raiseError=True)

    def test_isSITK_slice_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_slice(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_slice(invalidInput, raiseError=True)


class test_isSITK_volume(unittest.TestCase):

    def setUp(self):
        self.imgVolume3D = ft.createImg([10, 10, 10])
        self.imgVolume4D = sitk.Image([1, 5, 6, 7], sitk.sitkFloat32)
        self.imgSlice = ft.createImg([10, 10, 1])
        self.imgTimeVolume = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)
        self.invalidInputs = [np.zeros((10, 10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_volume_3D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_volume(self.imgVolume3D))

    def test_isSITK_volume_4D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_volume(self.imgVolume4D))

    def test_isSITK_volume_slice(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_volume(self.imgSlice))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_volume(self.imgSlice, raiseError=True)

    def test_isSITK_volume_timevolume(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_volume(self.imgTimeVolume))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_volume(self.imgTimeVolume, raiseError=True)

    def test_isSITK_volume_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_volume(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_volume(invalidInput, raiseError=True)


class test_isSITK_timevolume(unittest.TestCase):

    def setUp(self):
        self.imgTimeVolume = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)
        self.imgVolume4D = sitk.Image([1, 5, 6, 7], sitk.sitkFloat32)
        self.imgVolume3D = ft.createImg([10, 10, 10])
        self.invalidInputs = [np.zeros((4, 5, 6, 7)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_timevolume_4D(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_timevolume(self.imgTimeVolume))

    def test_isSITK_timevolume_volume4D(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_timevolume(self.imgVolume4D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_timevolume(self.imgVolume4D, raiseError=True)

    def test_isSITK_timevolume_volume3D(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_timevolume(self.imgVolume3D))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_timevolume(self.imgVolume3D, raiseError=True)

    def test_isSITK_timevolume_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_timevolume(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_timevolume(invalidInput, raiseError=True)


class test_isSITK_vector(unittest.TestCase):

    def setUp(self):
        self.imgVector = ft.createImg([10, 10, 10], components=3)
        self.imgScalar = ft.createImg([10, 10, 10])
        self.invalidInputs = [np.zeros((10, 10, 10, 3)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_vector_vector(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_vector(self.imgVector))

    def test_isSITK_vector_scalar(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_vector(self.imgScalar))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_vector(self.imgScalar, raiseError=True)

    def test_isSITK_vector_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_vector(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_vector(invalidInput, raiseError=True)


class test_isSITK_transform(unittest.TestCase):

    def setUp(self):
        self.transforms = [sitk.Transform(), sitk.Euler3DTransform(), sitk.AffineTransform(3), sitk.TranslationTransform(3)]
        self.img3D = ft.createImg([10, 10, 10])
        self.invalidInputs = [np.eye(4), None, "transform", [1, 2, 3]]

    def test_isSITK_transform_transform(self):
        for transform in self.transforms:
            with self.subTest(transform=type(transform)):
                self.assertTrue(ft._imgTypeChecker.isSITK_transform(transform))

    def test_isSITK_transform_image(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_transform(self.img3D))  # type: ignore
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_transform(self.img3D, raiseError=True)  # type: ignore

    def test_isSITK_transform_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_transform(invalidInput))  # type: ignore
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_transform(invalidInput, raiseError=True)  # type: ignore


class test_isSITK_maskBinary(unittest.TestCase):

    def setUp(self):
        self.maskBinary = ft.createEllipseMask(ft.createImg([10, 10, 10]), point=[5, 5, 5], radii=[3, 3, 3])
        self.maskBinaryZeros = sitk.Image([5, 5, 5], sitk.sitkUInt8)
        self.maskFloating = sitk.Cast(self.maskBinary, sitk.sitkFloat32) * 0.5
        self.maskInt16 = sitk.Cast(self.maskBinary, sitk.sitkInt16)
        self.maskBinaryInvalidValue = sitk.Image(self.maskBinary)
        self.maskBinaryInvalidValue[0, 0, 0] = 2
        self.invalidInputs = [np.zeros((10, 10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_maskBinary_binary(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_maskBinary(self.maskBinary))

    def test_isSITK_maskBinary_zeros(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_maskBinary(self.maskBinaryZeros))

    def test_isSITK_maskBinary_floating(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_maskBinary(self.maskFloating))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_maskBinary(self.maskFloating, raiseError=True)

    def test_isSITK_maskBinary_int16(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_maskBinary(self.maskInt16))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_maskBinary(self.maskInt16, raiseError=True)

    def test_isSITK_maskBinary_invalid_value(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_maskBinary(self.maskBinaryInvalidValue))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_maskBinary(self.maskBinaryInvalidValue, raiseError=True)

    def test_isSITK_maskBinary_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_maskBinary(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_maskBinary(invalidInput, raiseError=True)


class test_isSITK_maskFloating(unittest.TestCase):

    def setUp(self):
        self.maskBinary = ft.createEllipseMask(ft.createImg([10, 10, 10]), point=[5, 5, 5], radii=[3, 3, 3])
        self.maskFloating32 = sitk.Cast(self.maskBinary, sitk.sitkFloat32) * 0.5
        self.maskFloating64 = sitk.Cast(self.maskBinary, sitk.sitkFloat64) * 0.5
        self.maskFloatingZeros = sitk.Image([5, 5, 5], sitk.sitkFloat32)
        self.maskFloatingAboveOne = sitk.Image(self.maskFloating32)
        self.maskFloatingAboveOne[0, 0, 0] = 1.5
        self.maskFloatingNegative = sitk.Image(self.maskFloating32)
        self.maskFloatingNegative[0, 0, 0] = -0.5
        self.invalidInputs = [np.zeros((10, 10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_maskFloating_float32(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_maskFloating(self.maskFloating32))

    def test_isSITK_maskFloating_float64(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_maskFloating(self.maskFloating64))

    def test_isSITK_maskFloating_zeros(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_maskFloating(self.maskFloatingZeros))

    def test_isSITK_maskFloating_binary(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_maskFloating(self.maskBinary))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_maskFloating(self.maskBinary, raiseError=True)

    def test_isSITK_maskFloating_above_one(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_maskFloating(self.maskFloatingAboveOne))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_maskFloating(self.maskFloatingAboveOne, raiseError=True)

    def test_isSITK_maskFloating_negative(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_maskFloating(self.maskFloatingNegative))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_maskFloating(self.maskFloatingNegative, raiseError=True)

    def test_isSITK_maskFloating_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_maskFloating(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_maskFloating(invalidInput, raiseError=True)


class test_isSITK_mask(unittest.TestCase):

    def setUp(self):
        self.maskBinary = ft.createEllipseMask(ft.createImg([10, 10, 10]), point=[5, 5, 5], radii=[3, 3, 3])
        self.maskFloating = sitk.Cast(self.maskBinary, sitk.sitkFloat32) * 0.5
        self.maskInt16 = sitk.Cast(self.maskBinary, sitk.sitkInt16)
        self.maskBinaryInvalidValue = sitk.Image(self.maskBinary)
        self.maskBinaryInvalidValue[0, 0, 0] = 2
        self.maskFloatingAboveOne = sitk.Image(self.maskFloating)
        self.maskFloatingAboveOne[0, 0, 0] = 1.5
        self.invalidInputs = [np.zeros((10, 10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_isSITK_mask_binary(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_mask(self.maskBinary))

    def test_isSITK_mask_floating(self):
        self.assertTrue(ft._imgTypeChecker.isSITK_mask(self.maskFloating))

    def test_isSITK_mask_int16(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_mask(self.maskInt16))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_mask(self.maskInt16, raiseError=True)

    def test_isSITK_mask_invalid_binary_value(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_mask(self.maskBinaryInvalidValue))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_mask(self.maskBinaryInvalidValue, raiseError=True)

    def test_isSITK_mask_invalid_floating_value(self):
        self.assertFalse(ft._imgTypeChecker.isSITK_mask(self.maskFloatingAboveOne))
        with self.assertRaises(TypeError):
            ft._imgTypeChecker.isSITK_mask(self.maskFloatingAboveOne, raiseError=True)

    def test_isSITK_mask_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                self.assertFalse(ft._imgTypeChecker.isSITK_mask(invalidInput))
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.isSITK_mask(invalidInput, raiseError=True)


class test_getMaskType(unittest.TestCase):

    def setUp(self):
        self.maskBinary = ft.createEllipseMask(ft.createImg([10, 10, 10]), point=[5, 5, 5], radii=[3, 3, 3])
        self.maskBinaryZeros = sitk.Image([5, 5, 5], sitk.sitkUInt8)
        self.maskFloating32 = sitk.Cast(self.maskBinary, sitk.sitkFloat32) * 0.5
        self.maskFloating64 = sitk.Cast(self.maskBinary, sitk.sitkFloat64) * 0.5
        self.maskFloatingZeros = sitk.Image([5, 5, 5], sitk.sitkFloat32)
        self.maskInt16 = sitk.Cast(self.maskBinary, sitk.sitkInt16)
        self.maskBinaryInvalidValue = sitk.Image(self.maskBinary)
        self.maskBinaryInvalidValue[0, 0, 0] = 2
        self.maskFloatingAboveOne = sitk.Image(self.maskFloating32)
        self.maskFloatingAboveOne[0, 0, 0] = 1.5
        self.invalidInputs = [np.zeros((10, 10, 10)), None, "image", [1, 2, 3], sitk.Euler3DTransform()]

    def test_getMaskType_binary(self):
        self.assertEqual(ft._imgTypeChecker.getMaskType(self.maskBinary), "binary")
        self.assertEqual(ft._imgTypeChecker.getMaskType(self.maskBinaryZeros), "binary")

    def test_getMaskType_floating(self):
        self.assertEqual(ft._imgTypeChecker.getMaskType(self.maskFloating32), "floating")
        self.assertEqual(ft._imgTypeChecker.getMaskType(self.maskFloating64), "floating")
        self.assertEqual(ft._imgTypeChecker.getMaskType(self.maskFloatingZeros), "floating")

    def test_getMaskType_invalid_mask(self):
        for invalidMask in [self.maskInt16, self.maskBinaryInvalidValue, self.maskFloatingAboveOne]:
            with self.subTest(pixelType=invalidMask.GetPixelIDTypeAsString()):
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.getMaskType(invalidMask)

    def test_getMaskType_invalid_input(self):
        for invalidInput in self.invalidInputs:
            with self.subTest(invalidInput=type(invalidInput)):
                with self.assertRaises(TypeError):
                    ft._imgTypeChecker.getMaskType(invalidInput)


if __name__ == '__main__':
    unittest.main()
