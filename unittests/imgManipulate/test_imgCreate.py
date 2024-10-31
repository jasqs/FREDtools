import unittest
import SimpleITK as sitk
import fredtools as ft


class test_CreateImg(unittest.TestCase):

    def test_createImg_default(self):
        img = ft.createImg()
        self.assertEqual(img.GetSize(), (10, 20, 30))
        self.assertEqual(img.GetSpacing(), (1.0, 1.0, 1.0))
        self.assertEqual(img.GetOrigin(), (0.5, 0.5, 0.5))
        self.assertEqual(img.GetPixelID(), sitk.sitkFloat32)

    def test_createImg_custom(self):
        img = ft.createImg(size=[5, 5], components=7, spacing=[0.5, 0.5], centred=True)
        self.assertEqual(img.GetSize(), (5, 5))
        self.assertEqual(img.GetSpacing(), (0.5, 0.5))
        self.assertEqual(img.GetOrigin(), (-1.0, -1.0))
        self.assertEqual(img.GetNumberOfComponentsPerPixel(), 7)
        self.assertEqual(img.GetPixelID(), sitk.sitkVectorFloat32)

    def test_createImg_invalid_size(self):
        with self.assertRaises(ValueError):
            ft.createImg(size=[5])

    def test_createImg_invalid_components(self):
        with self.assertRaises(ValueError):
            ft.createImg(components=-1)


class test_CreateEllipseMask(unittest.TestCase):

    def test_createEllipseMask(self):
        img = ft.createImg(size=[20, 20, 20])
        mask = ft.createEllipseMask(img, point=[10, 10, 10], radii=[5, 5, 5])
        self.assertEqual(mask.GetSize(), img.GetSize())
        self.assertEqual(mask.GetPixelID(), sitk.sitkUInt8)

    def test_createEllipseMask_invalid_radii(self):
        img = ft.createImg(size=[20, 20, 20])
        with self.assertRaises(ValueError):
            ft.createEllipseMask(img, point=[10, 10, 10], radii=[5, 5])


class test_CreateConeMask(unittest.TestCase):

    def test_createConeMask(self):
        img = ft.createImg(size=[20, 20, 20])
        mask = ft.createConeMask(img, startPoint=[5, 5, 5], endPoint=[15, 15, 15], startRadius=5, endRadius=2)
        self.assertEqual(mask.GetSize(), img.GetSize())
        self.assertEqual(mask.GetPixelID(), sitk.sitkUInt8)

    def test_createConeMask_invalid_points(self):
        img = ft.createImg(size=[20, 20, 20])
        with self.assertRaises(TypeError):
            ft.createConeMask(img, startPoint=[5, 5], endPoint=[15, 15, 15], startRadius=5, endRadius=2)


class test_CreateCylinderMask(unittest.TestCase):

    def test_createCylinderMask(self):
        img = ft.createImg(size=[20, 20, 20])
        mask = ft.createCylinderMask(img, startPoint=[5, 5, 5], endPoint=[15, 15, 15], radious=5)
        self.assertEqual(mask.GetSize(), img.GetSize())
        self.assertEqual(mask.GetPixelID(), sitk.sitkUInt8)

    def test_createCylinderMask_invalid_points(self):
        img = ft.createImg(size=[20, 20, 20])
        with self.assertRaises(TypeError):
            ft.createCylinderMask(img, startPoint=[5, 5], endPoint=[15, 15, 15], radious=5)


if __name__ == '__main__':
    unittest.main()
