import unittest
import numpy as np
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

    def test_createImg_dtype(self):
        dtypes = [("uint8", sitk.sitkUInt8),
                  ("uint16", sitk.sitkUInt16),
                  ("uint32", sitk.sitkUInt32),
                  ("uint64", sitk.sitkUInt64),
                  ("int8", sitk.sitkInt8),
                  ("int16", sitk.sitkInt16),
                  ("int32", sitk.sitkInt32),
                  ("int64", sitk.sitkInt64),
                  ("float32", sitk.sitkFloat32),
                  ("float64", sitk.sitkFloat64),
                  ("f4", sitk.sitkFloat32),
                  (np.int16, sitk.sitkInt16),
                  (np.dtype("uint32"), sitk.sitkUInt32),
                  (np.dtype(">f4"), sitk.sitkFloat32),
                  (np.dtype("<f8"), sitk.sitkFloat64),
                  (float, sitk.sitkFloat64),
                  (int, sitk.sitkInt64),
                  ]
        for dtype, pixelID in dtypes:
            with self.subTest(dtype=dtype):
                img = ft.createImg(size=[5, 6, 7], dtype=dtype)
                self.assertEqual(img.GetPixelID(), pixelID)
                self.assertEqual(img.GetSize(), (5, 6, 7))
                self.assertEqual(sitk.GetArrayViewFromImage(img).max(), 0)

    def test_createImg_dtype_vector(self):
        dtypes = {"int16": sitk.sitkVectorInt16,
                  "uint8": sitk.sitkVectorUInt8,
                  "float64": sitk.sitkVectorFloat64,
                  }
        for dtype, pixelID in dtypes.items():
            with self.subTest(dtype=dtype):
                img = ft.createImg(size=[5, 6, 7], components=3, dtype=dtype)
                self.assertEqual(img.GetPixelID(), pixelID)
                self.assertEqual(img.GetNumberOfComponentsPerPixel(), 3)

    def test_createImg_dtype_bool(self):
        img = ft.createImg(dtype=bool)
        self.assertEqual(img.GetPixelID(), sitk.sitkUInt8)

    def test_createImg_dtype_invalid(self):
        for dtype in ["float16", "complex64", "U3", "notatype", None]:
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    ft.createImg(dtype=dtype)

    def test_createImg_fillRandom_dtype(self):
        img = ft.createImg(size=[20, 20, 20], fillRandom=True, dtype="int16")
        self.assertEqual(img.GetPixelID(), sitk.sitkInt16)
        self.assertAlmostEqual(np.mean(sitk.GetArrayViewFromImage(img)), 10, delta=0.5)


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
        mask = ft.createCylinderMask(img, startPoint=[5, 5, 5], endPoint=[15, 15, 15], radius=5)
        self.assertEqual(mask.GetSize(), img.GetSize())
        self.assertEqual(mask.GetPixelID(), sitk.sitkUInt8)

    def test_createCylinderMask_invalid_points(self):
        img = ft.createImg(size=[20, 20, 20])
        with self.assertRaises(TypeError):
            ft.createCylinderMask(img, startPoint=[5, 5], endPoint=[15, 15, 15], radius=5)


if __name__ == '__main__':
    unittest.main()
