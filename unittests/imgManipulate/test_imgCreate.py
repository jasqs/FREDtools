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


class test_CreateImgFromFunction(unittest.TestCase):

    def setUp(self):
        # dyadic geometry, so that the voxel centres are represented exactly in float64
        self.img = ft.createImg(size=[5, 6, 7], spacing=[0.5, 0.25, 2], origin=[-1.25, 0.375, 3], dtype="float64")

    @staticmethod
    def affine(x, y, z):
        return 1.2 + 0.04 * x - 0.3 * y + 0.125 * z

    def test_createImgFromFunction_affine(self):
        imgOut = ft.createImgFromFunction(self.img, self.affine)
        zz, yy, xx = np.meshgrid(*ft.getVoxelCentres(self.img)[::-1], indexing="ij")
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(imgOut), self.affine(xx, yy, zz)))

    def test_createImgFromFunction_rotated(self):
        from scipy.spatial.transform import Rotation
        self.img.SetDirection(Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix().flatten())
        voxelPositions = [ft.createImgFromFunction(self.img, lambda x, y, z, axis=axis: [x, y, z][axis]) for axis in range(3)]
        voxelPositions = np.stack([sitk.GetArrayFromImage(imgAxis) for imgAxis in voxelPositions], axis=-1)
        for index in np.ndindex(*self.img.GetSize()):
            np.testing.assert_allclose(voxelPositions[index[::-1]], self.img.TransformIndexToPhysicalPoint(index), rtol=0, atol=1e-12)

    def test_createImgFromFunction_2D(self):
        img = ft.createImg(size=[5, 6], spacing=[0.5, 0.25], origin=[-1.25, 0.375], dtype="float64")
        imgOut = ft.createImgFromFunction(img, lambda x, y: x - 2 * y)
        yy, xx = np.meshgrid(*ft.getVoxelCentres(img)[::-1], indexing="ij")
        self.assertEqual(imgOut.GetSize(), (5, 6))
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(imgOut), xx - 2 * yy))

    def test_createImgFromFunction_constant(self):
        imgOut = ft.createImgFromFunction(self.img, lambda x, y, z: 2.5)
        self.assertTrue(np.all(sitk.GetArrayViewFromImage(imgOut) == 2.5))

    def test_createImgFromFunction_geometry_type(self):
        self.img.SetDirection([0, 1, 0, -1, 0, 0, 0, 0, 1])
        imgOut64 = ft.createImgFromFunction(self.img, self.affine)
        imgOut32 = ft.createImgFromFunction(sitk.Cast(self.img, sitk.sitkFloat32), self.affine)
        self.assertTrue(ft.compareImgFoR(imgOut64, self.img))
        self.assertEqual(imgOut64.GetDirection(), self.img.GetDirection())
        self.assertEqual(imgOut64.GetPixelID(), sitk.sitkFloat64)
        self.assertEqual(imgOut32.GetPixelID(), sitk.sitkFloat32)
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(imgOut32), sitk.GetArrayFromImage(imgOut64).astype(np.float32)))

    def test_createImgFromFunction_integer(self):
        img = ft.createImg(size=[2, 1, 1], dtype="int16")
        imgOut = ft.createImgFromFunction(img, lambda x, y, z: np.where(x < 1, -1.7, 2.9))
        self.assertEqual(imgOut.GetPixelID(), sitk.sitkInt16)
        self.assertEqual(sitk.GetArrayFromImage(imgOut).flatten().tolist(), [-1, 2])

    def test_createImgFromFunction_mask(self):
        img = ft.createImg(size=[20, 20, 20], centred=True, dtype="uint8")
        imgMask = ft.createImgFromFunction(img, lambda x, y, z: x**2 + y**2 + z**2 <= 5**2)
        self.assertTrue(ft._imgTypeChecker.isSITK_maskBinary(imgMask))
        self.assertGreater(sitk.GetArrayViewFromImage(imgMask).sum(), 0)

    def test_createImgFromFunction_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.createImgFromFunction(np.zeros([5, 5, 5]), self.affine)
        with self.assertRaises(TypeError):
            ft.createImgFromFunction(ft.createImg(components=3), self.affine)
        with self.assertRaises(ValueError):
            ft.createImgFromFunction(sitk.Image([5, 5, 5, 5], sitk.sitkFloat32), lambda x, y, z, t: x)

    def test_createImgFromFunction_invalid_function(self):
        with self.assertRaises(TypeError):
            ft.createImgFromFunction(self.img, 5)
        with self.assertRaises(ValueError):
            ft.createImgFromFunction(self.img, lambda x, y, z: np.zeros(3))

    def test_createImgFromFunction_invalid_integer_values(self):
        img = ft.createImg(size=[5, 5, 5], dtype="uint8")
        for value in [256, -1, np.nan, np.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ft.createImgFromFunction(img, lambda x, y, z: value)
        # values truncated into the type range are allowed
        imgOut = ft.createImgFromFunction(img, lambda x, y, z: np.where(x < 2, -0.9, 255.9))
        self.assertEqual(sitk.GetArrayViewFromImage(imgOut).min(), 0)
        self.assertEqual(sitk.GetArrayViewFromImage(imgOut).max(), 255)


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
