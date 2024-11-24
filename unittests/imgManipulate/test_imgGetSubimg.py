import unittest
import fredtools as ft


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
        self.assertAlmostEqual(ft.arr(imgRef).tolist(), ft.arr(imgEval).tolist(), places=7)

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


if __name__ == '__main__':
    unittest.main()
