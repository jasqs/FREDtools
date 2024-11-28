import unittest
import SimpleITK as sitk
import numpy as np

import fredtools as ft
print(ft.__version__)


class test_getExtent(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0])
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 2.0, 2.0], origin=[-5.0, -5.0, -5.0])
        self.img2D_custom = ft.createImg([10, 10], spacing=[2.0, 2.0], origin=[-5.0, -5.0])

    def test_getExtent_identity_direction_3D(self):
        expected_extent = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
        extent = ft.getExtent(self.img3D, displayInfo=True)
        self.assertEqual(extent, expected_extent)

    def test_getExtent_non_identity_direction_3D(self):
        self.img3D = sitk.Flip(self.img3D, flipAxes=[True, True, False])
        expected_extent = ((10.0, 0.0), (10.0, 0.0), (0.0, 10.0))
        extent = ft.getExtent(self.img3D, displayInfo=True)
        self.assertEqual(extent, expected_extent)

    def test_getExtent_identity_direction_2D(self):
        expected_extent = ((0.0, 10.0), (0.0, 10.0))
        extent = ft.getExtent(self.img2D, displayInfo=True)
        self.assertEqual(extent, expected_extent)

    def test_getExtent_non_identity_direction_2D(self):
        self.img2D = sitk.Flip(self.img2D, flipAxes=[True, True])
        expected_extent = ((10.0, 0.0), (10.0, 0.0))
        extent = ft.getExtent(self.img2D, displayInfo=True)
        self.assertEqual(extent, expected_extent)

    def test_getExtent_custom_spacing_origin_3D(self):
        expected_extent = ((-6.0, 14.0), (-6.0, 14.0), (-6.0, 14.0))
        extent = ft.getExtent(self.img3D_custom, displayInfo=True)
        self.assertEqual(extent, expected_extent)

    def test_getExtent_custom_spacing_origin_2D(self):
        expected_extent = ((-6.0, 14.0), (-6.0, 14.0))
        extent = ft.getExtent(self.img2D_custom, displayInfo=True)
        self.assertEqual(extent, expected_extent)


class test_getSize(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 2.0, 2.0], origin=[-5.0, -5.0, -5.0])
        self.img2D_custom = ft.createImg([10, 10], spacing=[2.0, 2.0], origin=[-5.0, -5.0])

    def test_getSize_identity_3D(self):
        expected_size = (10.0, 10.0, 10.0)
        size = ft.getSize(self.img3D, displayInfo=True)
        self.assertEqual(size, expected_size)

    def test_getSize_non_identity_3D(self):
        self.img3D = sitk.Flip(self.img3D, flipAxes=[True, True, False])
        expected_size = (10.0, 10.0, 10.0)
        size = ft.getSize(self.img3D, displayInfo=True)
        self.assertEqual(size, expected_size)

    def test_getSize_identity_2D(self):
        expected_size = (10.0, 10.0)
        size = ft.getSize(self.img2D, displayInfo=True)
        self.assertEqual(size, expected_size)

    def test_getSize_non_identity_2D(self):
        self.img2D = sitk.Flip(self.img2D, flipAxes=[True, True])
        expected_size = (10.0, 10.0)
        size = ft.getSize(self.img2D, displayInfo=True)
        self.assertEqual(size, expected_size)

    def test_getSize_custom_spacing_origin_3D(self):
        expected_size = (20.0, 20.0, 20.0)
        size = ft.getSize(self.img3D_custom, displayInfo=True)
        self.assertEqual(size, expected_size)

    def test_getSize_custom_spacing_origin_2D(self):
        expected_size = (20.0, 20.0)
        size = ft.getSize(self.img2D_custom, displayInfo=True)
        self.assertEqual(size, expected_size)


class test_getImageCenter(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[4.0, 2.0, 1.5], centred=True)
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.img2D_custom = ft.createImg([10, 10], spacing=[2.0, 2.0], centred=True)

    def test_getImageCenter_3D(self):
        expected_center = (4.5, 4.5, 4.5)
        center = ft.getImageCenter(self.img3D, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getImageCenter_custom_spacing_origin_3D(self):
        expected_center = (0.0, 0.0, 0.0)
        center = ft.getImageCenter(self.img3D_custom, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getImageCenter_non_identity_3D(self):
        self.img3D = sitk.Flip(self.img3D, flipAxes=[True, True, False])
        expected_center = (4.5, 4.5, 4.5)
        center = ft.getImageCenter(self.img3D, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getImageCenter_2D(self):
        expected_center = (4.5, 4.5)
        center = ft.getImageCenter(self.img2D, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getImageCenter_custom_spacing_origin_2D(self):
        expected_center = (0.0, 0.0)
        center = ft.getImageCenter(self.img2D_custom, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getImageCenter_non_identity_2D(self):
        self.img2D = sitk.Flip(self.img2D, flipAxes=[True, True])
        expected_center = (4.5, 4.5)
        center = ft.getImageCenter(self.img2D, displayInfo=True)
        self.assertEqual(center, expected_center)


class test_getMassCenter(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0])
        self.img3D[5, 5, 5] = 1.0
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 3.0, 1.5], centred=True)
        self.img3D_custom[5, 5, 5] = 1.0
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0])
        self.img2D[5, 5] = 1.0
        self.img2D_custom = ft.createImg([10, 10], spacing=[2.0, 2.0], centred=True)
        self.img2D_custom[5, 5] = 1.0
        self.img3D_vector = ft.createImg([10, 10, 10], components=7, spacing=[1.0, 1.0, 1.0])

    def test_getMassCenter_3D(self):
        expected_center = (5.5, 5.5, 5.5)
        center = ft.getMassCenter(self.img3D, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getMassCenter_custom_spacing_origin_3D(self):
        expected_center = (1.0, 1.5, 0.75)
        center = ft.getMassCenter(self.img3D_custom, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getMassCenter_non_identity_3D(self):
        self.img3D = sitk.Flip(self.img3D, flipAxes=[True, True, False])
        expected_center = (5.5, 5.5, 5.5)
        center = ft.getMassCenter(self.img3D, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getMassCenter_2D(self):
        expected_center = (5.5, 5.5)
        center = ft.getMassCenter(self.img2D, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getMassCenter_custom_spacing_origin_2D(self):
        expected_center = (1.0, 1.0)
        center = ft.getMassCenter(self.img2D_custom, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getMassCenter_non_identity_2D(self):
        self.img2D = sitk.Flip(self.img2D, flipAxes=[True, True])
        expected_center = (5.5, 5.5)
        center = ft.getMassCenter(self.img2D, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getMassCenter_vector_image(self):
        expected_center = (np.nan, np.nan, np.nan)
        center = ft.getMassCenter(self.img3D_vector, displayInfo=True)
        self.assertEqual(center, expected_center)


class test_getMaxPosition(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D[5, 5, 5] = 1.0
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 2.0, 2.0], centred=True)
        self.img3D_custom[5, 5, 5] = 1.0

    def test_getMaxPosition(self):
        expected_position = (5.0, 5.0, 5.0)
        position = ft.getMaxPosition(self.img3D, displayInfo=True)
        self.assertEqual(position, expected_position)

    def test_getMaxPosition_custom_spacing_origin(self):
        expected_position = (1.0, 1.0, 1.0)
        position = ft.getMaxPosition(self.img3D_custom, displayInfo=True)
        self.assertEqual(position, expected_position)


class test_getMinPosition(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D[5, 5, 5] = -1.0
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 2.0, 2.0], centred=True)
        self.img3D_custom[5, 5, 5] = -1.0

    def test_getMinPosition(self):
        expected_position = (5.0, 5.0, 5.0)
        position = ft.getMinPosition(self.img3D, displayInfo=True)
        self.assertEqual(position, expected_position)

    def test_getMinPosition_custom_spacing_origin(self):
        expected_position = (1.0, 1.0, 1.0)
        position = ft.getMinPosition(self.img3D_custom, displayInfo=True)
        self.assertEqual(position, expected_position)


class test_getVoxelCentres(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])

    def test_getVoxelCentres_3D(self):
        expected_centres = (tuple(np.linspace(0.0, 9.0, 10)),
                            tuple(np.linspace(0.0, 9.0, 10)),
                            tuple(np.linspace(0.0, 9.0, 10)))
        centres = ft.getVoxelCentres(self.img3D, displayInfo=True)
        self.assertEqual(centres, expected_centres)

    def test_getVoxelCentres_2D(self):
        expected_centres = (tuple(np.linspace(0.0, 9.0, 10)),
                            tuple(np.linspace(0.0, 9.0, 10)))
        centres = ft.getVoxelCentres(self.img2D, displayInfo=True)
        self.assertEqual(centres, expected_centres)


class test_getVoxelEdges(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_getVoxelEdges(self):
        expected_edges = (tuple(np.linspace(-0.5, 9.5, 11)),
                          tuple(np.linspace(-0.5, 9.5, 11)),
                          tuple(np.linspace(-0.5, 9.5, 11)))
        edges = ft.getVoxelEdges(self.img3D, displayInfo=True)
        self.assertEqual(edges, expected_edges)


class test_getVoxelPhysicalPoints(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_getVoxelPhysicalPoints(self):
        points = np.array(ft.getVoxelPhysicalPoints(self.img3D, displayInfo=True))
        self.assertEqual(points.shape, (1000, 3))


class test_getExtMpl(unittest.TestCase):
    def setUp(self):
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 2.0], centred=True)

    def test_getExtMpl(self):
        expected_extent = (-5.0, 5.0, 10.0, -10.0)
        extent = ft.getExtMpl(self.img2D)
        self.assertEqual(extent, expected_extent)


class test_pos(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_pos(self):
        expected_pos = (tuple(np.linspace(0.0, 9.0, 10)),
                        tuple(np.linspace(0.0, 9.0, 10)),
                        tuple(np.linspace(0.0, 9.0, 10)))
        pos_values = ft.pos(self.img3D)
        self.assertEqual(pos_values, expected_pos)


class test_arr(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_arr(self):
        array = ft.arr(self.img3D)
        self.assertEqual(array.shape, (10, 10, 10))


class test_vec(unittest.TestCase):
    def setUp(self):
        self.img3Dprofile = ft.createImg([1, 10, 1], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_vec(self):
        vector = ft.vec(self.img3Dprofile)
        self.assertEqual(vector.shape, (10,))


class test_isPointInside(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_isPointInside(self):
        point = (5.0, 5.0, 5.0)
        self.assertTrue(ft.isPointInside(self.img3D, point, displayInfo=True))


class test_getStatistics(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_getStatistics(self):
        stats = ft.getStatistics(self.img3D, displayInfo=True)
        self.assertEqual(stats.GetMean(), 0.0)


class test_compareImgFoR(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_compareImgFoR(self):
        img2 = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.assertTrue(ft.compareImgFoR(self.img3D, img2, displayInfo=True))

    def test_compareImgFoR_fail(self):
        img2 = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[1.0, 0.0, 0.0])
        self.assertFalse(ft.compareImgFoR(self.img3D, img2, displayInfo=True))


class test_compareImg(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_compareImg(self):
        img2 = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.assertTrue(ft.compareImg(self.img3D, img2, displayInfo=True))

    def test_compareImg_fail(self):
        img2 = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        img2[5, 5, 5] = 1.0
        self.assertFalse(ft.compareImg(self.img3D, img2, displayInfo=True))

    def test_compareImg_differentFoR(self):
        img2 = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[1.0, 0.0, 0.0])
        self.assertTrue(ft.compareImg(self.img3D, img2, displayInfo=True))


if __name__ == '__main__':
    unittest.main()
