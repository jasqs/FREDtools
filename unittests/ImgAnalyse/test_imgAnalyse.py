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
        self.img4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)

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

    def test_getExtent_4D(self):
        expected_extent = ((-0.5, 3.5), (-0.5, 4.5), (-0.5, 5.5), (-0.5, 6.5))
        extent = ft.getExtent(self.img4D, displayInfo=True)
        self.assertEqual(extent, expected_extent)

    def test_getExtent_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getExtent(ft.arr(self.img3D))  # type: ignore


class test_getSize(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 2.0, 2.0], origin=[-5.0, -5.0, -5.0])
        self.img2D_custom = ft.createImg([10, 10], spacing=[2.0, 2.0], origin=[-5.0, -5.0])
        self.img4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)

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

    def test_getSize_4D(self):
        expected_size = (4.0, 5.0, 6.0, 7.0)
        size = ft.getSize(self.img4D, displayInfo=True)
        self.assertEqual(size, expected_size)

    def test_getSize_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getSize(ft.arr(self.img3D))  # type: ignore


class test_getImageCenter(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[4.0, 2.0, 1.5], centred=True)
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.img2D_custom = ft.createImg([10, 10], spacing=[2.0, 2.0], centred=True)
        self.img4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)

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

    def test_getImageCenter_4D(self):
        expected_center = (1.5, 2.0, 2.5, 3.0)
        center = ft.getImageCenter(self.img4D, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getImageCenter_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getImageCenter(ft.arr(self.img3D))  # type: ignore


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
        self.img3D_vector[5, 5, 5] = [1, 2, 3, 4, 5, 6, 7]
        self.img3D_zeros = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D_nan = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D_nan[5, 5, 5] = 1.0
        self.img3D_nan[0, 0, 0] = float("nan")

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
        expected_center = (5.5, 5.5, 5.5)
        center = ft.getMassCenter(self.img3D_vector, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getMassCenter_zeros(self):
        center = ft.getMassCenter(self.img3D_zeros, displayInfo=True)
        self.assertEqual(center, ft.getImageCenter(self.img3D_zeros))

    def test_getMassCenter_NaN(self):
        expected_center = (5.0, 5.0, 5.0)
        center = ft.getMassCenter(self.img3D_nan, displayInfo=True)
        self.assertEqual(center, expected_center)

    def test_getMassCenter_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getMassCenter(ft.arr(self.img3D))  # type: ignore


class test_getMaxPosition(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D[5, 5, 5] = 1.0
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 2.0, 2.0], centred=True)
        self.img3D_custom[5, 5, 5] = 1.0
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.img2D[3, 4] = 1.0
        self.img3D_twoMaxima = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D_twoMaxima[2, 2, 2] = 1.0
        self.img3D_twoMaxima[7, 7, 7] = 1.0

    def test_getMaxPosition(self):
        expected_position = (5.0, 5.0, 5.0)
        position = ft.getMaxPosition(self.img3D, displayInfo=True)
        self.assertEqual(position, expected_position)

    def test_getMaxPosition_custom_spacing_origin(self):
        expected_position = (1.0, 1.0, 1.0)
        position = ft.getMaxPosition(self.img3D_custom, displayInfo=True)
        self.assertEqual(position, expected_position)

    def test_getMaxPosition_2D(self):
        expected_position = (3.0, 4.0)
        position = ft.getMaxPosition(self.img2D, displayInfo=True)
        self.assertEqual(position, expected_position)

    def test_getMaxPosition_multiple_maxima(self):
        expected_position = (2.0, 2.0, 2.0)
        with self.assertLogs(ft.ImgAnalyse.imgAnalyse._logger, level="WARNING"):
            position = ft.getMaxPosition(self.img3D_twoMaxima)
        self.assertEqual(position, expected_position)

    def test_getMaxPosition_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getMaxPosition(ft.arr(self.img3D))  # type: ignore


class test_getMinPosition(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D[5, 5, 5] = -1.0
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 2.0, 2.0], centred=True)
        self.img3D_custom[5, 5, 5] = -1.0
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.img2D[3, 4] = -1.0
        self.img3D_uniform = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_getMinPosition(self):
        expected_position = (5.0, 5.0, 5.0)
        position = ft.getMinPosition(self.img3D, displayInfo=True)
        self.assertEqual(position, expected_position)

    def test_getMinPosition_custom_spacing_origin(self):
        expected_position = (1.0, 1.0, 1.0)
        position = ft.getMinPosition(self.img3D_custom, displayInfo=True)
        self.assertEqual(position, expected_position)

    def test_getMinPosition_2D(self):
        expected_position = (3.0, 4.0)
        position = ft.getMinPosition(self.img2D, displayInfo=True)
        self.assertEqual(position, expected_position)

    def test_getMinPosition_multiple_minima(self):
        expected_position = (0.0, 0.0, 0.0)
        with self.assertLogs(ft.ImgAnalyse.imgAnalyse._logger, level="WARNING"):
            position = ft.getMinPosition(self.img3D_uniform)
        self.assertEqual(position, expected_position)

    def test_getMinPosition_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getMinPosition(ft.arr(self.img3D))  # type: ignore


class test_getVoxelCentres(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 2.0, 2.0], origin=[-5.0, -5.0, -5.0])
        self.img4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)

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

    def test_getVoxelCentres_custom_spacing_origin(self):
        expected_centres = (tuple(np.linspace(-5.0, 13.0, 10)),
                            tuple(np.linspace(-5.0, 13.0, 10)),
                            tuple(np.linspace(-5.0, 13.0, 10)))
        centres = ft.getVoxelCentres(self.img3D_custom, displayInfo=True)
        self.assertEqual(centres, expected_centres)

    def test_getVoxelCentres_4D(self):
        centres = ft.getVoxelCentres(self.img4D, displayInfo=True)
        self.assertEqual([len(centresAxis) for centresAxis in centres], [4, 5, 6, 7])
        self.assertEqual(centres[3], tuple(np.linspace(0.0, 6.0, 7)))

    def test_getVoxelCentres_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getVoxelCentres(ft.arr(self.img3D))  # type: ignore


class test_getVoxelEdges(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 2.0], origin=[0.0, 0.0])
        self.img3D_custom = ft.createImg([10, 10, 10], spacing=[2.0, 2.0, 2.0], origin=[-5.0, -5.0, -5.0])
        self.img4D = sitk.Image([4, 5, 6, 7], sitk.sitkFloat32)

    def test_getVoxelEdges(self):
        expected_edges = (tuple(np.linspace(-0.5, 9.5, 11)),
                          tuple(np.linspace(-0.5, 9.5, 11)),
                          tuple(np.linspace(-0.5, 9.5, 11)))
        edges = ft.getVoxelEdges(self.img3D, displayInfo=True)
        self.assertEqual(edges, expected_edges)

    def test_getVoxelEdges_2D(self):
        expected_edges = (tuple(np.linspace(-0.5, 9.5, 11)),
                          tuple(np.linspace(-1.0, 19.0, 11)))
        edges = ft.getVoxelEdges(self.img2D, displayInfo=True)
        self.assertEqual(edges, expected_edges)

    def test_getVoxelEdges_custom_spacing_origin(self):
        expected_edges = (tuple(np.linspace(-6.0, 14.0, 11)),
                          tuple(np.linspace(-6.0, 14.0, 11)),
                          tuple(np.linspace(-6.0, 14.0, 11)))
        edges = ft.getVoxelEdges(self.img3D_custom, displayInfo=True)
        self.assertEqual(edges, expected_edges)

    def test_getVoxelEdges_4D(self):
        edges = ft.getVoxelEdges(self.img4D, displayInfo=True)
        self.assertEqual([len(edgesAxis) for edgesAxis in edges], [5, 6, 7, 8])
        self.assertEqual(edges[3], tuple(np.linspace(-0.5, 6.5, 8)))

    def test_getVoxelEdges_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getVoxelEdges(ft.arr(self.img3D))  # type: ignore


class test_getVoxelPhysicalPoints(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.imgSmall = ft.createImg([2, 2, 2], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.imgMask = ft.createEllipseMask(self.img3D, point=[5, 5, 5], radii=[2, 2, 2])

    def test_getVoxelPhysicalPoints(self):
        points = np.array(ft.getVoxelPhysicalPoints(self.img3D, displayInfo=True))
        self.assertEqual(points.shape, (1000, 3))

    def test_getVoxelPhysicalPoints_2D(self):
        points = np.array(ft.getVoxelPhysicalPoints(self.img2D, displayInfo=True))
        self.assertEqual(points.shape, (100, 2))

    def test_getVoxelPhysicalPoints_values(self):
        points = np.array(ft.getVoxelPhysicalPoints(self.imgSmall))
        expected_points = {(x, y, z) for x in [0.0, 1.0] for y in [0.0, 1.0] for z in [0.0, 1.0]}
        self.assertEqual({tuple(point) for point in points.tolist()}, expected_points)

    def test_getVoxelPhysicalPoints_insideMask(self):
        points = np.array(ft.getVoxelPhysicalPoints(self.imgMask, insideMask=True, displayInfo=True))
        self.assertEqual(points.shape, (int(ft.getStatistics(self.imgMask).GetSum()), 3))
        self.assertTrue(np.all(points >= 3.0) and np.all(points <= 7.0))

    def test_getVoxelPhysicalPoints_insideMask_invalid_mask(self):
        with self.assertRaises(TypeError):
            ft.getVoxelPhysicalPoints(self.img3D, insideMask=True)

    def test_getVoxelPhysicalPoints_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getVoxelPhysicalPoints(ft.arr(self.img3D))  # type: ignore


class test_getExtMpl(unittest.TestCase):
    def setUp(self):
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 2.0], centred=True)
        self.img3Dslice = ft.createImg([10, 1, 20], spacing=[1.0, 1.0, 2.0], origin=[0.0, 0.0, 0.0])
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dprofile = ft.createImg([1, 10, 1], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_getExtMpl(self):
        expected_extent = (-5.0, 5.0, 10.0, -10.0)
        extent = ft.getExtMpl(self.img2D)
        self.assertEqual(extent, expected_extent)

    def test_getExtMpl_3D_slice(self):
        expected_extent = (-0.5, 9.5, 39.0, -1.0)
        extent = ft.getExtMpl(self.img3Dslice)
        self.assertEqual(extent, expected_extent)

    def test_getExtMpl_invalid_volume(self):
        with self.assertRaises(TypeError):
            ft.getExtMpl(self.img3D)

    def test_getExtMpl_invalid_profile(self):
        with self.assertRaises(TypeError):
            ft.getExtMpl(self.img3Dprofile)

    def test_getExtMpl_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getExtMpl(ft.arr(self.img2D))  # type: ignore


class test_pos(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dprofile = ft.createImg([1, 5, 1], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dslice = ft.createImg([10, 1, 20], spacing=[1.0, 1.0, 2.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])

    def test_pos(self):
        expected_pos = (tuple(np.linspace(0.0, 9.0, 10)),
                        tuple(np.linspace(0.0, 9.0, 10)),
                        tuple(np.linspace(0.0, 9.0, 10)))
        pos_values = ft.pos(self.img3D)
        self.assertEqual(pos_values, expected_pos)

    def test_pos_profile(self):
        expected_pos = (0.0, 1.0, 2.0, 3.0, 4.0)
        pos_values = ft.pos(self.img3Dprofile)
        self.assertEqual(pos_values, expected_pos)

    def test_pos_slice(self):
        expected_pos = (tuple(np.linspace(0.0, 9.0, 10)),
                        tuple(np.linspace(0.0, 38.0, 20)))
        pos_values = ft.pos(self.img3Dslice)
        self.assertEqual(pos_values, expected_pos)

    def test_pos_2D(self):
        expected_pos = (tuple(np.linspace(0.0, 9.0, 10)),
                        tuple(np.linspace(0.0, 9.0, 10)))
        pos_values = ft.pos(self.img2D)
        self.assertEqual(pos_values, expected_pos)

    def test_pos_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.pos(ft.arr(self.img3D))  # type: ignore


class test_arr(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dprofile = ft.createImg([1, 5, 1], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dpoint = ft.createImg([1, 1, 1], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D_vector = ft.createImg([4, 4, 4], components=3, spacing=[1.0, 1.0, 1.0])

    def test_arr(self):
        array = ft.arr(self.img3D)
        self.assertEqual(array.shape, (10, 10, 10))

    def test_arr_values(self):
        self.img3D[1, 2, 3] = 5.0
        array = ft.arr(self.img3D)
        self.assertEqual(array[3, 2, 1], 5.0)
        self.assertEqual(array.sum(), 5.0)

    def test_arr_profile(self):
        array = ft.arr(self.img3Dprofile)
        self.assertEqual(array.shape, (5,))

    def test_arr_point(self):
        array = ft.arr(self.img3Dpoint)
        self.assertEqual(array.shape, ())

    def test_arr_vector_image(self):
        array = ft.arr(self.img3D_vector)
        self.assertEqual(array.shape, (4, 4, 4, 3))

    def test_arr_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.arr(np.zeros((10, 10, 10)))  # type: ignore


class test_vec(unittest.TestCase):
    def setUp(self):
        self.img3Dprofile = ft.createImg([1, 10, 1], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2Dprofile = ft.createImg([10, 1], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_vec(self):
        vector = ft.vec(self.img3Dprofile)
        self.assertEqual(vector.shape, (10,))

    def test_vec_values(self):
        self.img3Dprofile[0, 3, 0] = 1.0
        vector = ft.vec(self.img3Dprofile)
        self.assertEqual(vector[3], 1.0)
        self.assertEqual(vector.sum(), 1.0)

    def test_vec_2D(self):
        vector = ft.vec(self.img2Dprofile)
        self.assertEqual(vector.shape, (10,))

    def test_vec_invalid_volume(self):
        with self.assertRaises(TypeError):
            ft.vec(self.img3D)

    def test_vec_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.vec(ft.arr(self.img3Dprofile))  # type: ignore


class test_isPointInside(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])

    def test_isPointInside(self):
        point = (5.0, 5.0, 5.0)
        self.assertTrue(ft.isPointInside(self.img3D, point, displayInfo=True))

    def test_isPointInside_outside(self):
        point = (20.0, 0.0, 0.0)
        self.assertFalse(ft.isPointInside(self.img3D, point, displayInfo=True))

    def test_isPointInside_border(self):
        self.assertTrue(ft.isPointInside(self.img3D, (-0.5, -0.5, -0.5)))
        self.assertTrue(ft.isPointInside(self.img3D, (9.5, 9.5, 9.5)))
        self.assertFalse(ft.isPointInside(self.img3D, (-0.6, 0.0, 0.0)))

    def test_isPointInside_list_of_points(self):
        points = [[5.0, 5.0, 5.0], [-0.5, 0.0, 0.0], [-0.6, 0.0, 0.0], [9.5, 9.5, 9.5], [20.0, 0.0, 0.0]]
        expected_inside = (True, True, False, True, False)
        self.assertEqual(ft.isPointInside(self.img3D, points, displayInfo=True), expected_inside)
        self.assertEqual(ft.isPointInside(self.img3D, np.array(points)), expected_inside)

    def test_isPointInside_non_identity_direction(self):
        self.img3D = sitk.Flip(self.img3D, flipAxes=[True, False, False])
        self.assertEqual(ft.isPointInside(self.img3D, [[5.0, 5.0, 5.0], [20.0, 0.0, 0.0]]), (True, False))

    def test_isPointInside_2D(self):
        self.assertTrue(ft.isPointInside(self.img2D, (5.0, 5.0)))
        self.assertFalse(ft.isPointInside(self.img2D, (5.0, 20.0)))

    def test_isPointInside_invalid_point(self):
        for point in [(5.0, 5.0), (5.0, 5.0, 5.0, 5.0), 5.0, np.zeros((2, 2, 3))]:
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    ft.isPointInside(self.img3D, point)  # type: ignore

    def test_isPointInside_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.isPointInside(ft.arr(self.img3D), (5.0, 5.0, 5.0))  # type: ignore


class test_getStatistics(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.img3D_vector = ft.createImg([4, 4, 4], components=3, spacing=[1.0, 1.0, 1.0])
        self.img3D_vector[1, 1, 1] = [1, 2, 3]

    def test_getStatistics(self):
        stats = ft.getStatistics(self.img3D, displayInfo=True)
        self.assertEqual(stats.GetMean(), 0.0)

    def test_getStatistics_values(self):
        self.img3D[5, 5, 5] = 2.0
        stats = ft.getStatistics(self.img3D, displayInfo=True)
        self.assertEqual(stats.GetSum(), 2.0)
        self.assertEqual(stats.GetMaximum(), 2.0)
        self.assertEqual(stats.GetMinimum(), 0.0)
        self.assertAlmostEqual(stats.GetMean(), 0.002)

    def test_getStatistics_2D(self):
        self.img2D[3, 4] = 1.0
        stats = ft.getStatistics(self.img2D, displayInfo=True)
        self.assertEqual(stats.GetSum(), 1.0)
        self.assertEqual(stats.GetMaximum(), 1.0)

    def test_getStatistics_vector_image(self):
        with self.assertLogs(ft.ImgAnalyse.imgAnalyse._logger, level="WARNING"):
            stats = ft.getStatistics(self.img3D_vector)
        self.assertEqual(stats.GetSum(), 6.0)
        self.assertEqual(stats.GetMaximum(), 6.0)

    def test_getStatistics_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getStatistics(ft.arr(self.img3D))  # type: ignore


class test_getIntegral(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 2.0, 3.0], origin=[0.0, 0.0, 0.0]) + 2.0
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 2.0], origin=[0.0, 0.0]) + 1.0
        self.img3D_zeros = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3D_vector = ft.createImg([4, 4, 4], components=3, spacing=[1.0, 1.0, 1.0])
        self.img3D_vector[1, 1, 1] = [1, 2, 3]

    def test_getIntegral(self):
        integral = ft.getIntegral(self.img3D, displayInfo=True)
        self.assertIsInstance(integral, float)
        self.assertAlmostEqual(integral, 2.0 * 1000 * 6.0)

    def test_getIntegral_2D(self):
        integral = ft.getIntegral(self.img2D, displayInfo=True)
        self.assertAlmostEqual(integral, 1.0 * 100 * 2.0)

    def test_getIntegral_zeros(self):
        self.assertEqual(ft.getIntegral(self.img3D_zeros), 0.0)

    def test_getIntegral_vector_image(self):
        with self.assertLogs(ft.ImgAnalyse.imgAnalyse._logger, level="WARNING"):
            integral = ft.getIntegral(self.img3D_vector)
        self.assertAlmostEqual(integral, 6.0)

    def test_getIntegral_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.getIntegral(ft.arr(self.img3D))  # type: ignore


class test_compareImgFoR(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test_compareImgFoR(self):
        img2 = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.assertTrue(ft.compareImgFoR(self.img3D, img2, displayInfo=True))

    def test_compareImgFoR_fail(self):
        img2 = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[1.0, 0.0, 0.0])
        self.assertFalse(ft.compareImgFoR(self.img3D, img2, displayInfo=True))

    def test_compareImgFoR_different_size(self):
        img2 = ft.createImg([10, 10, 5], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.assertFalse(ft.compareImgFoR(self.img3D, img2, displayInfo=True))

    def test_compareImgFoR_different_spacing(self):
        img2 = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 2.0], origin=[0.0, 0.0, 0.0])
        self.assertFalse(ft.compareImgFoR(self.img3D, img2, displayInfo=True))

    def test_compareImgFoR_different_direction(self):
        img2 = sitk.Flip(self.img3D, flipAxes=[True, False, False])
        self.assertFalse(ft.compareImgFoR(self.img3D, img2, displayInfo=True))

    def test_compareImgFoR_different_dimension(self):
        img2 = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])
        self.assertFalse(ft.compareImgFoR(self.img3D, img2, displayInfo=True))

    def test_compareImgFoR_decimal(self):
        img2 = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0004, 0.0, 0.0])
        self.assertTrue(ft.compareImgFoR(self.img3D, img2, decimal=3))
        self.assertFalse(ft.compareImgFoR(self.img3D, img2, decimal=5))

    def test_compareImgFoR_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.compareImgFoR(self.img3D, ft.arr(self.img3D))  # type: ignore
        with self.assertRaises(TypeError):
            ft.compareImgFoR(ft.arr(self.img3D), self.img3D)  # type: ignore


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

    def test_compareImg_decimal(self):
        img2 = self.img3D + 0.0004
        self.assertTrue(ft.compareImg(self.img3D, img2, decimal=3))
        self.assertFalse(ft.compareImg(self.img3D, img2, decimal=5))

    def test_compareImg_different_pixel_type(self):
        img2 = sitk.Cast(self.img3D, sitk.sitkFloat64)
        self.assertTrue(ft.compareImg(self.img3D, img2, displayInfo=True))

    def test_compareImg_different_size(self):
        img2 = ft.createImg([5, 5, 5], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.assertFalse(ft.compareImg(self.img3D, img2, displayInfo=True))

    def test_compareImg_NaN(self):
        self.img3D[0, 0, 0] = float("nan")
        img2 = sitk.Image(self.img3D)
        self.assertTrue(ft.compareImg(self.img3D, img2))

    def test_compareImg_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.compareImg(self.img3D, ft.arr(self.img3D))  # type: ignore
        with self.assertRaises(TypeError):
            ft.compareImg(ft.arr(self.img3D), self.img3D)  # type: ignore


class test__getAxesVectorNotUnity(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dslice = ft.createImg([10, 1, 20], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dpoint = ft.createImg([1, 1, 1], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test__getAxesVectorNotUnity(self):
        self.assertEqual(ft.ImgAnalyse.imgAnalyse._getAxesVectorNotUnity(self.img3D), (1, 1, 1))
        self.assertEqual(ft.ImgAnalyse.imgAnalyse._getAxesVectorNotUnity(self.img3Dslice), (1, 0, 1))
        self.assertEqual(ft.ImgAnalyse.imgAnalyse._getAxesVectorNotUnity(self.img3Dpoint), (0, 0, 0))

    def test__getAxesVectorNotUnity_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.ImgAnalyse.imgAnalyse._getAxesVectorNotUnity(ft.arr(self.img3D))  # type: ignore


class test__getAxesNumberNotUnity(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dslice = ft.createImg([10, 1, 20], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dpoint = ft.createImg([1, 1, 1], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test__getAxesNumberNotUnity(self):
        self.assertEqual(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(self.img3D), (0, 1, 2))
        self.assertEqual(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(self.img3Dslice), (0, 2))
        self.assertEqual(ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(self.img3Dpoint), ())

    def test__getAxesNumberNotUnity_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.ImgAnalyse.imgAnalyse._getAxesNumberNotUnity(ft.arr(self.img3D))  # type: ignore


class test__getAxesNumberUnity(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dslice = ft.createImg([10, 1, 20], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img3Dpoint = ft.createImg([1, 1, 1], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])

    def test__getAxesNumberUnity(self):
        self.assertEqual(ft.ImgAnalyse.imgAnalyse._getAxesNumberUnity(self.img3D), ())
        self.assertEqual(ft.ImgAnalyse.imgAnalyse._getAxesNumberUnity(self.img3Dslice), (1,))
        self.assertEqual(ft.ImgAnalyse.imgAnalyse._getAxesNumberUnity(self.img3Dpoint), (0, 1, 2))

    def test__getAxesNumberUnity_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.ImgAnalyse.imgAnalyse._getAxesNumberUnity(ft.arr(self.img3D))  # type: ignore


class test__getDirectionArray(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])

    def test__getDirectionArray(self):
        direction = ft.ImgAnalyse.imgAnalyse._getDirectionArray(self.img3D)
        self.assertEqual(direction.shape, (3, 3))
        self.assertTrue(np.array_equal(direction, np.identity(3)))

    def test__getDirectionArray_2D(self):
        direction = ft.ImgAnalyse.imgAnalyse._getDirectionArray(self.img2D)
        self.assertEqual(direction.shape, (2, 2))
        self.assertTrue(np.array_equal(direction, np.identity(2)))

    def test__getDirectionArray_non_identity(self):
        direction = ft.ImgAnalyse.imgAnalyse._getDirectionArray(sitk.Flip(self.img3D, flipAxes=[True, False, False]))
        self.assertTrue(np.array_equal(direction, np.diag([-1.0, 1.0, 1.0])))

    def test__getDirectionArray_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.ImgAnalyse.imgAnalyse._getDirectionArray(ft.arr(self.img3D))  # type: ignore


class test__isDirectionIdentity(unittest.TestCase):
    def setUp(self):
        self.img3D = ft.createImg([10, 10, 10], spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0])
        self.img2D = ft.createImg([10, 10], spacing=[1.0, 1.0], origin=[0.0, 0.0])

    def test__isDirectionIdentity(self):
        self.assertTrue(ft.ImgAnalyse.imgAnalyse._isDirectionIdentity(self.img3D))
        self.assertTrue(ft.ImgAnalyse.imgAnalyse._isDirectionIdentity(self.img2D))

    def test__isDirectionIdentity_flipped(self):
        self.assertFalse(ft.ImgAnalyse.imgAnalyse._isDirectionIdentity(sitk.Flip(self.img3D, flipAxes=[True, False, False])))
        self.assertFalse(ft.ImgAnalyse.imgAnalyse._isDirectionIdentity(sitk.Flip(self.img2D, flipAxes=[False, True])))

    def test__isDirectionIdentity_permuted(self):
        self.assertFalse(ft.ImgAnalyse.imgAnalyse._isDirectionIdentity(sitk.PermuteAxes(self.img3D, [1, 0, 2])))

    def test__isDirectionIdentity_invalid_img(self):
        with self.assertRaises(TypeError):
            ft.ImgAnalyse.imgAnalyse._isDirectionIdentity(ft.arr(self.img3D))  # type: ignore


if __name__ == '__main__':
    unittest.main()
