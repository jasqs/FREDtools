import unittest
import SimpleITK as sitk
import itk
import fredtools as ft
from fredtools._typing import *


class test_imgConverter(unittest.TestCase):

    def setUp(self):
        # Create a SimpleITK image for testing
        self.imgSITK = ft.createImg((10, 10), spacing=(0.5, 0.5), origin=(1, 2), fillRandom=True)

        # Create an ITK image for testing
        self.imgITK = itk.image_from_array(sitk.GetArrayFromImage(self.imgSITK))
        self.imgITK.SetOrigin(self.imgSITK.GetOrigin())
        self.imgITK.SetSpacing(self.imgSITK.GetSpacing())

    def test_SITK2ITK(self):
        imgITK = ft.SITK2ITK(self.imgSITK)
        self.assertTrue(isinstance(imgITK, ITKImage))
        self.assertEqual(imgITK.GetOrigin(), self.imgSITK.GetOrigin())
        self.assertEqual(imgITK.GetSpacing(), self.imgSITK.GetSpacing())
        self.assertTrue((itk.array_from_matrix(imgITK.GetDirection()) == itk.array_from_matrix(self.imgITK.GetDirection())).all())
        self.assertEqual(sitk.GetArrayFromImage(self.imgSITK).tolist(), itk.array_from_image(imgITK).tolist())

    def test_ITK2SITK(self):
        imgSITK = ft.ITK2SITK(self.imgITK)
        self.assertTrue(isinstance(imgSITK, SITKImage))
        self.assertEqual(imgSITK.GetOrigin(), self.imgITK.GetOrigin())
        self.assertEqual(imgSITK.GetSpacing(), self.imgITK.GetSpacing())
        self.assertTrue((sitk.GetArrayFromImage(imgSITK) == sitk.GetArrayFromImage(self.imgSITK)).all())
        self.assertEqual(sitk.GetArrayFromImage(imgSITK).tolist(), itk.array_from_image(self.imgITK).tolist())


if __name__ == '__main__':
    unittest.main()
