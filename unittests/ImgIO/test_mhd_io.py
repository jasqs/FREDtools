import unittest
import os
from pathlib import Path
import SimpleITK as sitk
import fredtools as ft

testPath = Path(os.path.dirname(__file__))


class test_mhd_io(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path.joinpath(testPath, "test_mhd_io")
        self.test_dir.mkdir(exist_ok=True)
        self.filePath = Path.joinpath(self.test_dir, "test.mhd")
        self.img = ft.createImg((10, 10, 10), spacing=(1, 1, 1), origin=(0, 0, 0), fillRandom=True)
        self.filePaths = [Path.joinpath(self.test_dir, "test1.mhd"),
                          Path.joinpath(self.test_dir, "test2.mhd")]

    def tearDown(self):
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()

    def test_writeMHD_single_file(self):
        ft.writeMHD(self.img, self.filePath, singleFile=True, overwrite=True, displayInfo=True)
        self.assertTrue(self.filePath.exists())

    def test_writeMHD_double_file(self):
        ft.writeMHD(self.img, self.filePath, singleFile=False, overwrite=True)
        self.assertTrue(self.filePath.exists())
        self.assertTrue(Path.joinpath(self.test_dir, "test.raw").exists())

    def test_writeMHD_overwrite(self):
        ft.writeMHD(self.img, self.filePath, singleFile=True, overwrite=True)
        with self.assertRaises(ValueError):
            ft.writeMHD(self.img, self.filePath, singleFile=True, overwrite=False)

    def test_writeMHD_compression(self):
        ft.writeMHD(self.img, self.filePath, singleFile=True, overwrite=True, useCompression=True)
        self.assertTrue(self.filePath.exists())

    def test_readMHD_single_file(self):
        ft.writeMHD(self.img, self.filePath, singleFile=True, overwrite=True)
        result = ft.readMHD(self.filePath, displayInfo=True)
        self.assertIsInstance(result, sitk.Image)

    def test_readMHD_multiple_files(self):
        for path in self.filePaths:
            ft.writeMHD(self.img, path, singleFile=True, overwrite=True)
        result = ft.readMHD([path for path in self.filePaths])
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], sitk.Image)
        self.assertIsInstance(result[1], sitk.Image)

    def test_readMHD_compare(self):
        ft.writeMHD(self.img, self.filePath, singleFile=True, overwrite=True)
        result = ft.readMHD(self.filePath)
        self.assertTrue(ft.compareImgFoR(result, self.img))
        self.assertEqual(sitk.GetArrayFromImage(result).tolist(), sitk.GetArrayFromImage(self.img).tolist())

    def test_convertMHDtoSingleFile(self):
        ft.writeMHD(self.img, self.filePath, singleFile=False, overwrite=True)
        ft.convertMHDtoSingleFile(self.filePath, displayInfo=True)
        self.assertTrue(self.filePath.exists())
        self.assertFalse(Path.joinpath(self.test_dir, "test.raw").exists())
        result = ft.readMHD(self.filePath)
        self.assertTrue(ft.compareImgFoR(result, self.img))
        self.assertEqual(sitk.GetArrayFromImage(result).tolist(), sitk.GetArrayFromImage(self.img).tolist())

    def test_convertMHDtoDoubleFiles(self):
        ft.writeMHD(self.img, str(self.filePath), singleFile=True, overwrite=True)
        ft.convertMHDtoDoubleFiles(str(self.filePath), displayInfo=True)
        self.assertTrue(self.filePath.exists())
        self.assertTrue(Path.joinpath(self.test_dir, "test.raw").exists())
        result = ft.readMHD(self.filePath)
        self.assertTrue(ft.compareImgFoR(result, self.img))
        self.assertEqual(sitk.GetArrayFromImage(result).tolist(), sitk.GetArrayFromImage(self.img).tolist())


if __name__ == '__main__':
    unittest.main()
