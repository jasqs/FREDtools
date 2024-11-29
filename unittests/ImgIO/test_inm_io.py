import unittest
import os
from pathlib import Path
import SimpleITK as sitk
import fredtools as ft
import pandas as pd
import numpy as np
import re
from itertools import chain

testPath = Path(os.path.dirname(__file__))


class test_getInmFREDVersion(unittest.TestCase):

    def setUp(self):
        self.testDir = Path("unittests/testData/INMImages")
        self.filePaths = Path.glob(self.testDir, "v*.*.bin")

    def test_getInmFREDVersion(self):
        for filePath in self.filePaths:
            fileVersion = re.findall(r"v(\d+).\S+.bin", str(filePath))
            if fileVersion:
                fileVersion = float(fileVersion[0])
                with self.subTest(version=fileVersion, fileName=filePath.name):
                    self.assertEqual(ft.ImgIO.inm_io.getInmFREDVersion(filePath), fileVersion)


class test_isInmFRED(unittest.TestCase):

    def setUp(self):
        self.testDir = Path("unittests/testData/INMImages")
        self.filePaths = Path.glob(self.testDir, "v*.*.*")

    def test_isInmFRED(self):
        for filePath in self.filePaths:
            with self.subTest(fileName=filePath.name):
                if re.search(r"bin$", str(filePath)):
                    self.assertTrue(ft.ImgIO.inm_io._isInmFRED(filePath))
                else:
                    self.assertFalse(ft.ImgIO.inm_io._isInmFRED(filePath))
                    with self.assertRaises(TypeError):
                        ft.ImgIO.inm_io._isInmFRED(filePath, raiseError=True)


class test_getInmFREDInfo(unittest.TestCase):

    def setUp(self):
        self.testDir = Path("unittests/testData/INMImages")
        self.filePathsDose = Path.glob(self.testDir, "v*.Dose.bin")
        self.filePathsLETd = Path.glob(self.testDir, "v*.LETd.bin")
        self.filePaths = chain(self.filePathsDose, self.filePathsLETd)

    def test_getInmFREDInfo(self):
        for filePath in self.filePaths:
            with self.subTest(fileName=filePath.name):
                inmInfo = ft.getInmFREDInfo(filePath, displayInfo=True)
                self.assertIsInstance(inmInfo, pd.DataFrame)
                self.assertIn("FID", inmInfo.columns)
                self.assertIn("PBID", inmInfo.columns)
                self.assertIn("voxelsNo", inmInfo.columns)

    def test_getInmFREDInfo_content_Dose(self):
        inmInfoFiles = []
        for filePath in self.filePathsDose:
            inmInfoFiles.append(ft.getInmFREDInfo(filePath, displayInfo=True))
        for inmInfo in inmInfoFiles:
            self.assertEqual(inmInfo.shape[0], 9)
            self.assertTrue(inmInfoFiles[0].equals(inmInfo))

    def test_getInmFREDInfo_content_LETd(self):
        inmInfoFiles = []
        for filePath in self.filePathsLETd:
            inmInfoFiles.append(ft.getInmFREDInfo(filePath, displayInfo=True))
        for inmInfo in inmInfoFiles:
            self.assertEqual(inmInfo.shape[0], 9)
            self.assertTrue(inmInfoFiles[0].equals(inmInfo))


class test_getInmFREDBaseImg(unittest.TestCase):

    def setUp(self):
        self.testDir = Path("unittests/testData/INMImages")
        self.filePathsDose = Path.glob(self.testDir, "v*.Dose.bin")
        self.filePathsLETd = Path.glob(self.testDir, "v*.LETd.bin")
        self.filePaths = chain(self.filePathsDose, self.filePathsLETd)

    def test_getInmFREDBaseImg(self):
        for filePath in self.filePaths:
            with self.subTest(fileName=filePath.name):
                imgRef = ft.readMHD(str(filePath).replace(".bin", ".mhd"))
                imgBase = ft.getInmFREDBaseImg(filePath, displayInfo=True)
                self.assertTrue(ft.compareImgFoR(imgRef, imgBase, decimal=5))


class test_getInmFREDSparse(unittest.TestCase):

    def setUp(self):
        self.testDir = Path("unittests/testData/INMImages")
        self.filePathsDose = Path.glob(self.testDir, "v*.Dose.bin")
        self.filePathsLETd = Path.glob(self.testDir, "v*.LETd.bin")
        self.filePaths = chain(self.filePathsDose, self.filePathsLETd)
        self.primaryNo = pd.read_csv(self.testDir.joinpath("primaryNo.csv"), delimiter=r"\s+")
        self.imgROI = ft.readMHD(self.testDir.joinpath("roiBrain.mhd"))
        self.imgROIpoints = ft.getVoxelPhysicalPoints(self.imgROI, insideMask=True)

    def test_getInmFREDSparse_CPU_singleComponent(self):
        for filePath in self.filePathsDose:
            with self.subTest(fileName=filePath.name):
                # read the reference image
                imgRef = ft.readMHD(str(filePath).replace(".bin", ".mhd"))
                # replace NaNs with zeros in the reference image
                imgRef = ft.setNaNImg(imgRef, value=0)
                # read the sparse matrix
                imnSparse = ft.getInmFREDSparse(filePath, points=self.imgROIpoints, interpreter="numpy", displayInfo=True)
                self.assertIsInstance(imnSparse, list)
                self.assertEqual(len(imnSparse), 1)

                # sum the sparse matrix and convert to image
                vecSum = np.asarray(imnSparse[0].T.dot(self.primaryNo.N))
                imgSum = sitk.GetImageFromArray(np.reshape(vecSum, imgRef.GetSize()[::-1]))
                imgSum.CopyInformation(imgRef)
                imgSum = sitk.Cast(imgSum, sitk.sitkFloat32)

                self.assertTrue(ft.compareImg(imgRef, imgSum, decimal=6))

    def test_getInmFREDSparse_GPU_singleComponent(self):
        if not ft._helper.checkGPUcupy():
            self.skipTest("No GPU available")
        else:
            import cupy as cp

        for filePath in self.filePathsDose:
            with self.subTest(fileName=filePath.name):
                # read the reference image
                imgRef = ft.readMHD(str(filePath).replace(".bin", ".mhd"))
                # replace NaNs with zeros in the reference image
                imgRef = ft.setNaNImg(imgRef, value=0)
                # read the sparse matrix
                imnSparse = ft.getInmFREDSparse(filePath, points=self.imgROIpoints, interpreter="cupy", displayInfo=True)
                self.assertIsInstance(imnSparse, list)
                self.assertEqual(len(imnSparse), 1)

                # sum the sparse matrix and convert to image
                primaryNo = cp.asarray(self.primaryNo.N)
                vecSum = cp.asarray(imnSparse[0].T.dot(primaryNo))
                vecSum = cp.asnumpy(vecSum)
                imgSum = sitk.GetImageFromArray(np.reshape(vecSum, imgRef.GetSize()[::-1]))
                imgSum.CopyInformation(imgRef)
                imgSum = sitk.Cast(imgSum, sitk.sitkFloat32)

                self.assertTrue(ft.compareImg(imgRef, imgSum, decimal=6))

    def test_getInmFREDSparse_CPU_twoComponents(self):
        for filePath in self.filePathsLETd:
            with self.subTest(fileName=filePath.name):
                # read the reference image
                imgRef = ft.readMHD(str(filePath).replace(".bin", ".mhd"))
                # replace NaNs with zeros in the reference image
                imgRef = ft.setNaNImg(imgRef, value=0)
                # read the sparse matrix
                imnSparse = ft.getInmFREDSparse(filePath, points=self.imgROIpoints, interpreter="numpy", displayInfo=True)
                self.assertIsInstance(imnSparse, list)
                self.assertEqual(len(imnSparse), 2)

                # sum the sparse matrix and convert to image for each component
                vecSum0 = np.asarray(imnSparse[0].T.dot(self.primaryNo.N))
                imgSum0 = sitk.GetImageFromArray(np.reshape(vecSum0, imgRef.GetSize()[::-1]))
                imgSum0.CopyInformation(imgRef)
                vecSum1 = np.asarray(imnSparse[1].T.dot(self.primaryNo.N))
                imgSum1 = sitk.GetImageFromArray(np.reshape(vecSum1, imgRef.GetSize()[::-1]))
                imgSum1.CopyInformation(imgRef)

                # divide the two components
                imgSum = ft.divideImg(imgSum0, imgSum1)
                imgSum = sitk.Cast(imgSum, sitk.sitkFloat32)

                self.assertTrue(ft.compareImg(imgRef, imgSum, decimal=4))

    def test_getInmFREDSparse_GPU_twoComponents(self):
        if not ft._helper.checkGPUcupy():
            self.skipTest("No GPU available")
        else:
            import cupy as cp

        for filePath in self.filePathsLETd:
            with self.subTest(fileName=filePath.name):
                # read the reference image
                imgRef = ft.readMHD(str(filePath).replace(".bin", ".mhd"))
                # replace NaNs with zeros in the reference image
                imgRef = ft.setNaNImg(imgRef, value=0)
                # read the sparse matrix
                imnSparse = ft.getInmFREDSparse(filePath, points=self.imgROIpoints, interpreter="cupy", displayInfo=True)
                self.assertIsInstance(imnSparse, list)
                self.assertEqual(len(imnSparse), 2)

                # sum the sparse matrix and convert to image for each component
                primaryNo = cp.asarray(self.primaryNo.N)
                vecSum0 = cp.asarray(imnSparse[0].T.dot(primaryNo))
                vecSum0 = cp.asnumpy(vecSum0)
                imgSum0 = sitk.GetImageFromArray(np.reshape(vecSum0, imgRef.GetSize()[::-1]))
                imgSum0.CopyInformation(imgRef)
                vecSum1 = cp.asarray(imnSparse[1].T.dot(primaryNo))
                vecSum1 = cp.asnumpy(vecSum1)
                imgSum1 = sitk.GetImageFromArray(np.reshape(vecSum1, imgRef.GetSize()[::-1]))
                imgSum1.CopyInformation(imgRef)

                # divide the two components
                imgSum = ft.divideImg(imgSum0, imgSum1)
                imgSum = sitk.Cast(imgSum, sitk.sitkFloat32)

                self.assertTrue(ft.compareImg(imgRef, imgSum, decimal=4))


if __name__ == "__main__":
    unittest.main()
