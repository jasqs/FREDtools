import unittest
import os
from pathlib import Path
import SimpleITK as sitk
import fredtools as ft
import pandas as pd
import numpy as np


class test_inmManipulate(unittest.TestCase):

    def setUp(self):
        testDir = Path("unittests/testData/INMImages")
        self.imgRef = ft.readMHD(testDir.joinpath("v3.Dose.mhd"))
        self.imgRef = ft.setNaNImg(self.imgRef, value=0)

        self.imgBase = ft.getInmFREDBaseImg(testDir.joinpath("v3.Dose.bin"))
        points = ft.getVoxelPhysicalPoints(self.imgBase)
        self.inmBase = ft.getInmFREDSparse(testDir.joinpath("v3.Dose.bin"), points)[0]
        self.weights = pd.read_csv(testDir.joinpath("primaryNo.csv"), delimiter=r"\s+").N.to_numpy()

    def test_inmSumVec_CPU(self):
        vecRef = np.swapaxes(sitk.GetArrayViewFromImage(self.imgRef), 0, 2).flatten(order='F')
        vecSum = ft.inmSumVec(self.inmBase, self.weights, displayInfo=True)
        self.assertTrue(np.isclose(vecSum, vecRef, rtol=0, atol=1E-6).all())

    def test_inmSumVec_GPU(self):
        if not ft._helper.checkGPUcupy():
            self.skipTest("No GPU available")
        else:
            import cupy as cp

        vecRef = np.swapaxes(sitk.GetArrayViewFromImage(self.imgRef), 0, 2).flatten(order='F')
        vecSum = ft.inmSumVec(cp.sparse.csr_matrix(self.inmBase), self.weights, displayInfo=True)
        self.assertTrue(np.isclose(vecSum, vecRef, rtol=0, atol=1E-6).all())

    def test_inmSumImg_CPU(self):
        imgSum = ft.inmSumImg(self.inmBase, self.weights, self.imgBase, displayInfo=True)
        self.assertTrue(ft.compareImgFoR(imgSum, self.imgRef))
        self.assertTrue(ft.compareImg(imgSum, self.imgRef))

    def test_inmSumImg_GPU(self):
        if not ft._helper.checkGPUcupy():
            self.skipTest("No GPU available")
        else:
            import cupy as cp

        imgSum = ft.inmSumImg(cp.sparse.csr_matrix(self.inmBase), self.weights, self.imgBase, displayInfo=True)
        self.assertTrue(ft.compareImgFoR(imgSum, self.imgRef))
        self.assertTrue(ft.compareImg(imgSum, self.imgRef))

    def test_inmSumVec_invalidWeights(self):
        with self.assertRaises(ValueError):
            ft.inmSumVec(self.inmBase, self.weights[:-1])


if __name__ == '__main__':
    unittest.main()
