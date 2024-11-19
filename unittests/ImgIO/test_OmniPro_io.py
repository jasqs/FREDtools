import unittest
from pathlib import Path
import numpy as np

import fredtools as ft
from fredtools._typing import *


class TestOmniProIO(unittest.TestCase):

    def setUp(self):
        self.testDir = Path("unittests/testData/OmniPro")
        self.filePath_opg = Path.joinpath(self.testDir, "image.opg")
        self.filePath_opd = Path.joinpath(self.testDir, "image.opd")

    def test_readOPG(self):
        img = ft.readOPG(self.filePath_opg, depth=5.0, displayInfo=True)
        self.assertIsInstance(img, SITKImage)
        self.assertEqual(img.GetDepth(), 1)
        self.assertListEqual(np.round(img.GetSpacing(), decimals=5).tolist(), [7.61935, 7.61935, 0.1])
        self.assertListEqual(np.round(img.GetOrigin(), decimals=5).tolist(), [-118.1, -118.1, 5.0])
        self.assertAlmostEqual((ft.getStatistics(img).GetMean()), 0.0609984375)

    def test_readOPD(self):
        imgs = ft.readOPD(self.filePath_opd, depth=5.0, returnImg=["Integral", "Sum"], raiseWarning=True, displayInfo=True)
        self.assertIsInstance(imgs, list)
        self.assertGreater(len(imgs), 0)
        meanSum = 0
        for img in imgs:
            self.assertIsInstance(img, SITKImage)
            self.assertEqual(img.GetDepth(), 1)
            self.assertListEqual(np.round(img.GetSpacing(), decimals=5).tolist(), [7.61935, 7.61935, 0.1])
            self.assertListEqual(np.round(img.GetOrigin(), decimals=5).tolist(), [-118.1, -118.1, 5.0])
            meanSum += ft.getStatistics(img).GetMean()
        self.assertAlmostEqual((meanSum), 0.06504363671874999)


if __name__ == '__main__':
    unittest.main()
