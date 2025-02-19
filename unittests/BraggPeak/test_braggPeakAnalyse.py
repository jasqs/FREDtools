import unittest
import numpy as np
import pandas as pd
import fredtools as ft


class test_braggPeak(unittest.TestCase):

    def setUp(self):
        # Load data from CSV file
        data = pd.read_csv('unittests/testData/BraggPeaks/E70.00_NeT0.0.csv', delimiter='\t')
        self.pos = np.array(data['depth'].values)
        self.vec = np.array(data['signal'].values)
        self.braggPeak = ft.braggPeak(self.pos, self.vec)

    def test_braggPeak_initialization(self):
        self.assertTrue(np.array_equal(self.braggPeak.bp[0], self.pos))
        self.assertTrue(np.array_equal(self.braggPeak.bp[1], self.vec))
        self.assertEqual(self.braggPeak.accuracy, 0.01)
        self.assertEqual(self.braggPeak.offset, 0)
        self.assertEqual(self.braggPeak.interpolation, 'spline')
        self.assertEqual(self.braggPeak.splineOrder, 3)
        self.assertEqual(self.braggPeak.bortCut, 0.6)

    def test_braggPeak_offset_setter(self):
        self.braggPeak.offset = 5
        self.assertEqual(self.braggPeak.offset, 5)

    def test_braggPeak_interpolation_setter(self):
        self.braggPeak.interpolation = 'linear'
        self.assertEqual(self.braggPeak.interpolation, 'linear')
        with self.assertRaises(ValueError):
            self.braggPeak.interpolation = 'invalid'  # type: ignore

    def test_braggPeak_splineOrder_setter(self):
        self.braggPeak.splineOrder = 4
        self.assertEqual(self.braggPeak.splineOrder, 4)
        with self.assertRaises(ValueError):
            self.braggPeak.splineOrder = 6

    def test_braggPeak_accuracy_setter(self):
        self.braggPeak.accuracy = 0.05
        self.assertEqual(self.braggPeak.accuracy, 0.05)
        with self.assertRaises(ValueError):
            self.braggPeak.accuracy = -0.01

    def test_braggPeak_bortCut_setter(self):
        self.braggPeak.bortCut = 0.8
        self.assertEqual(self.braggPeak.bortCut, 0.8)
        with self.assertRaises(ValueError):
            self.braggPeak.bortCut = 1.2

    def test_braggPeak_getDInterp(self):
        self.assertAlmostEqual(self.braggPeak.getDInterp(50), 1.0, places=2)

    def test_braggPeak_getDBort(self):
        self.assertAlmostEqual(self.braggPeak.getDBort(50), 1.0, places=2)

    def test_braggPeak_getRInterp(self):
        self.assertAlmostEqual(self.braggPeak.getRInterp(80), 50, places=2)

    def test_braggPeak_getRBort(self):
        self.assertAlmostEqual(self.braggPeak.getRBort(0.5), 50, places=2)

    def test_braggPeak_getWInterp(self):
        self.assertAlmostEqual(self.braggPeak.getWInterp(0.5), 20, places=2)

    def test_braggPeak_getWBort(self):
        self.assertAlmostEqual(self.braggPeak.getWBort(0.5), 20, places=2)

    def test_braggPeak_getDFOInterp(self):
        self.assertAlmostEqual(self.braggPeak.getDFOInterp(0.8, 0.2), 20, places=2)

    def test_braggPeak_getDFOBort(self):
        self.assertAlmostEqual(self.braggPeak.getDFOBort(0.8, 0.2), 20, places=2)


if __name__ == '__main__':
    unittest.main()
