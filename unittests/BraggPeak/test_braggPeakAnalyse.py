import unittest
import numpy as np
import pandas as pd
import fredtools as ft


class test_braggPeak(unittest.TestCase):

    def setUp(self):
        # Load data from CSV file
        self.BPenergies = [70, 150, 225]
        fileNames = [f'unittests/testData/BraggPeaks/E{BPenergy}.00.csv' for BPenergy in self.BPenergies]
        self.BPdata = [pd.read_csv(fileName, delimiter='\t') for fileName in fileNames]
        # self.braggPeaks = [ft.braggPeak(data['depth'], data['signal']) for data in self.BPdata]
        # self.BPdata = pd.read_csv(, delimiter='\t')
        # self.braggPeak = ft.braggPeak(self.BPdata['depth'], self.BPdata['signal'])

    def test_braggPeak_initialization(self):
        braggPeak = ft.braggPeak(self.BPdata[1]['depth'], self.BPdata[1]['signal'])
        self.assertTrue(np.array_equal(braggPeak.bp[0], self.BPdata[1]['depth']))
        self.assertTrue(np.array_equal(braggPeak.bp[1], self.BPdata[1]['signal']))
        self.assertEqual(braggPeak.accuracy, 0.01)
        self.assertEqual(braggPeak.offset, 0)
        self.assertEqual(braggPeak.interpolation, 'spline')
        self.assertEqual(braggPeak.splineOrder, 3)
        self.assertEqual(braggPeak.bortCut, 0.6)

    def test_braggPeak_offset_setter(self):
        braggPeak = ft.braggPeak(self.BPdata[1]['depth'], self.BPdata[1]['signal'])
        braggPeak.offset = 5
        self.assertEqual(braggPeak.offset, 5)

    def test_braggPeak_interpolation_setter(self):
        braggPeak = ft.braggPeak(self.BPdata[1]['depth'], self.BPdata[1]['signal'])
        braggPeak.interpolation = 'linear'
        self.assertEqual(braggPeak.interpolation, 'linear')
        braggPeak.interpolation = 'nearest'
        self.assertEqual(braggPeak.interpolation, 'nearest')
        braggPeak.interpolation = 'spline'
        self.assertEqual(braggPeak.interpolation, 'spline')
        with self.assertRaises(ValueError):
            braggPeak.interpolation = 'invalid'  # type: ignore

    def test_braggPeak_splineOrder_setter(self):
        braggPeak = ft.braggPeak(self.BPdata[1]['depth'], self.BPdata[1]['signal'])
        braggPeak.splineOrder = 4
        self.assertEqual(braggPeak.splineOrder, 4)
        with self.assertRaises(ValueError):
            braggPeak.splineOrder = 6

    def test_braggPeak_accuracy_setter(self):
        braggPeak = ft.braggPeak(self.BPdata[1]['depth'], self.BPdata[1]['signal'])
        braggPeak.accuracy = 0.05
        self.assertEqual(braggPeak.accuracy, 0.05)
        with self.assertRaises(ValueError):
            braggPeak.accuracy = -0.01

    def test_braggPeak_bortCut_setter(self):
        braggPeak = ft.braggPeak(self.BPdata[1]['depth'], self.BPdata[1]['signal'])
        braggPeak.bortCut = 0.8
        self.assertEqual(braggPeak.bortCut, 0.8)
        with self.assertRaises(ValueError):
            braggPeak.bortCut = 1.2

    def test_braggPeak_getRInterp(self):
        for energyIdx, energy in enumerate(self.BPenergies):
            with self.subTest(f"Bragg peak E{energy}"):
                braggPeak = ft.braggPeak(self.BPdata[energyIdx]['depth'], self.BPdata[energyIdx]['signal'])
                # getR(1) agree within 0.5% or 0.05 mm
                evalValue = self.BPdata[energyIdx].loc[self.BPdata[energyIdx].signal == self.BPdata[energyIdx].signal.max()].depth.values[0]
                self.assertTrue(np.allclose(braggPeak.getRInterp(1), evalValue, rtol=0.005, atol=0.05), msg=f"getR(1)\n\tRef:  {braggPeak.getRInterp(1)}\n\tEval: {evalValue}")

    def test_braggPeak_getDInterp(self):
        for energyIdx, energy in enumerate(self.BPenergies):
            with self.subTest(f"Bragg peak E{energy}"):
                braggPeak = ft.braggPeak(self.BPdata[energyIdx]['depth'], self.BPdata[energyIdx]['signal'])
                # getD(50) agree within 0.05%
                evalValue = self.BPdata[energyIdx]['signal'].max()
                self.assertTrue(np.allclose(braggPeak.getDInterp(braggPeak.getRInterp(1)), evalValue, rtol=0.005, atol=0.00), msg=f"braggPeak.getDInterp(braggPeak.getRInterp(1))\n\tRef:  {braggPeak.getDInterp(braggPeak.getRInterp(1))}\n\tEval: {evalValue}")

    # def test_braggPeak_getWInterp(self):
    #     self.assertAlmostEqual(self.braggPeak.getWInterp(0.5), 20, places=2)

    # def test_braggPeak_getWBort(self):
    #     self.assertAlmostEqual(self.braggPeak.getWBort(0.5), 20, places=2)

    # def test_braggPeak_getDFOInterp(self):
    #     self.assertAlmostEqual(self.braggPeak.getDFOInterp(0.8, 0.2), 20, places=2)

    # def test_braggPeak_getDFOBort(self):
    #     self.assertAlmostEqual(self.braggPeak.getDFOBort(0.8, 0.2), 20, places=2)


if __name__ == '__main__':
    unittest.main()
