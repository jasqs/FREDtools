import unittest
import pandas as pd
import fredtools as ft


class test_beamModel(unittest.TestCase):

    def test_beamModel_splineOrder(self):
        """The spline order of the energy model must be in range 1-5."""
        beamModel = ft.beamModel()
        for splineOrder in range(1, 6):
            beamModel.splineOrder = splineOrder
            self.assertEqual(beamModel.splineOrder, splineOrder)
        for splineOrder in (0, 6):
            with self.subTest(splineOrder=splineOrder):
                with self.assertRaises(AttributeError):
                    beamModel.splineOrder = splineOrder


class test_interpolateBeamModel(unittest.TestCase):

    def test_interpolateBeamModel_invalidSplineOrder(self):
        """The spline order must be in range 1-5 (it is checked before the beam model data are used)."""
        for splineOrder in (0, 6):
            with self.subTest(splineOrder=splineOrder):
                with self.assertRaises(ValueError):
                    ft.interpolateBeamModel(pd.DataFrame(), 100, interpolation="spline", splineOrder=splineOrder)


if __name__ == '__main__':
    unittest.main()
