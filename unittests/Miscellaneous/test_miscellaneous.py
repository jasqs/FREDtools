import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import os
import shutil
import fitz  # from pymupdf

import fredtools as ft

testPath = Path(os.path.dirname(__file__))


class test_mergePDF(unittest.TestCase):

    def setUp(self):
        self.testDir = Path("unittests/testData/PDFfiles")
        self.filesPath_pdf = list(self.testDir.glob("*.pdf"))
        self.filePaths_pdf_output = testPath.joinpath("merged_output.pdf")
        self.testDirCopy = testPath.joinpath("PDFfiles")
        if self.testDirCopy.exists():
            shutil.rmtree(self.testDirCopy)
        shutil.copytree(self.testDir, self.testDirCopy)
        self.pdf_files_copy = list(self.testDirCopy.glob("*.pdf"))

    def tearDown(self):
        if self.filePaths_pdf_output.exists():
            self.filePaths_pdf_output.unlink()
        if self.testDirCopy.exists():
            shutil.rmtree(self.testDirCopy)

    def test_mergePDF(self):
        ft.mergePDF(self.filesPath_pdf, self.filePaths_pdf_output, displayInfo=True)
        self.assertTrue(self.filePaths_pdf_output.exists())

    def test_mergePDF_with_nonexistent_file(self):
        non_existent_file = self.testDir.joinpath("non_existent.pdf")
        pdf_files_with_nonexistent = self.filesPath_pdf + [non_existent_file]

        with self.assertRaises(FileNotFoundError):
            ft.mergePDF(pdf_files_with_nonexistent, self.filePaths_pdf_output)

    def test_mergePDF_remove_source(self):
        ft.mergePDF(self.pdf_files_copy, self.filePaths_pdf_output, removeSource=True)
        self.assertTrue(self.filePaths_pdf_output.exists())
        # Check that the source files have been removed
        for pdf_file in self.pdf_files_copy:
            self.assertFalse(pdf_file.exists())

    def test_mergePDF_page_count(self):
        # Calculate the total number of pages in the source PDFs
        total_pages = sum(fitz.open(pdf).page_count for pdf in self.filesPath_pdf)
        ft.mergePDF(self.filesPath_pdf, self.filePaths_pdf_output)
        self.assertTrue(self.filePaths_pdf_output.exists())
        # Check the number of pages in the merged PDF
        merged_pdf = fitz.open(self.filePaths_pdf_output)
        self.assertEqual(merged_pdf.page_count, total_pages)

    def test_mergePDF_single_file(self):
        with self.assertRaises(TypeError):
            ft.mergePDF(self.filesPath_pdf[0], self.filePaths_pdf_output)  # type: ignore


class test_getHistogram(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'dataX': np.random.normal(size=1000),
                                'dataY': np.random.normal(size=1000)})
        self.bins = np.linspace(-3, 3, 50)

    def test_getHistogram_with_dataframe(self):
        # Test histogram without dataY
        bin_centers, hist_values = ft.getHistogram(self.df['dataX'], bins=self.bins)
        self.assertIsInstance(bin_centers, np.ndarray)
        self.assertIsInstance(hist_values, np.ndarray)
        self.assertEqual(len(bin_centers), len(self.bins) - 1)

        # Test differential histogram with dataY for all kinds
        for kind in ['sum', 'mean', 'std', 'median', 'mean-std', 'mean+std', 'min', 'max']:
            with self.subTest(kind=kind):
                bin_centers, hist_values = ft.getHistogram(self.df['dataX'], dataY=self.df['dataY'], bins=self.bins, kind=kind)
                self.assertIsInstance(bin_centers, np.ndarray)
                self.assertIsInstance(hist_values, np.ndarray)
                self.assertEqual(len(bin_centers), len(self.bins) - 1)

    def test_getHistogram_invalid_kind(self):
        with self.assertRaises(ValueError):
            ft.getHistogram(self.df['dataX'], dataY=self.df['dataY'], bins=self.bins, kind="invalid_kind")

    def test_getHistogram_missing_bins(self):
        bin_centers, hist_values = ft.getHistogram(self.df['dataX'], dataY=self.df['dataY'], bins=None)
        self.assertEqual(len(bin_centers), 100-1)


class test_sigma2FWHM(unittest.TestCase):

    def test_sigma2fwhm(self):
        sigma = 1.0
        fwhm = ft.sigma2fwhm(sigma)
        self.assertAlmostEqual(fwhm, 2.35482, places=5)


class test_FWHM2Sigma(unittest.TestCase):

    def test_fwhm2sigma(self):
        fwhm = 2.35482
        sigma = ft.fwhm2sigma(fwhm)
        self.assertAlmostEqual(sigma, 1.0, places=5)


class test_getLineFromFile(unittest.TestCase):

    def setUp(self):
        self.test_file = Path("test_file.txt")
        with open(self.test_file, "w") as f:
            f.write("line 1\nline 2\n# comment\nline 3\n")

    def tearDown(self):
        self.test_file.unlink()

    def test_getLineFromFile(self):
        result = ft.getLineFromFile(r"line", self.test_file, kind="all")
        self.assertEqual(result, ((1, 2, 4), ("line 1", "line 2", "line 3")))

        result = ft.getLineFromFile(r"line", self.test_file, kind="first")
        self.assertEqual(result, (1, "line 1"))

        result = ft.getLineFromFile(r"line", self.test_file, kind="last")
        self.assertEqual(result, (4, "line 3"))

    def test_getLineFromFile_with_nonexistent_file(self):
        non_existent_file = Path("non_existent.txt")
        with self.assertRaises(FileNotFoundError):
            ft.getLineFromFile(r"line", non_existent_file)

    def test_getLineFromFile_with_invalid_kind(self):
        with self.assertRaises(AttributeError):
            ft.getLineFromFile(r"line", self.test_file, kind="invalid_kind")  # type: ignore


class test_getCPUNo(unittest.TestCase):

    def test_getCPUNo(self):
        self.assertEqual(ft.getCPUNo("auto"), os.cpu_count())
        self.assertEqual(ft.getCPUNo(4), 4)
        with self.assertRaises(ValueError):
            ft.getCPUNo(-1)


if __name__ == '__main__':
    unittest.main()
