import unittest
import os
from pathlib import Path
import fredtools as ft
from fredtools._typing import *
import pydicom as dicom

testPath = Path(os.path.dirname(__file__))


class test_getSOPInstanceUID(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getSOPInstanceUID_single_file(self):
        SOPInstanceUID = ft.Miscellaneous.dicom_uid.getSOPInstanceUID(self.dicomFiles.RSfileNames, displayInfo=True)
        self.assertIsInstance(SOPInstanceUID, dicom.uid.UID)
        self.assertTrue(SOPInstanceUID.is_valid)

    def test_getSOPInstanceUID_multiple_files(self):
        SOPInstanceUIDs = ft.Miscellaneous.dicom_uid.getSOPInstanceUID(self.dicomFiles.CTfileNames, displayInfo=True)
        self.assertIsInstance(SOPInstanceUIDs, list)
        self.assertEqual(len(SOPInstanceUIDs), 240)
        self.assertEqual(len(set(SOPInstanceUIDs)), 240)

    def test_getSOPInstanceUID_single_element_list(self):
        SOPInstanceUIDs = ft.Miscellaneous.dicom_uid.getSOPInstanceUID([self.dicomFiles.CTfileNames[0]])
        self.assertIsInstance(SOPInstanceUIDs, list)
        self.assertEqual(len(SOPInstanceUIDs), 1)

    def test_getSOPInstanceUID_empty_list(self):
        SOPInstanceUIDs = ft.Miscellaneous.dicom_uid.getSOPInstanceUID([])
        self.assertIsInstance(SOPInstanceUIDs, list)
        self.assertEqual(len(SOPInstanceUIDs), 0)


class test_getRNReferencedStructureSetUID(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNReferencedStructureSetUID(self):
        ReferencedStructureSetUID = ft.Miscellaneous.dicom_uid.getRNReferencedStructureSetUID(self.dicomFiles.RNfileNames, displayInfo=True)
        self.assertIsInstance(ReferencedStructureSetUID, dicom.uid.UID)
        self.assertEqual(ReferencedStructureSetUID, ft.Miscellaneous.dicom_uid.getSOPInstanceUID(self.dicomFiles.RSfileNames))

    def test_getRNReferencedStructureSetUID_not_RN(self):
        with self.assertRaises(TypeError):
            ft.Miscellaneous.dicom_uid.getRNReferencedStructureSetUID(self.dicomFiles.RSfileNames)


class test_getRSReferencedImageUIDs(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRSReferencedImageUIDs(self):
        ReferencedImageUIDs = ft.Miscellaneous.dicom_uid.getRSReferencedImageUIDs(self.dicomFiles.RSfileNames, displayInfo=True)
        self.assertIsInstance(ReferencedImageUIDs, list)
        self.assertEqual(sorted(ReferencedImageUIDs), sorted(ft.Miscellaneous.dicom_uid.getSOPInstanceUID(self.dicomFiles.CTfileNames)))

    def test_getRSReferencedImageUIDs_not_RS(self):
        with self.assertRaises(TypeError):
            ft.Miscellaneous.dicom_uid.getRSReferencedImageUIDs(self.dicomFiles.RNfileNames)


class test_getRDReferencedPlanUID(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRDReferencedPlanUID(self):
        ReferencedPlanUID = ft.Miscellaneous.dicom_uid.getRDReferencedPlanUID(self.dicomFiles.RDfileNames[0], displayInfo=True)
        self.assertIsInstance(ReferencedPlanUID, dicom.uid.UID)
        self.assertEqual(ReferencedPlanUID, ft.Miscellaneous.dicom_uid.getSOPInstanceUID(self.dicomFiles.RNfileNames))

    def test_getRDReferencedPlanUID_not_RD(self):
        with self.assertRaises(TypeError):
            ft.Miscellaneous.dicom_uid.getRDReferencedPlanUID(self.dicomFiles.RNfileNames)


class test_getFrameOfReferenceUID(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getFrameOfReferenceUID_single_file(self):
        FrameOfReferenceUID = ft.Miscellaneous.dicom_uid.getFrameOfReferenceUID(self.dicomFiles.RNfileNames, displayInfo=True)
        self.assertIsInstance(FrameOfReferenceUID, dicom.uid.UID)

    def test_getFrameOfReferenceUID_multiple_files(self):
        FrameOfReferenceUIDs = ft.Miscellaneous.dicom_uid.getFrameOfReferenceUID([self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames[0], self.dicomFiles.RDfileNames[0]])
        self.assertIsInstance(FrameOfReferenceUIDs, list)
        self.assertEqual(len(FrameOfReferenceUIDs), 4)
        self.assertEqual(len(set(FrameOfReferenceUIDs)), 1)


class test_checkUID_RNtoRS(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_checkUID_RNtoRS_matching(self):
        self.assertTrue(ft.Miscellaneous.dicom_uid.checkUID_RNtoRS(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames))

    def test_checkUID_RNtoRS_not_RN(self):
        with self.assertRaises(TypeError):
            ft.Miscellaneous.dicom_uid.checkUID_RNtoRS(self.dicomFiles.RSfileNames, self.dicomFiles.RSfileNames)

    def test_checkUID_RNtoRS_not_RS(self):
        with self.assertRaises(TypeError):
            ft.Miscellaneous.dicom_uid.checkUID_RNtoRS(self.dicomFiles.RNfileNames, self.dicomFiles.RNfileNames)


class test_checkUID_RStoCT(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_checkUID_RStoCT_matching(self):
        self.assertTrue(ft.Miscellaneous.dicom_uid.checkUID_RStoCT(self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames))

    def test_checkUID_RStoCT_single_CT_file_not_matching(self):
        self.assertFalse(ft.Miscellaneous.dicom_uid.checkUID_RStoCT(self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames[0]))

    def test_checkUID_RStoCT_not_RS(self):
        with self.assertRaises(TypeError):
            ft.Miscellaneous.dicom_uid.checkUID_RStoCT(self.dicomFiles.RNfileNames, self.dicomFiles.CTfileNames)

    def test_checkUID_RStoCT_not_CT(self):
        with self.assertRaises(TypeError):
            ft.Miscellaneous.dicom_uid.checkUID_RStoCT(self.dicomFiles.RSfileNames, self.dicomFiles.RNfileNames)

    def test_checkUID_RStoCT_duplicated_CT_warning(self):
        with self.assertLogs(ft.Miscellaneous.dicom_uid._logger, level='WARNING') as logsContext:
            self.assertFalse(ft.Miscellaneous.dicom_uid.checkUID_RStoCT(self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames + [self.dicomFiles.CTfileNames[0]]))
        self.assertTrue(any("duplicated SOPInstanceUIDs" in logMessage for logMessage in logsContext.output))


class test_checkUID_RNtoRD(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_checkUID_RNtoRD_matching(self):
        self.assertTrue(ft.Miscellaneous.dicom_uid.checkUID_RNtoRD(self.dicomFiles.RNfileNames, self.dicomFiles.RDfileNames))

    def test_checkUID_RNtoRD_single_RD_file(self):
        self.assertTrue(ft.Miscellaneous.dicom_uid.checkUID_RNtoRD(self.dicomFiles.RNfileNames, self.dicomFiles.RDfileNames[0]))

    def test_checkUID_RNtoRD_empty_RD_list(self):
        with self.assertRaises(ValueError):
            ft.Miscellaneous.dicom_uid.checkUID_RNtoRD(self.dicomFiles.RNfileNames, [])

    def test_checkUID_RNtoRD_not_RN(self):
        with self.assertRaises(TypeError):
            ft.Miscellaneous.dicom_uid.checkUID_RNtoRD(self.dicomFiles.RSfileNames, self.dicomFiles.RDfileNames)

    def test_checkUID_RNtoRD_not_RD(self):
        with self.assertRaises(TypeError):
            ft.Miscellaneous.dicom_uid.checkUID_RNtoRD(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames)


if __name__ == '__main__':
    unittest.main()
