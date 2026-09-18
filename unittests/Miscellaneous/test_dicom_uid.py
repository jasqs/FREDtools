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


class test_getDicomsInfo(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)
        self.dicomsInfo = ft.getDicomsInfo(self.testDataFolder, recursive=True)

    def test_getDicomsInfo_shape(self):
        self.assertEqual(len(self.dicomsInfo), 245)
        self.assertEqual(self.dicomsInfo.dicomType.value_counts().to_dict(), {'CT': 240, 'RD': 3, 'RS': 1, 'RN': 1})
        for columnName in ['fileName', 'folderName', 'SOPClassUID', 'dicomTypeName', 'dicomType', 'SOPInstanceUID',
                           'FrameOfReferenceUID', 'referencedSOPInstanceUIDs', 'referencedSeriesNo', 'beamNumbers',
                           'referencedBeamNumber', 'doseSummationType', 'structureSetDate', 'structureSetTime',
                           'SeriesInstanceUID', 'StudyInstanceUID', 'error']:
            self.assertIn(columnName, self.dicomsInfo.columns)

    def test_getDicomsInfo_single_file(self):
        dicomsInfo = ft.getDicomsInfo(self.dicomFiles.RNfileNames)
        self.assertEqual(len(dicomsInfo), 1)
        self.assertEqual(dicomsInfo.dicomType.iloc[0], 'RN')

    def test_getDicomsInfo_non_recursive(self):
        dicomsInfo = ft.getDicomsInfo(self.testDataFolder, recursive=False)
        self.assertEqual((dicomsInfo.dicomType == 'CT').sum(), 0)
        self.assertEqual((dicomsInfo.dicomType == 'RD').sum(), 3)

    def test_getDicomsInfo_SOPInstanceUID_matches_getSOPInstanceUID(self):
        CTfileNames = self.dicomsInfo.fileName[self.dicomsInfo.dicomType == 'CT'].tolist()
        SOPInstanceUIDs = [str(SOPInstanceUID) for SOPInstanceUID in ft.getSOPInstanceUID(CTfileNames)]
        self.assertEqual(self.dicomsInfo.SOPInstanceUID[self.dicomsInfo.dicomType == 'CT'].tolist(), SOPInstanceUIDs)

    def test_getDicomsInfo_FrameOfReferenceUID_matches_getFrameOfReferenceUID(self):
        fileNames = self.dicomsInfo.fileName.tolist()
        FrameOfReferenceUIDs = [str(FrameOfReferenceUID) for FrameOfReferenceUID in ft.getFrameOfReferenceUID(fileNames)]
        self.assertEqual(self.dicomsInfo.FrameOfReferenceUID.tolist(), FrameOfReferenceUIDs)

    def test_getDicomsInfo_references_match_the_getters(self):
        RSrow = self.dicomsInfo[self.dicomsInfo.dicomType == 'RS'].iloc[0]
        self.assertEqual(list(RSrow.referencedSOPInstanceUIDs),
                         [str(UID) for UID in ft.getRSReferencedImageUIDs(RSrow.fileName)])

        RNrow = self.dicomsInfo[self.dicomsInfo.dicomType == 'RN'].iloc[0]
        self.assertEqual(RNrow.referencedSOPInstanceUIDs[0], str(ft.getRNReferencedStructureSetUID(RNrow.fileName)))
        self.assertEqual(RNrow.beamNumbers, (1, 2, 3))

        for RDrow in self.dicomsInfo[self.dicomsInfo.dicomType == 'RD'].itertuples():
            self.assertEqual(RDrow.referencedSOPInstanceUIDs[0], str(ft.getRDReferencedPlanUID(RDrow.fileName)))
            self.assertIn(RDrow.referencedBeamNumber, RNrow.beamNumbers)

    def test_getDicomsInfo_readFrameOfReferenceUID_False(self):
        """The file meta information must give the same identity as the dataset (DICOM PS3.10 7.1)."""
        dicomsInfo = ft.getDicomsInfo(self.testDataFolder, recursive=True, readReferences=False, readFrameOfReferenceUID=False)
        self.assertEqual(dicomsInfo.SOPClassUID.tolist(), self.dicomsInfo.SOPClassUID.tolist())
        self.assertEqual(dicomsInfo.SOPInstanceUID.tolist(), self.dicomsInfo.SOPInstanceUID.tolist())
        self.assertEqual(dicomsInfo.dicomType.tolist(), self.dicomsInfo.dicomType.tolist())
        self.assertTrue(dicomsInfo.FrameOfReferenceUID.isna().all())

    def test_getDicomsInfo_readReferences_False(self):
        dicomsInfo = ft.getDicomsInfo(self.testDataFolder, recursive=True, readReferences=False)
        self.assertTrue(all(len(referenced) == 0 for referenced in dicomsInfo.referencedSOPInstanceUIDs))

    def test_getDicomsInfo_reads_every_dicom_only_once(self):
        """The dicoms must not be read more than once, which is the whole point of the function."""
        from unittest import mock
        with mock.patch('pydicom.dcmread', wraps=dicom.dcmread) as dcmreadMock:
            ft.getDicomsInfo(self.testDataFolder, recursive=True)
        self.assertEqual(dcmreadMock.call_count, 245 + 5)  # one read per dicom plus one per RS/RN/RD reference read

    def test_getDicomsInfo_reads_only_the_file_meta_when_possible(self):
        from unittest import mock
        with mock.patch('pydicom.dcmread', wraps=dicom.dcmread) as dcmreadMock:
            ft.getDicomsInfo(self.testDataFolder, recursive=True, readReferences=False, readFrameOfReferenceUID=False)
        self.assertEqual(dcmreadMock.call_count, 0)

    def test_getDicomsInfo_unreadable_file(self):
        """A file which is not a readable dicom must be reported, not raised for."""
        import tempfile
        with tempfile.TemporaryDirectory() as unreadableFolder:
            Path(unreadableFolder).joinpath('notADicom.dcm').write_text('this is not a dicom')
            dicomsInfo = ft.getDicomsInfo(unreadableFolder, recursive=True)
            self.assertEqual(len(dicomsInfo), 1)
            self.assertEqual(dicomsInfo.dicomType.iloc[0], 'Unknown')
            self.assertIsNotNone(dicomsInfo.error.iloc[0])

    def test_getDicomsInfo_no_dicoms(self):
        dicomsInfo = ft.getDicomsInfo('unittests/testData/TPSDicoms', recursive=False)
        self.assertEqual(len(dicomsInfo), 0)


class test_sortDicomsFromInfo(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'

    def _flattenDicomFiles(self, dicomFiles):
        return [fileName for fileNames in dicomFiles.values() for fileName in (fileNames if isinstance(fileNames, list) else [fileNames])]

    def test_sortDicomsFromInfo_holds_every_dicom_exactly_once(self):
        for recursive in [True, False]:
            dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=recursive)
            foundFileNames = [str(fileName) for fileName in (Path(self.testDataFolder).rglob('*.dcm') if recursive else Path(self.testDataFolder).glob('*.dcm'))]
            self.assertEqual(sorted(self._flattenDicomFiles(dicomFiles)), sorted(foundFileNames))

    def test_sortDicomsFromInfo_preserves_the_order_the_dicoms_were_found(self):
        """The file names must keep the order of the folder walk, as the callers rely on the slice order."""
        dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)
        foundFileNames = [str(fileName) for fileName in Path(self.testDataFolder).rglob('*.dcm')]
        for dicomType in ['CTfileNames', 'RDfileNames']:
            fileNames = dicomFiles[dicomType]
            positions = [foundFileNames.index(fileName) for fileName in (fileNames if isinstance(fileNames, list) else [fileNames])]
            self.assertEqual(positions, sorted(positions), msg=dicomType)

    def test_sortDicomsFromInfo_collapseSingle(self):
        dicomsInfo = ft.getDicomsInfo(self.testDataFolder, recursive=True, readReferences=False, readFrameOfReferenceUID=False)
        dicomFiles = ft.sortDicomsFromInfo(dicomsInfo, collapseSingle=True)
        self.assertIsInstance(dicomFiles.RSfileNames, str)
        dicomFiles = ft.sortDicomsFromInfo(dicomsInfo, collapseSingle=False)
        self.assertIsInstance(dicomFiles.RSfileNames, list)
        self.assertEqual(len(dicomFiles.RSfileNames), 1)


class test_matchDicomsByUID(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomsInfo = ft.getDicomsInfo(self.testDataFolder, recursive=True)

    def test_matchDicomsByUID_matching(self):
        matchResults = ft.matchDicomsByUID(self.dicomsInfo)
        self.assertEqual(len(matchResults), 1)
        matchResult = matchResults.iloc[0]
        self.assertEqual(len(matchResult.CTfileNames), 240)
        self.assertEqual(len(matchResult.RDfileNames), 3)
        for checkName in ['UIDRNtoRS', 'UIDRStoCT', 'UIDRNtoRD', 'UIDFoR', 'RDbeamNumbers']:
            self.assertTrue(matchResult[checkName], msg=checkName)

    def test_matchDicomsByUID_equals_checkDicomsUID(self):
        dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)
        checkResults = ft.checkDicomsUID(dicomFiles.RNfileNames, dicomFiles.RSfileNames, dicomFiles.CTfileNames, dicomFiles.RDfileNames)
        matchResult = ft.matchDicomsByUID(self.dicomsInfo).iloc[0]
        for checkName in ['UIDRNtoRS', 'UIDRStoCT', 'UIDRNtoRD', 'UIDFoR', 'RDbeamNumbers']:
            self.assertEqual(matchResult[checkName], checkResults[checkName], msg=checkName)

    def test_matchDicomsByUID_does_not_read_any_dicom(self):
        from unittest import mock
        with mock.patch('pydicom.dcmread', wraps=dicom.dcmread) as dcmreadMock:
            ft.matchDicomsByUID(self.dicomsInfo)
        self.assertEqual(dcmreadMock.call_count, 0)

    def test_matchDicomsByUID_no_RS_in_index(self):
        dicomsInfo = self.dicomsInfo[self.dicomsInfo.dicomType != 'RS']
        matchResult = ft.matchDicomsByUID(dicomsInfo).iloc[0]
        self.assertFalse(matchResult.UIDRNtoRS)
        self.assertIsNone(matchResult.RSfileName)

    def test_matchDicomsByUID_empty_index(self):
        matchResults = ft.matchDicomsByUID(self.dicomsInfo[self.dicomsInfo.dicomType == 'CT'])
        self.assertEqual(len(matchResults), 0)


class test_dicomVarInput(unittest.TestCase):
    """The UID getters must accept dicom tags already read, without reading the file again."""

    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getSOPInstanceUID_dataset(self):
        dicomTags = dicom.dcmread(self.dicomFiles.RSfileNames)
        self.assertEqual(ft.getSOPInstanceUID(dicomTags), ft.getSOPInstanceUID(self.dicomFiles.RSfileNames))

    def test_getSOPInstanceUID_dataset_is_not_iterated(self):
        """A pydicom Dataset is iterable, so it must not be mistaken for an iterable of dicoms."""
        dicomTags = dicom.dcmread(self.dicomFiles.RSfileNames)
        self.assertIsInstance(ft.getSOPInstanceUID(dicomTags), dicom.uid.UID)

    def test_getSOPInstanceUID_dataset_list(self):
        dicomTagsList = [dicom.dcmread(fileName) for fileName in self.dicomFiles.RDfileNames]
        self.assertEqual(ft.getSOPInstanceUID(dicomTagsList), ft.getSOPInstanceUID(self.dicomFiles.RDfileNames))

    def test_getFrameOfReferenceUID_dataset(self):
        dicomTags = dicom.dcmread(self.dicomFiles.RNfileNames)
        self.assertEqual(ft.getFrameOfReferenceUID(dicomTags), ft.getFrameOfReferenceUID(self.dicomFiles.RNfileNames))

    def test_getRNReferencedStructureSetUID_dataset(self):
        dicomTags = dicom.dcmread(self.dicomFiles.RNfileNames)
        self.assertEqual(ft.getRNReferencedStructureSetUID(dicomTags), ft.getRNReferencedStructureSetUID(self.dicomFiles.RNfileNames))

    def test_getRSReferencedImageUIDs_dataset(self):
        dicomTags = dicom.dcmread(self.dicomFiles.RSfileNames)
        self.assertEqual(ft.getRSReferencedImageUIDs(dicomTags), ft.getRSReferencedImageUIDs(self.dicomFiles.RSfileNames))

    def test_getRDReferencedPlanUID_dataset(self):
        dicomTags = dicom.dcmread(self.dicomFiles.RDfileNames[0])
        self.assertEqual(ft.getRDReferencedPlanUID(dicomTags), ft.getRDReferencedPlanUID(self.dicomFiles.RDfileNames[0]))

    def test_getRNReferencedStructureSetUID_dataset_not_RN(self):
        dicomTags = dicom.dcmread(self.dicomFiles.RSfileNames)
        with self.assertRaises(TypeError):
            ft.getRNReferencedStructureSetUID(dicomTags)

    def test_checkUID_RStoCT_reads_every_dicom_only_once(self):
        from unittest import mock
        with mock.patch('pydicom.dcmread', wraps=dicom.dcmread) as dcmreadMock:
            ft.checkUID_RStoCT(self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames)
        self.assertEqual(dcmreadMock.call_count, 1 + len(self.dicomFiles.CTfileNames))

    def test_checkUID_RNtoRS_reads_every_dicom_only_once(self):
        from unittest import mock
        with mock.patch('pydicom.dcmread', wraps=dicom.dcmread) as dcmreadMock:
            ft.checkUID_RNtoRS(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames)
        self.assertEqual(dcmreadMock.call_count, 2)

    def test_checkUID_RNtoRD_reads_every_dicom_only_once(self):
        from unittest import mock
        with mock.patch('pydicom.dcmread', wraps=dicom.dcmread) as dcmreadMock:
            ft.checkUID_RNtoRD(self.dicomFiles.RNfileNames, self.dicomFiles.RDfileNames)
        self.assertEqual(dcmreadMock.call_count, 1 + len(self.dicomFiles.RDfileNames))


if __name__ == '__main__':
    unittest.main()
