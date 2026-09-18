import unittest
import os
from pathlib import Path
import fredtools as ft
from fredtools._typing import *
import shutil
import pydicom as dicom

testPath = Path(os.path.dirname(__file__))


class test_sortDicoms(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'

    def test_sortDicoms_recursive(self):
        dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True, displayInfo=True)
        self.assertIn('CTfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['CTfileNames']), 240)
        self.assertIn('RSfileNames', dicomFiles)
        self.assertIsInstance(dicomFiles['RSfileNames'], str)
        self.assertIn('RNfileNames', dicomFiles)
        self.assertIsInstance(dicomFiles['RNfileNames'], str)
        self.assertIn('RDfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['RDfileNames']), 3)
        self.assertIn('PETfileNames', dicomFiles)
        # self.assertIsInstance(dicomFiles['PETfileNames'], str)
        self.assertIn('Unknown', dicomFiles)

    def test_sortDicoms_non_recursive(self):
        dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=False, displayInfo=True)
        self.assertIn('CTfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['CTfileNames']), 0)
        self.assertIn('RSfileNames', dicomFiles)
        self.assertIsInstance(dicomFiles['RSfileNames'], str)
        self.assertIn('RNfileNames', dicomFiles)
        self.assertIsInstance(dicomFiles['RNfileNames'], str)
        self.assertIn('RDfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['RDfileNames']), 3)
        self.assertIn('PETfileNames', dicomFiles)
        # self.assertIsInstance(dicomFiles['PETfileNames'], str)
        self.assertIn('Unknown', dicomFiles)

    def test_sortDicoms_no_dicoms(self):
        dicomFiles = ft.sortDicoms("unittests/testData/TPSDicoms", recursive=False)
        self.assertIn('CTfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['CTfileNames']), 0)
        self.assertIn('RSfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['RSfileNames']), 0)
        self.assertIn('RNfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['RNfileNames']), 0)
        self.assertIn('RDfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['RDfileNames']), 0)
        self.assertIn('PETfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['PETfileNames']), 0)
        self.assertIn('Unknown', dicomFiles)


class test_getDicomTypeName(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getDicomTypeName(self):
        self.assertEqual(ft.getDicomTypeName(self.dicomFiles.CTfileNames[0]), "CT Image Storage")
        self.assertEqual(ft.getDicomTypeName(self.dicomFiles.RSfileNames), "RT Structure Set Storage")
        self.assertEqual(ft.getDicomTypeName(self.dicomFiles.RNfileNames), "RT Ion Plan Storage")
        self.assertEqual(ft.getDicomTypeName(self.dicomFiles.RDfileNames[0]), "RT Dose Storage")
        # self.assertEqual(getDicomTypeName(self.dicomFiles.pet_file), "Positron Emission Tomography Image Storage")
        with self.assertRaises(TypeError):
            ft.getDicomTypeName(1)  # type: ignore


class test_isDicomCT(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomCT(self):
        self.assertTrue(ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.CTfileNames[0]))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.RSfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.RNfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.RDfileNames[0]))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.RSfileNames, raiseError=True)


class test_isDicomRS(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomRS(self):
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.CTfileNames[0]))
        self.assertTrue(ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.RSfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.RNfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.RDfileNames[0]))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.RNfileNames, raiseError=True)


class test_isDicomRN(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomRN(self):
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.CTfileNames[0]))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.RSfileNames))
        self.assertTrue(ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.RNfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.RDfileNames[0]))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.RSfileNames, raiseError=True)


class test_isDicomRD(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomRD(self):
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.CTfileNames[0]))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.RSfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.RNfileNames))
        self.assertTrue(ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.RDfileNames[0]))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.RSfileNames, raiseError=True)


@unittest.skip("TODO: generate a dicom file with a PET image")
class test_isDicomPET(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomPET(self):
        self.assertFalse(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.CTfileNames[0]))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.RSfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.RNfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.RDfileNames[0]))
        self.assertTrue(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.PETfileNames))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.RSfileNames, raiseError=True)


class test_getRNMachineName(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNMachineName(self):
        machine_name = ft.getRNMachineName(self.dicomFiles.RNfileNames)
        self.assertIsInstance(machine_name, str)
        self.assertEqual(machine_name, 'GTR3')


class test_getRNIsocenter(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNIsocenter(self):
        isocenter = ft.getRNIsocenter(self.dicomFiles.RNfileNames)
        self.assertIsInstance(isocenter, tuple)
        self.assertEqual(len(isocenter), 3)
        self.assertEqual(isocenter, (38.64990791296525, -239.956199552955, -569.809756326375))


class test_getRNSpots(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNSpots(self):
        spotsInfo = ft.getRNSpots(self.dicomFiles.RNfileNames)
        self.assertFalse(spotsInfo.empty)
        self.assertEqual(spotsInfo.loc[spotsInfo.PBMU != 0].shape, (5726, 21))
        self.assertEqual(spotsInfo.PBMU.sum(), 480.3859992669222)
        self.assertEqual(spotsInfo[["PBPosX", "PBPosY"]].mean().tolist(), [1.7813482361159623, 0.32963674467341947])


class test_getRNFields(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNFields(self):
        fieldsInfo = ft.getRNFields(self.dicomFiles.RNfileNames)
        self.assertFalse(fieldsInfo.empty)
        self.assertEqual(fieldsInfo.shape, (3, 18))
        self.assertEqual(fieldsInfo.loc[fieldsInfo.FNo == 1].FName.tolist(), ['Field 1'])
        self.assertAlmostEqual(fieldsInfo.FMU.sum(), 480.3859992669222, places=5)


class test_getRNInfo(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNInfo(self):
        planInfo = ft.getRNInfo(self.dicomFiles.RNfileNames)
        self.assertIsInstance(planInfo, dict)
        self.assertEqual(planInfo.fractionNo, 10)
        self.assertAlmostEqual(planInfo.dosePrescribed, 20.0)
        self.assertEqual(planInfo.targetStructName, '')
        self.assertEqual(planInfo.planLabel, 'Sphere_3F_FTR')
        self.assertEqual(planInfo.planDate, '20250206')
        self.assertEqual(planInfo.planTime, '174427.977')
        self.assertEqual(planInfo.patientName, 'BbdWxIAgkKdXCJMh')
        self.assertEqual(planInfo.patientBirthDate, '')
        self.assertEqual(planInfo.patientID, 'EIizsQEwa7JwkpT3omoijBlQR')
        self.assertEqual(planInfo.manufacturer, 'Varian Medical Systems')
        self.assertEqual(planInfo.softwareVersions, '16.1.3')
        self.assertEqual(planInfo.stationName, '')
        self.assertEqual(planInfo.machineName, 'GTR3')
        self.assertEqual(planInfo.totalFieldsNumber, 3)
        self.assertEqual(planInfo.treatmentFieldsNumber, 3)
        self.assertEqual(planInfo.setupFieldsNumber, 0)
        self.assertEqual(planInfo.otherFieldsNumber, 0)


class test_getRSInfo(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRSInfo(self):
        rsInfo = ft.getRSInfo(self.dicomFiles.RSfileNames)
        self.assertFalse(rsInfo.empty)
        self.assertEqual(rsInfo.groupby("ROIType")["ROIType"].count().index.to_list(), ['AVOIDANCE', 'CONTROL', 'EXTERNAL', 'ORGAN', 'PTV'])
        self.assertEqual(rsInfo.groupby("ROIType")["ROIType"].count().tolist(), [1, 11, 1, 1, 4])


class test_getExternalName(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getExternalName(self):
        externalName = ft.getExternalName(self.dicomFiles.RSfileNames)
        self.assertIsInstance(externalName, str)
        self.assertEqual(externalName, 'External')


class test_getCT(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getCT(self):
        imgCT = ft.getCT(self.dicomFiles.CTfileNames, displayInfo=True)
        self.assertIsInstance(imgCT, SITKImage)
        self.assertEqual(imgCT.GetSize(), (512, 512, 240))
        self.assertListEqual(np.round(imgCT.GetSpacing(), decimals=5).tolist(), [0.97656, 0.97656, 1.2])
        self.assertListEqual(np.round(imgCT.GetOrigin(), decimals=5).tolist(), [-249.51172, -470.51172, -718.3])
        self.assertAlmostEqual(ft.getStatistics(imgCT).GetMean(), -893.670319255193)


class test_getRD(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRD(self):
        imgDose = ft.sumImg(ft.getRD(self.dicomFiles.RDfileNames, displayInfo=True))
        self.assertIsNotNone(imgDose)
        self.assertEqual(imgDose.GetSize(), (233, 289, 288))
        self.assertListEqual(np.round(imgDose.GetSpacing(), decimals=5).tolist(), [1.0, 1.0, 1.0])
        self.assertListEqual(np.round(imgDose.GetOrigin(), decimals=5).tolist(), [-114.734, -357.219, -718.4])
        self.assertAlmostEqual(ft.getStatistics(imgDose).GetMean(), 0.5648171966439947)


class test_getRDFileNameForFieldNumber(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRDFileNameForFieldNumber(self):
        RDfileNameField1 = ft.getRDFileNameForFieldNumber(self.dicomFiles.RDfileNames, 1, displayInfo=True)
        self.assertIsNotNone(RDfileNameField1)
        self.assertIn("Field 1", str(RDfileNameField1))

    def test_getRDFileNameForFieldNumber_non_existing_field(self):
        RDfileNameField3 = ft.getRDFileNameForFieldNumber(self.dicomFiles.RDfileNames, 10, displayInfo=True)
        self.assertIsNone(RDfileNameField3)


class test_anonymizeDicoms(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.test_dir = Path.joinpath(testPath, "anonymizeDataFolder")
        self.test_dir.mkdir(exist_ok=True)
        # Copy all files recursively from testDataFolder to test_dir
        shutil.copytree(self.testDataFolder, self.test_dir, dirs_exist_ok=True)
        self.dicomFiles = ft.sortDicoms(self.test_dir, recursive=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_anonymizeDicoms(self):
        dicomFilesFlatten = [v for _, values in self.dicomFiles.items() for v in (values if isinstance(values, list) else [values])]
        ft.anonymizeDicoms(dicomFilesFlatten, removePrivateTags=True, displayInfo=True)

        for dicomFile in dicomFilesFlatten:
            dicomTags = dicom.dcmread(dicomFile)
            if "PatientName" in dicomTags:
                self.assertEqual(dicomTags.PatientName, "")
            if "PatientBirthDate" in dicomTags:
                self.assertEqual(dicomTags.PatientBirthDate, "")
            if "PatientBirthTime" in dicomTags:
                self.assertEqual(dicomTags.PatientBirthTime, "")
            if "PatientSex" in dicomTags:
                self.assertEqual(dicomTags.PatientSex, "")
            if "ReferringPhysicianName" in dicomTags:
                self.assertEqual(dicomTags.ReferringPhysicianName, "")
            if "ReviewerName" in dicomTags:
                self.assertEqual(dicomTags.ReviewerName, "")
            if "ReviewDate" in dicomTags:
                self.assertEqual(dicomTags.ReviewDate, "")
            if "ReviewTime" in dicomTags:
                self.assertEqual(dicomTags.ReviewTime, "")
            if "OperatorsName" in dicomTags:
                self.assertEqual(dicomTags.OperatorsName, "")
            if "PhysiciansOfRecord" in dicomTags:
                self.assertEqual(dicomTags.PhysiciansOfRecord, "")

    def test_anonymizeDicoms_returns_written_file_names(self):
        writtenFileNames = ft.anonymizeDicoms(self.dicomFiles.RDfileNames)
        self.assertEqual(writtenFileNames, list(self.dicomFiles.RDfileNames))

    def test_anonymizeDicoms_destination_folder(self):
        destinationFolder = Path.joinpath(testPath, "anonymizeDestinationFolder")
        writtenFileNames = ft.anonymizeDicoms(self.dicomFiles.RDfileNames, destination=destinationFolder)
        try:
            self.assertEqual(len(writtenFileNames), len(self.dicomFiles.RDfileNames))
            for writtenFileName, sourceFileName in zip(writtenFileNames, self.dicomFiles.RDfileNames):
                self.assertTrue(os.path.isfile(writtenFileName))
                self.assertEqual(os.path.basename(writtenFileName), os.path.basename(sourceFileName))
                self.assertEqual(dicom.dcmread(writtenFileName).PatientName, "")
                # the source dicoms must not be touched
                self.assertNotEqual(dicom.dcmread(sourceFileName).PatientName, "")
        finally:
            shutil.rmtree(destinationFolder, ignore_errors=True)

    def test_anonymizeDicoms_destination_list(self):
        destinationFolder = Path.joinpath(testPath, "anonymizeDestinationFolder")
        destinationFileNames = [str(Path.joinpath(destinationFolder, f"RD.{idx}.dcm")) for idx in range(len(self.dicomFiles.RDfileNames))]
        try:
            writtenFileNames = ft.anonymizeDicoms(self.dicomFiles.RDfileNames, destination=destinationFileNames)
            self.assertEqual(writtenFileNames, destinationFileNames)
            for destinationFileName in destinationFileNames:
                self.assertTrue(os.path.isfile(destinationFileName))
        finally:
            shutil.rmtree(destinationFolder, ignore_errors=True)

    def test_anonymizeDicoms_destination_length_mismatch(self):
        with self.assertRaises(ValueError):
            ft.anonymizeDicoms(self.dicomFiles.RDfileNames, destination=["onlyOneDestination.dcm"])

    def test_anonymizeDicoms_patientName_and_patientID(self):
        ft.anonymizeDicoms(self.dicomFiles.RDfileNames, patientName="12345", patientID="12345")
        for dicomFile in self.dicomFiles.RDfileNames:
            dicomTags = dicom.dcmread(dicomFile)
            self.assertEqual(dicomTags.PatientName, "12345")
            self.assertEqual(dicomTags.PatientID, "12345")

    def test_anonymizeDicoms_tags(self):
        ft.anonymizeDicoms(self.dicomFiles.RDfileNames, tags={"InstitutionName": "anonymous"})
        for dicomFile in self.dicomFiles.RDfileNames:
            self.assertEqual(dicom.dcmread(dicomFile).InstitutionName, "anonymous")

    def test_anonymizeDicoms_reads_and_writes_every_dicom_only_once(self):
        from unittest import mock
        destinationFolder = Path.joinpath(testPath, "anonymizeDestinationFolder")
        try:
            with mock.patch("pydicom.dcmread", wraps=dicom.dcmread) as dcmreadMock, \
                 mock.patch("pydicom.dcmwrite", wraps=dicom.dcmwrite) as dcmwriteMock:
                ft.anonymizeDicoms(self.dicomFiles.RDfileNames, destination=destinationFolder, patientID="12345")
            self.assertEqual(dcmreadMock.call_count, len(self.dicomFiles.RDfileNames))
            self.assertEqual(dcmwriteMock.call_count, len(self.dicomFiles.RDfileNames))
        finally:
            shutil.rmtree(destinationFolder, ignore_errors=True)


class test_checkDicomsUIDwithIndex(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)
        self.dicomsInfo = ft.getDicomsInfo(self.testDataFolder, recursive=True)

    def test_checkDicomsUID_withIndex_equals_without(self):
        checkResults = ft.checkDicomsUID(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames, self.dicomFiles.RDfileNames)
        checkResultsIndexed = ft.checkDicomsUID(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames, self.dicomFiles.RDfileNames,
                                                dicomsInfo=self.dicomsInfo)
        self.assertEqual(dict(checkResultsIndexed), dict(checkResults))

    def test_checkDicomsUID_withIndex_does_not_read_any_dicom(self):
        from unittest import mock
        with mock.patch("pydicom.dcmread", wraps=dicom.dcmread) as dcmreadMock:
            ft.checkDicomsUID(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames, self.dicomFiles.RDfileNames,
                              dicomsInfo=self.dicomsInfo)
        self.assertEqual(dcmreadMock.call_count, 0)

    def test_checkDicomsUID_withIndex_no_RD(self):
        checkResults = ft.checkDicomsUID(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames, dicomsInfo=self.dicomsInfo)
        self.assertIsNone(checkResults.UIDRNtoRD)
        self.assertIsNone(checkResults.RDbeamNumbers)
        self.assertTrue(checkResults.UIDRNtoRS)
        self.assertTrue(checkResults.UIDRStoCT)
        self.assertTrue(checkResults.UIDFoR)

    def test_checkDicomsUID_withIndex_missing_file(self):
        dicomsInfo = self.dicomsInfo[self.dicomsInfo.dicomType != "CT"]
        with self.assertRaises(ValueError):
            ft.checkDicomsUID(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames, dicomsInfo=dicomsInfo)

    def test_checkDicomsUID_withIndex_without_FrameOfReferenceUID(self):
        dicomsInfo = ft.getDicomsInfo(self.testDataFolder, recursive=True, readFrameOfReferenceUID=False)
        checkResults = ft.checkDicomsUID(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames, dicomsInfo=dicomsInfo)
        self.assertIsNone(checkResults.UIDFoR)


class test_isDicomTypeCheckers(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicom_raiseError_reads_the_dicom_only_once(self):
        """The failing type check must not read the dicom a second time to build the error message."""
        from unittest import mock
        from fredtools.ImgIO import dicom_io
        with mock.patch("pydicom.dcmread", wraps=dicom.dcmread) as dcmreadMock:
            with self.assertRaises(TypeError):
                dicom_io._isDicomCT(self.dicomFiles.RSfileNames, raiseError=True)
        self.assertEqual(dcmreadMock.call_count, 1)


class test_getStructureContoursByName(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)
        # print(ft.getRSInfo(self.dicomFiles.RSfileNames).to_string())

    def test_getStructureContoursByName_valid_structure(self):
        structContours, structInfo = ft.ImgIO.dicom_io._getStructureContoursByName(self.dicomFiles.RSfileNames, 'PTV_cubic')
        self.assertIsInstance(structContours, list)
        self.assertTrue(all(isinstance(contour, np.ndarray) for contour in structContours))
        self.assertIsInstance(structInfo, dict)
        self.assertEqual(structInfo['Name'], 'PTV_cubic')
        self.assertIn('Number', structInfo)
        self.assertIn('Type', structInfo)
        self.assertIn('Color', structInfo)

    def test_getStructureContoursByName_invalid_structure(self):
        with self.assertRaises(ValueError):
            ft.ImgIO.dicom_io._getStructureContoursByName(self.dicomFiles.RSfileNames, 'InvalidStructure')

    def test_getStructureContoursByName_no_contour_sequence(self):
        contours, info = ft.ImgIO.dicom_io._getStructureContoursByName(self.dicomFiles.RSfileNames, 'NoContour')
        self.assertIsInstance(contours, list)
        self.assertEqual(len(contours), 0)
        self.assertIsInstance(info, dict)
        self.assertEqual(info['Name'], 'NoContour')
        self.assertIn('Number', info)
        self.assertIn('Type', info)
        self.assertIn('Color', info)

    def test_getStructureContoursByName_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            ft.ImgIO.dicom_io._getStructureContoursByName('invalid_file.dcm', 'PTV_cubic')
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._getStructureContoursByName(self.dicomFiles.RNfileNames, 'PTV_cubic')


class test_checkDicomsUID(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms/TPSPlan'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_checkDicomsUID_with_RD(self):
        checkResults = ft.checkDicomsUID(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames, self.dicomFiles.RDfileNames, displayInfo=True)
        self.assertTrue(checkResults.UIDRNtoRS)
        self.assertTrue(checkResults.UIDRStoCT)
        self.assertTrue(checkResults.UIDRNtoRD)
        self.assertTrue(checkResults.UIDFoR)
        self.assertTrue(checkResults.RDbeamNumbers)

    def test_checkDicomsUID_without_RD(self):
        for RDfileNames in [None, []]:
            checkResults = ft.checkDicomsUID(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames, RDfileNames, displayInfo=True)
            self.assertTrue(checkResults.UIDRNtoRS)
            self.assertTrue(checkResults.UIDRStoCT)
            self.assertIsNone(checkResults.UIDRNtoRD)
            self.assertTrue(checkResults.UIDFoR)
            self.assertIsNone(checkResults.RDbeamNumbers)

    def test_checkDicomsUID_empty_CT(self):
        with self.assertRaises(ValueError):
            ft.checkDicomsUID(self.dicomFiles.RNfileNames, self.dicomFiles.RSfileNames, [])

    def test_checkDicomsUID_nonexistent_RN(self):
        with self.assertRaises(FileNotFoundError):
            ft.checkDicomsUID('invalid_file.dcm', self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames)

    def test_checkDicomsUID_RN_not_single_file(self):
        with self.assertRaises(ValueError):
            ft.checkDicomsUID([self.dicomFiles.RNfileNames], self.dicomFiles.RSfileNames, self.dicomFiles.CTfileNames)  # type: ignore


if __name__ == '__main__':
    unittest.main()
