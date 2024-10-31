import unittest
import os
from pathlib import Path
import fredtools as ft
from fredtools._typing import *
import shutil
import pydicom as dicom

testPath = Path(os.path.dirname(__file__))


class test_SortDicoms(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'

    def test_sortDicoms_recursive(self):
        dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True, displayInfo=True)
        self.assertIn('CTfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['CTfileNames']), 240)
        self.assertIn('RSfileNames', dicomFiles)
        self.assertIsInstance(dicomFiles['RSfileNames'], str)
        self.assertIn('RNfileNames', dicomFiles)
        self.assertIsInstance(dicomFiles['RNfileNames'], str)
        self.assertIn('RDfileNames', dicomFiles)
        self.assertEqual(len(dicomFiles['RDfileNames']), 2)
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
        self.assertEqual(len(dicomFiles['RDfileNames']), 2)
        self.assertIn('PETfileNames', dicomFiles)
        # self.assertIsInstance(dicomFiles['PETfileNames'], str)
        self.assertIn('Unknown', dicomFiles)

    def test_sortDicoms_no_dicoms(self):
        dicomFiles = ft.sortDicoms("unittests/testData", recursive=False)
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


class test_GetDicomTypeName(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getDicomTypeName(self):
        self.assertEqual(ft.getDicomTypeName(self.dicomFiles.CTfileNames[0]), "CT Image Storage")
        self.assertEqual(ft.getDicomTypeName(self.dicomFiles.RSfileNames), "RT Structure Set Storage")
        self.assertEqual(ft.getDicomTypeName(self.dicomFiles.RNfileNames), "RT Ion Plan Storage")
        self.assertEqual(ft.getDicomTypeName(self.dicomFiles.RDfileNames[0]), "RT Dose Storage")
        # self.assertEqual(getDicomTypeName(self.dicomFiles.pet_file), "Positron Emission Tomography Image Storage")
        with self.assertRaises(TypeError):
            ft.getDicomTypeName(1)


class test_IsDicomCT(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomCT(self):
        self.assertTrue(ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.CTfileNames[0]))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.RSfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.RNfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.RDfileNames[0]))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomCT(self.dicomFiles.RSfileNames, raiseError=True)


class test_IsDicomRS(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomRS(self):
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.CTfileNames[0]))
        self.assertTrue(ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.RSfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.RNfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.RDfileNames[0]))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomRS(self.dicomFiles.RNfileNames, raiseError=True)


class test_IsDicomRN(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomRN(self):
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.CTfileNames[0]))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.RSfileNames))
        self.assertTrue(ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.RNfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.RDfileNames[0]))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomRN(self.dicomFiles.RSfileNames, raiseError=True)


class test_IsDicomRD(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomRD(self):
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.CTfileNames[0]))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.RSfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.RNfileNames))
        self.assertTrue(ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.RDfileNames[0]))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomRD(self.dicomFiles.RSfileNames, raiseError=True)


@unittest.skip("TODO: generate a dicom file with a PET image")
class test_IsDicomPET(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_isDicomPET(self):
        self.assertFalse(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.CTfileNames[0]))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.RSfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.RNfileNames))
        self.assertFalse(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.RDfileNames[0]))
        self.assertTrue(ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.PETfileNames))
        with self.assertRaises(TypeError):
            ft.ImgIO.dicom_io._isDicomPET(self.dicomFiles.RSfileNames, raiseError=True)


class test_GetRNMachineName(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNMachineName(self):
        machine_name = ft.getRNMachineName(self.dicomFiles.RNfileNames)
        self.assertIsInstance(machine_name, str)
        self.assertEqual(machine_name, 'GTR4')


class test_GetRNIsocenter(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNIsocenter(self):
        isocenter = ft.getRNIsocenter(self.dicomFiles.RNfileNames)
        self.assertIsInstance(isocenter, tuple)
        self.assertEqual(len(isocenter), 3)
        self.assertEqual(isocenter, (39.3807283826884, -238.93381359307, -570.12038065644))


class test_GetRNSpots(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNSpots(self):
        spotsInfo = ft.getRNSpots(self.dicomFiles.RNfileNames)
        self.assertFalse(spotsInfo.empty)
        self.assertEqual(spotsInfo.loc[spotsInfo.PBMU != 0].shape, (3205, 21))
        self.assertEqual(spotsInfo.PBMU.sum(), 564.9283290522891)
        self.assertEqual(spotsInfo[["PBPosX", "PBPosY"]].mean().tolist(), [-2.455148205928237, 0.24765990639625585])


class test_GetRNFields(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNFields(self):
        fieldsInfo = ft.getRNFields(self.dicomFiles.RNfileNames)
        self.assertFalse(fieldsInfo.empty)
        self.assertEqual(fieldsInfo.shape, (2, 18))
        self.assertEqual(fieldsInfo.loc[fieldsInfo.FNo == 1].FName.tolist(), ['Field 1'])
        self.assertAlmostEqual(fieldsInfo.FMU.sum(), 564.9283290522891, places=5)


class test_GetRNInfo(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRNInfo(self):
        planInfo = ft.getRNInfo(self.dicomFiles.RNfileNames)
        self.assertIsInstance(planInfo, dict)
        self.assertEqual(planInfo.fractionNo, 35)
        self.assertEqual(planInfo.dosePrescribed, 70.0)
        self.assertEqual(planInfo.targetStructName, '')
        self.assertEqual(planInfo.planLabel, 'Sphere_2F')
        self.assertEqual(planInfo.planDate, '20210702')
        self.assertEqual(planInfo.planTime, '171808.89')
        self.assertEqual(planInfo.patientName, 'RASISndwYCjIhgDE')
        self.assertEqual(planInfo.patientBirthDate, '')
        self.assertEqual(planInfo.patientID, '0h3XaG9gsktUcYRGBpgpW9d33')
        self.assertEqual(planInfo.manufacturer, 'Varian Medical Systems')
        self.assertEqual(planInfo.softwareVersions, '16.1.3')
        self.assertEqual(planInfo.stationName, '')
        self.assertEqual(planInfo.machineName, 'GTR4')
        self.assertEqual(planInfo.totalFieldsNumber, 2)
        self.assertEqual(planInfo.treatmentFieldsNumber, 2)
        self.assertEqual(planInfo.setupFieldsNumber, 0)
        self.assertEqual(planInfo.otherFieldsNumber, 0)


class test_GetRSInfo(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRSInfo(self):
        rsInfo = ft.getRSInfo(self.dicomFiles.RSfileNames)
        self.assertFalse(rsInfo.empty)
        self.assertEqual(rsInfo.groupby("ROIType")["ROIType"].count().index.to_list(), ['AVOIDANCE', 'CONTROL', 'EXTERNAL', 'ORGAN', 'PTV'])
        self.assertEqual(rsInfo.groupby("ROIType")["ROIType"].count().tolist(), [1, 8, 1, 1, 4])


class test_GetExternalName(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getExternalName(self):
        externalName = ft.getExternalName(self.dicomFiles.RSfileNames)
        self.assertIsInstance(externalName, str)
        self.assertEqual(externalName, 'External')


class test_GetCT(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getCT(self):
        imgCT = ft.getCT(self.dicomFiles.CTfileNames, displayInfo=True)
        self.assertIsInstance(imgCT, SITKImage)
        self.assertEqual(imgCT.GetSize(), (512, 512, 240))
        self.assertListEqual(np.round(imgCT.GetSpacing(), decimals=5).tolist(), [0.97656, 0.97656, 1.2])
        self.assertListEqual(np.round(imgCT.GetOrigin(), decimals=5).tolist(), [-249.51172, -470.51172, -718.3])
        self.assertAlmostEqual(ft.getStatistics(imgCT).GetMean(), -893.670319255193)


class test_GetRD(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRD(self):
        imgDose = ft.sumImg(ft.getRD(self.dicomFiles.RDfileNames, displayInfo=True))
        self.assertIsNotNone(imgDose)
        self.assertEqual(imgDose.GetSize(), (93, 116, 121))
        self.assertListEqual(np.round(imgDose.GetSpacing(), decimals=5).tolist(), [2.5,  2.5, 2.4])
        self.assertListEqual(np.round(imgDose.GetOrigin(), decimals=5).tolist(), [-114.51172, -356.48828, -718.3])
        self.assertAlmostEqual(ft.getStatistics(imgDose).GetMean(), 1.5692553116950076)


class test_GetRDFileNameForFieldNumber(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
        self.dicomFiles = ft.sortDicoms(self.testDataFolder, recursive=True)

    def test_getRDFileNameForFieldNumber(self):
        RDfileNameField1 = ft.getRDFileNameForFieldNumber(self.dicomFiles.RDfileNames, 1, displayInfo=True)
        self.assertIsNotNone(RDfileNameField1)
        self.assertIn("Field 1", str(RDfileNameField1))

    def test_getRDFileNameForFieldNumber_non_existing_field(self):
        RDfileNameField3 = ft.getRDFileNameForFieldNumber(self.dicomFiles.RDfileNames, 3, displayInfo=True)
        self.assertIsNone(RDfileNameField3)


class test_anonymizeDicoms(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
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


class test_GetStructureContoursByName(unittest.TestCase):
    def setUp(self):
        self.testDataFolder = 'unittests/testData/TPSDicoms'
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

    @unittest.skip("TODO: generate a dicom file with a structure with no ContourSequence")
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


if __name__ == '__main__':
    unittest.main()
