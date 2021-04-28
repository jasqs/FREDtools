import os
import glob
import pydicom as dicom
import warnings


def sortDicoms(searchFolder, displayInfo=False):

    dicomfileNames = glob.glob(os.path.join(searchFolder, '*.dcm'))

    CTfileNames = []
    RSfileNames = []
    RNfileNames = []
    RDfileNames = []
    for dicomfileName in dicomfileNames:
        try:
            dicomTags = dicom.read_file(dicomfileName)
        except:
            warnings.warn('Warning: could not read file {:s}'.format(dicomfileName))
            continue

        if 'SOPClassUID' in dicomTags:  # check if SOPClassUID exists
            if dicomTags.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':     # CT Image Storage
                CTfileNames.append(dicomfileName)
            if dicomTags.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':  # Radiation Therapy Structure Set Storage
                RSfileNames.append(dicomfileName)
            if dicomTags.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.8':  # Radiation Therapy Ion Plan Storage
                RNfileNames.append(dicomfileName)
            if dicomTags.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.2':  # Radiation Therapy Dose Storage
                RDfileNames.append(dicomfileName)
    if displayInfo:
        print('############# sort dicoms ##############')
        print('# Found dicoms: {:d} x CT, {:d} x structure, {:d} x treatment plan and {:d} x dose.'.format(len(CTfileNames), len(RSfileNames), len(RNfileNames), len(RDfileNames)))
        print('########################################')

    return {'CTfileNames': CTfileNames,
            'RSfileNames': RSfileNames,
            'RNfileNames': RNfileNames,
            'RDfileNames': RDfileNames}
