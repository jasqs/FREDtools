import os
import warnings
import fredtools as ft


def setFieldsFolderStruct(folderPath, RNfileName, folderName='FRED', displayInfo=False):
    r"""Create folder structure for each field in treatment plan

    The function creates a folder structure in a given `folderPath` for each field separately.
    The folder structure is in form:

        folderPath/folderName:
                    |- F1
                    |- F2
                    ...

    Parameters
    ----------
    folderPath : path
        Path to folder to create the structure.
    RNfileName : path
        Path to RN dicom file of a treatment plan.
    folderName : string
        Name of the folder to create (def. 'FRED')
    displayInfo : bool
        Displays a summary of the function results (def. False).

    Returns
    -------
    path
        Path to created folder structure.

    """
    SimFolder = os.path.join(folderPath, folderName)
    if not os.path.exists(SimFolder):
        os.mkdir(SimFolder)
    else:
        warnings.warn('Warning: {:s} simulation folder already exists.'.format(folderName))

    # create subfolders for fields
    planInfo = ft.getRNInfo(RNfileName, displayInfo=False)
    for fieldNo in planInfo['fieldsNumber']:
        if not os.path.exists(os.path.join(SimFolder, 'F{:d}'.format(fieldNo))):
            os.mkdir(os.path.join(SimFolder, 'F{:d}'.format(fieldNo)))
    if displayInfo:
        print(f'### {ft._currentFuncName()} ###')
        print('# Created {:d} field folders in {:s}'.format(planInfo['numberOfBeams'], SimFolder))
        print('#' * len(f'### {ft._currentFuncName()} ###'))
    return SimFolder
