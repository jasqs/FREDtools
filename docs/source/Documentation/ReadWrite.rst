Image Reading and Writing
=================================

A collection of useful functions for reading and writing images are implemented. The supported image type are:

*   MetaImage format in double (\*.mhd+\*.raw) or single files (only \*.mhd),
*   Dicom format (reading only) for 3D/2D images (e.g. dose distribution), Structures (i.e. RS\*.dcm), Proton treatment plans (i.e. RN\*.dcm or RP\*.dcm) and CT images.

MetaImage files (\*.mhd, \*.mha)
------------------------------------------------

.. autofunction:: fredtools.readMHD

.. autofunction:: fredtools.writeMHD

.. autofunction:: fredtools.convertMHDtoSingleFile

.. autofunction:: fredtools.convertMHDtoDoubleFiles

DICOM files (\*.dcm, \*.dicom)
------------------------------------------------

.. autofunction:: fredtools.getDicomTypeName

.. autofunction:: fredtools.sortDicoms

.. autofunction:: fredtools.getRNMachineName

.. autofunction:: fredtools.getRNIsocenter

.. autofunction:: fredtools.getRNInfo

.. autofunction:: fredtools.getRNFields

.. autofunction:: fredtools.getRNSpots

.. autofunction:: fredtools.getRSInfo

.. autofunction:: fredtools.checkDicomsUID

.. autofunction:: fredtools.getExternalName

.. autofunction:: fredtools.getCT

.. autofunction:: fredtools.getPET

.. autofunction:: fredtools.getRD

.. autofunction:: fredtools.getRDFileNameForFieldNumber

.. autofunction:: fredtools.anonymizeDicoms

Other formats
------------------------------------------------

.. autofunction:: fredtools.readOPG

.. autofunction:: fredtools.readOPD

Format conversion
------------------------------------------------

.. autofunction:: fredtools.SITK2ITK

.. autofunction:: fredtools.ITK2SITK

.. autofunction:: fredtools.img2vec
