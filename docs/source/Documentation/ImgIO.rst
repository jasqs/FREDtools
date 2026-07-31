Image Reading and Writing
=================================

A collection of useful functions for reading and writing images, implemented in the ``fredtools.ImgIO`` subpackage. The supported image types are:

*   MetaImage format in double (\*.mhd+\*.raw) or single files (only \*.mhd),
*   Dicom format (reading only) for 3D/2D images (e.g. dose distribution), Structures (i.e. RS\*.dcm), Proton treatment plans (i.e. RN\*.dcm or RP\*.dcm) and CT images,
*   Influence matrices produced by the FRED Monte Carlo,
*   OmniPro measurement files.

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

FRED influence matrices
------------------------------------------------

Functions for reading influence matrices produced by the FRED Monte Carlo. The influence matrix is usually a 3D image describing the influence (dose, LET or other quantity) for each pencil beam, therefore it can be treated as a 4D image with geometrical X, Y, Z and pencil beam dimensions.

.. note::
    The binary influence matrix file format has changed across FRED versions. The functions support influence matrix files in format versions 2.x and 3.x and raise an error for unsupported versions.

.. autofunction:: fredtools.getInmFREDInfo

.. autofunction:: fredtools.getInmFREDBaseImg

.. autofunction:: fredtools.getInmFREDSparse

OmniPro files (\*.opg, \*.opd)
------------------------------------------------

.. autofunction:: fredtools.readOPG

.. autofunction:: fredtools.readOPD

Format conversion
------------------------------------------------

.. autofunction:: fredtools.SITK2ITK

.. autofunction:: fredtools.ITK2SITK

.. autofunction:: fredtools.img2vec
