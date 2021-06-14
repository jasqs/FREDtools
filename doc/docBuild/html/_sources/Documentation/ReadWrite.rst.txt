Image Reading and Writing
=================================

A collection of useful functions for reading and writing images are implemented. The supported image type are:

*   MetaImage format in double (\*.mhd+\*.raw) or single files (only \*.mhd),
*   Map3D images, i.e. an obsolate format used by the FRED Monte Carlo engine for saving 3D images (not recommended),
*   Dicom format (reading only) for 3D/2D images (e.g. dose distribution), Structures (i.e. RS\*.dcm), Proton treatment plans (i.e. RN\*.dcm or RP\*.dcm) and CT images.

MetaImage files (\*.mhd, \*.mha)
------------------------------------------------

.. autofunction:: fredtools.readMHD

.. autofunction:: fredtools.writeMHD

.. autofunction:: fredtools.convertMHDtoSingleFile

.. autofunction:: fredtools.convertMHDtoDoubleFiles

Map3D images (\*.m3d)
------------------------------------------------

.. caution:: The Map3D file format is obsolete and it is not recommended to be used to store the image data. Use MetaImage file format instead.

.. autofunction:: fredtools.readMap3D

.. autofunction:: fredtools.writeMap3D

DICOM files (\*.dcm, \*.dicom)
------------------------------------------------

.. autofunction:: fredtools.getDicomType

.. autofunction:: fredtools.sortDicoms

.. autofunction:: fredtools.getRNInfo

.. autofunction:: fredtools.getRSInfo

.. autofunction:: fredtools.getExternalName

.. autofunction:: fredtools.getCT
