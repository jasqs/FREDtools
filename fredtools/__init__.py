from ._logger import configureLogging, getLogger

from . import _typing
from . import _helper

from . import ImgAnalyse
from .ImgAnalyse import _imgTypeChecker
from .ImgAnalyse.dvhAnalyse import (getDVHMask, getDVHStruct)
from .ImgAnalyse.imgAnalyse import (getExtent, getSize, getImageCenter, getMassCenter, getMaxPosition, getMinPosition, getVoxelCentres, getVoxelEdges, getVoxelPhysicalPoints, getExtMpl, pos, arr, vec, isPointInside, getStatistics, compareImgFoR)
from .ImgAnalyse.imgDisplay import (showSlice, showSlices)
from .ImgAnalyse.imgInfo import (displayImageInfo)
from .ImgAnalyse.imgTransformCoordinates import (transformIndexToPhysicalPoint, transformContinuousIndexToPhysicalPoint, transformPhysicalPointToIndex, transformPhysicalPointToContinuousIndex)
from .ImgAnalyse.spotAnalyse import (fitSpotProfile)


from . import ImgIO
from .ImgIO.dicom_io import (getDicomTypeName, sortDicoms, getRNMachineName, getRNIsocenter, getRNSpots, getRNFields, getRNInfo, getRSInfo, getExternalName, getCT, getPET, getRD, getRDFileNameForFieldNumber, anonymizeDicoms)
from .ImgIO.imgConverter import (SITK2ITK, ITK2SITK)
from .ImgIO.influenceMatrix_io import (getInmFREDBaseImg, getInmFREDSumImage, getInmFREDPoint, getInmFREDInfo)
from .ImgIO.mhd_io import (writeMHD, readMHD, convertMHDtoSingleFile, convertMHDtoDoubleFiles)
from .ImgIO.OmniPro_io import (readOPG, readOPD)


from . import ImgManipulate
from .ImgManipulate.imgGetSubimg import (getSlice, getProfile, getPoint, getInteg, getCumSum)
from .ImgManipulate.imgManipulate import (mapStructToImg, floatingToBinaryMask, cropImgToMask, setValueMask, resampleImg, sumImg, imgDivide, sumVectorImg, getImgBEV, overwriteCTPhysicalProperties, setIdentityDirection, addMarginToMask, addGaussMarginToMask, addExpMarginToMask)
from .ImgManipulate.imgCreate import (createEllipseMask, createConeMask, createCylinderMask, createImg)

from . import Miscellaneous
from .Miscellaneous.landauVavilovGauss import (pdfLandau, pdfLandauGauss, fitLandau, fitLandauGauss, pdfVavilov, fitVavilov)
from .Miscellaneous.miscellaneous import (mergePDF, getHistogram, sigma2fwhm, fwhm2sigma, getLineFromFile, getCPUNo, re_number)


from . import MonteCarlo
from .MonteCarlo.beamModel import (readBeamModel, writeBeamModel, interpolateBeamModel, calcRaysVectors)
from .MonteCarlo.fredMC import (setFieldsFolderStruct, readFREDStat, getFREDVersions, checkFREDVersion, getFREDVersion, runFRED)
from .MonteCarlo.gateMC import (readGATE_HITSActor, readGATE_PSActor, readGATEStat)

from . import BraggPeak
from .BraggPeak.braggPeakAnalyse import (braggPeak)

from . import GammaIndex
from .GammaIndex.gammaIndex import (calcGammaIndex, getGIstat, getGIcmap)

from . import ProtonOptimisation

_version = [0, 8, 8]
__version__ = ".".join(map(str, _version))

# configure logging if no root logger configured
configureLogging(_logger.logging.INFO, force=False)

# log FREDtools version in DEBUG mode
getLogger(__name__).debug(f"Loaded FREDtools version {__version__}")


# global parameters
CPUNO: _typing.Literal["auto"] | _typing.NonNegativeInt = "auto"
'''CPU number to use for multiprocessing. If set to "auto", the number of CPUs will be determined automatically.'''

"""
The FREDtools is a library of packages collecting modules with functions. 
"""
