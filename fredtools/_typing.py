# https://docs.pydantic.dev/latest/concepts/types/

from typing import Any, Iterable, Sequence, Literal, Union, Annotated, TypeVar, List, Tuple, SupportsFloat, NewType, Type, NamedTuple, TypeAlias
from typing import cast, overload
from pydantic import Field, StringConstraints, NonNegativeInt, NonNegativeFloat
from dotted_dict import DottedDict

# Self (used in class definitions)
import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# pandas
from pandas import DataFrame
from pandas import Series as PDSeries

# lmfit
from lmfit.model import ModelResult as LMFitModelResult

# matplotlib
from matplotlib.colors import LinearSegmentedColormap, Colormap

# ITK and SimpleITK
from itk import Image as ITKImage  # type: ignore
from SimpleITK import StatisticsImageFilter
from SimpleITK import Image as SITKImage
from SimpleITK import Transform as SITKTransform

# numpy
from numpy.typing import NDArray, ArrayLike, DTypeLike
from scipy.sparse import spmatrix, csr_matrix
from cupy.sparse import spmatrix as cp_spmatrix
from cupy.sparse import csr_matrix as cp_csr_matrix
SparseMatrix: TypeAlias = Union[spmatrix, cp_spmatrix]
SparseMatrixCSR: TypeAlias = Union[csr_matrix, cp_csr_matrix]

# matplotlib
from matplotlib.axes import Axes  # noqa
from matplotlib.image import AxesImage  # noqa

# from pandas._typing import Scalar as PDScalar
# numeric
import numpy as np  # noqa
Numberic: TypeAlias = Union[int, float, np.number]
PointLike: TypeAlias = Iterable[Numberic]

# path
import os  # noqa
PathLike: TypeAlias = Union[os.PathLike, str]


# dicom
from pydicom import Dataset, FileDataset, DataElement  # noqa
DicomDataset: TypeAlias = Union[Dataset, FileDataset]
DicomDataElement: TypeAlias = DataElement

# shapely
from shapely.geometry import Polygon as ShapePolygon  # noqa
from shapely.geometry import MultiPolygon as ShapeMultiPolygon  # noqa
