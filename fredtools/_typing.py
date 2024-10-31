# https://docs.pydantic.dev/latest/concepts/types/

from typing import Any, Iterable, Sequence, Literal, Union, Annotated, TypeVar, List, Tuple, SupportsFloat, NewType, overload, Type, NamedTuple, TypeAlias
from pydantic import Field, StringConstraints, NonNegativeInt, NonNegativeFloat
from dotted_dict import DottedDict

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

from matplotlib.axes import Axes
from matplotlib.image import AxesImage

# from pandas._typing import Scalar as PDScalar
# numberic
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

# DVH
from dicompylercore.dvh import DVH  # noqa
