from typing import *
from SimpleITK import StatisticsImageFilter
from SimpleITK import Image as SITKImage
from SimpleITK import Transform as SITKTransform
from itk import Image as ITKImage
from typing import Any, Iterable, Literal, Union, Annotated
from typing import SupportsFloat as Numberic


from numpy.typing import NDArray
from dicompylercore.dvh import DVH

# https://docs.pydantic.dev/latest/concepts/types/
from pydantic import Field, StringConstraints, NonNegativeInt
