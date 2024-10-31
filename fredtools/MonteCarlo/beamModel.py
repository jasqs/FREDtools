from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def readBeamModel(fileName: PathLike) -> DottedDict:
    """Read beam model from a YAML file.

    The function reads the beam model parameters from a YAML beam model file.
    The beam model must be defined as a dictionary. All the pandas DataFrame-like lists
    will be converted to pandas.DataFrame objects, whereas any items in square
    brackets will be converted to a numpy array object.

    Parameters
    ----------
    fileName : string
        A string path to beam model YAML file.

    Returns
    -------
    dict
        A dictionary with the beam model and required kays.

    See Also
    --------
    writeBeamModel : write beam model to YAML file.
    interpolateBeamModel : interpolate all beam model parameters for a given nominal energy.
    """
    import numpy as np
    import yaml
    import pandas as pd
    from io import StringIO
    import re

    # load beam model from file
    with open(fileName, "r") as yaml_file:
        beamModel = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    # convert all dataFrame-like lists of strings to dataFrame
    def numpyArray(item):
        """Convert all items in dataFrame-like list in square brackets to numpy arrays"""
        if not isinstance(item, str):
            return item
        if itemArray := re.findall(r'\[(.*)\]', item):
            return np.fromstring(itemArray[0], sep=",")
        else:
            return item

    for key in beamModel.keys():
        if isinstance(beamModel[key], list):
            if ("row" in beamModel[key][-1]) and ("column" in beamModel[key][-1]):  # the key is a pandas DataFrame
                beamModel[key] = pd.read_csv(StringIO("\n".join(beamModel[key][:-1])), sep=r'\s+(?![^\[]*[\]])', engine="python")
                beamModel[key] = beamModel[key].map(numpyArray)  # map data in square brackets to numpy
                beamModel[key].set_index(beamModel[key].keys()[0], inplace=True)  # always set the first cloumn as index

    return DottedDict(beamModel)


def writeBeamModel(beamModel: dict, fileName: PathLike) -> None:
    """Write beam model to YAML.

    The function writes the beam model parameters in YAML format for a beam model file.
    The beam model must be defined as a dictionary. If a value of a given key is a pandas DataFrame,
    it will be saved to a nicely formatted table.

    Parameters
    ----------
    beamModel : dict
        Beam model defined as a dictionary with the required keys.
    fileName : string
        A string path to beam model YAML file. It is recommended to use .bm file extension.

    See Also
    --------
    readBeamModel : read beam model from YAML beam model file.
    interpolateBeamModel : interpolate all beam model parameters for a given nominal energy.
    """
    from datetime import datetime
    import os
    import numpy as np
    import yaml
    import pandas as pd
    from copy import deepcopy

    beamModelSave = deepcopy(beamModel)

    # convert all dataFrames to a nicely formated list of strings
    for key in beamModelSave.keys():
        if isinstance(beamModelSave[key], pd.DataFrame):
            beamModelSave[key].reset_index(inplace=True)
            if "nomEnergy" in beamModelSave[key].columns:
                beamModelSave[key]["nomEnergy"] = beamModelSave[key]["nomEnergy"].map(lambda x: "{:<17.2f}".format(x))
            if "Energy" in beamModelSave[key].columns:
                beamModelSave[key]["Energy"] = beamModelSave[key]["Energy"].map(lambda x: "{:<17.3f}".format(x))

            for columnName in beamModelSave[key].columns:
                if columnName in ("nomEnergy, Energy"):
                    continue
                if np.issubdtype(beamModelSave[key][columnName], np.number):
                    beamModelSave[key][columnName] = beamModelSave[key][columnName].map(lambda x: "{:<+17.10E}".format(x))
            beamModelSave[key] = beamModelSave[key].to_string(index=False, col_space=20, header=True, justify="left", show_dimensions=True).split("\n")
            beamModelSave[key] = [x.rstrip() for x in beamModelSave[key]]
    # write beam model to file
    with open(fileName, "w") as yaml_file:
        yaml.dump(beamModelSave, yaml_file, sort_keys=False, width=2000, default_flow_style=False, allow_unicode=True)


def interpolateBeamModel(beamModel: DataFrame, nomEnergy: Numberic | Iterable[Numberic], interpolation: Literal["linear", "spline", "nearest"] = "linear", splineOrder: Annotated[int, Field(strict=True, ge=0, le=5)] = 3) -> DataFrame:
    """Interpolate beam model for a given nominal energy.

    The function interpolates all the beam model parameters for a given nominal energies
    which must be in range of the defined nominal energies in the beam model. The possible
    interpolation methods are 'nearest', 'linear' or 'spline' with order in range 0-5.

    Parameters
    ----------
    beamModel : DataFrame
        Beam model defined as a pandas DataFrame object.
    nomEnergy : scalar or list
        The list of nominal energies to interpolate the beam model parameters for.
    interpolation : {'linear', 'nearest', 'spline'}, optional
        Determine the interpolation method. (def. 'linear')
    splineOrder : int, optional
        Order of spline interpolation. Must be in range 0-5. (def. 3)

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame with all parameters interpolated.

    See Also
    --------
    readBeamModel : read beam model from CSV beam model file.
    writeBeamModel : write beam model from DataFrame to a nicely formatted CSV.
    """
    from scipy.interpolate import interp1d
    import pandas as pd
    import numpy as np

    # validate nominal energy
    if isinstance(nomEnergy, Numberic):
        nomEnergy = [nomEnergy]

    # validate the interpolation method
    if not interpolation.lower() in ["linear", "nearest", "spline"]:
        error = ValueError(f"Interpolation type '{interpolation}' cannot be recognized. Only 'linear', 'nearest' and 'spline' are supported.")
        _logger.error(error)
        raise error

    interp = interpolation.lower()

    # set the proper interpolation method for spline
    if interp == "spline":
        if splineOrder > 5 or splineOrder < 0:
            error = ValueError(f"Spline order must be in range 0-5.")
            _logger.error(error)
            raise error
        else:
            interp = splineOrder

    # check if all given nomEnergy are in range of the beam model
    if np.array(nomEnergy).min() < beamModel.index.min() or np.array(nomEnergy).max() > beamModel.index.max():
        error = ValueError(f"The range of nominal energies for interpolation is {np.array(nomEnergy).min()}-{np.array(nomEnergy).max()} and it is outside the beam model nominal energy range {beamModel.index.min()}-{beamModel.index.max()}.")
        _logger.error(error)
        raise error

    # interpolate each parameter of the beam model with a given interpolation method
    beamModelEnergyInterp = {}
    beamModelEnergyInterp["nomEnergy"] = nomEnergy
    for key in beamModel.keys():
        beamModelEnergyInterp[key] = interp1d(beamModel.index, beamModel[key], kind=interp)(nomEnergy).tolist()  # type: ignore
    beamModelEnergyInterp = pd.DataFrame(beamModelEnergyInterp)

    return beamModelEnergyInterp


def calcRaysVectors(targetPoint: Iterable[Numberic] | Iterable[Iterable[Numberic]], SAD: Iterable[Numberic]) -> Tuple[NDArray, NDArray]:
    """Calculate rays positions and direction versors.

    The function calculates the ray position and direction versor from the target position.
    The target point can be a 3-element iterable or Nx3 iterable for multiple points.
    The Source-To-Axis Distance (SAD) describes the absolute distances of the spreading
    devices in order [X, Y]. It does not matter if the first divergence is in X or Y, the function
    takes this information from the distances, but the order [X,Y] must be preserved.


    Parameters
    ----------
    targetPoint : 3-element or Nx3 iterable
        The position of a single target point or positions of N target points.
    SAD : 2-element iterable
        The absolute distances of the spreading devices in order [X,Y].

    Returns
    -------
    (Nx3 numpy.array, Nx3 numpy.array)
        A tuple with two Nx3 arrays, where the first is the ray position and the second is the ray direction versor.
    """
    from collections.abc import Iterable
    import numpy as np

    # validate targetPoint
    raysTarget = np.asarray(targetPoint)
    if raysTarget.ndim == 1:
        raysTarget = np.expand_dims(raysTarget, 0)

    if raysTarget.shape[1] != 3 or raysTarget.ndim != 2:
        error = AttributeError("The targetPoint parameter must be an iterable of shape Nx3.")
        _logger.error(error)
        raise error

    # validate SAD
    if not isinstance(SAD, Iterable) or len(list(SAD)) != 2:
        error = AttributeError("The SAD parameterm must be an iterable with two elements.")
        _logger.error(error)
        raise error

    raysPosition = np.zeros((raysTarget.shape[0], 3), dtype=np.float64)
    SAD = list(SAD)

    if SAD[0] > SAD[1]:  # diverging first in X and then in Y directions, i.e. SAD[0] is upstream and SAD[1] is downstream
        raysPosition[:, 0] = (SAD[0] - SAD[1]) * raysTarget[:, 0] / (raysTarget[:, 2] + SAD[0])
        raysPosition[:, 2] = -SAD[1]
    elif SAD[0] < SAD[1]:  # diverging first in Y and then in X directions, i.e. SAD[1] is upstream and SAD[0] is downstream
        raysPosition[:, 1] = (SAD[1] - SAD[0]) * raysTarget[:, 1] / (raysTarget[:, 2] + SAD[1])
        raysPosition[:, 2] = -SAD[0]
    elif SAD[0] == SAD[1]:  # diverging in Y and X directions in the same place
        raysPosition[:, 2] = -SAD[0]

    raysVersor = raysTarget - raysPosition
    raysVersor = raysVersor / np.linalg.norm(raysVersor, axis=1)[:, None]

    return raysPosition, raysVersor
