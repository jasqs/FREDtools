from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


class beamModel:
    def __init__(self):
        from datetime import datetime
        import pandas as pd
        # beam model description
        self.name = None
        self.creationTime = datetime.now()
        self.radiationType = "proton"

        self.siteName: str | None = None
        self.machineName: str | None = None
        self.machineVendor: str | None = None
        self.machineModel: str | None = None

        self.notes: str | None = None

        # machine parameters
        self.spreadingDeviceDistance = (100.0, 100.0)  # Spreading Device Distance in mm, [X, Y]
        self.sourceToAxisDistance = 100.0  # Source-To-Axis Distance (SAD) in mm

        # energy model
        self.energyModel = pd.DataFrame(columns=["nomEnergy", "Energy", "dEnergy", "aX", "bX", "cX", "aY", "bY", "cY", "scalingFactor"])
        self.interpolation = "linear"
        self.splineOrder = 3

        # range shifters
        self.rsModel = pd.DataFrame(columns=["name", "L", "material", "direction", "voxels", "origin"])

        # nozzle materials
        self._materials = pd.DataFrame(columns=["name", "density", "basedOn"])

    @property
    def creationTime(self) -> str:
        """Returns the creation time of the beam model."""
        return self._creationTime.strftime("%Y-%m-%d %H:%M:%S")

    @creationTime.setter
    def creationTime(self, creationTime: DateTime) -> None:
        """Sets the creation time of the beam model."""
        if not isinstance(creationTime, DateTime):
            error = AttributeError("The creationTime must be a datetime object.")
            _logger.error(error)
            raise error
        self._creationTime = creationTime

    @property
    def radiationType(self) -> str:
        """Returns the radiation type."""
        return self._radiationType

    @radiationType.setter
    def radiationType(self, radiationType: str) -> None:
        """Sets the radiation type."""
        if radiationType not in ["proton", "carbon", "helium"]:
            error = AttributeError("Only the radiation types ['proton', 'carbon', or 'helium'] are supported now.")
            _logger.error(error)
            raise error
        self._radiationType = radiationType

    @property
    def spreadingDeviceDistance(self) -> Tuple[float, float]:
        """Returns the spreading device distance in mm.

        The spreading device distance describes the absolute distances of the spreading
        devices in the order [X, Y].
        """
        return self._spreadingDeviceDistance

    @spreadingDeviceDistance.setter
    def spreadingDeviceDistance(self, spreadingDeviceDistance: Tuple[float, float]) -> None:
        """Sets the spreading device distance in mm.

        The spreading device distance describes the absolute distances of the spreading
        devices in the order [X, Y]. It does not matter if the first divergence is in X or Y, the function
        takes this information from the distances, but the order [X,Y] must be preserved.
        """
        if not isinstance(spreadingDeviceDistance, tuple) or len(spreadingDeviceDistance) != 2:
            error = AttributeError("The spreadingDeviceDistance parameter must be a tuple with two elements.")
            _logger.error(error)
            raise error
        self._spreadingDeviceDistance: Tuple[float, float] = spreadingDeviceDistance

    @property
    def sourceToAxisDistance(self) -> float:
        """Returns the Source-To-Axis Distance (SAD) in mm.

        The Source-To-Axis Distance (SAD) describes the absolute distance of the
        source to the isocenter.        
        """
        return self._sourceToAxisDistance

    @sourceToAxisDistance.setter
    def sourceToAxisDistance(self, sourceToAxisDistance: float) -> None:
        """Sets the Source-To-Axis Distance (SAD) in mm.

        The Source-To-Axis Distance (SAD) describes the absolute distance of the
        source to the isocenter.
        """
        if not isinstance(sourceToAxisDistance, float):
            error = AttributeError("The sourceToAxisDistance parameter must be a float.")
            _logger.error(error)
            raise error
        self._sourceToAxisDistance: float = sourceToAxisDistance

    @property
    def energyModel(self) -> DataFrame:
        """Returns the energy model of the beam model."""
        if self._energyModel.empty:
            _logger.warning("The energy model is empty.")
        return self._energyModel

    @energyModel.setter
    def energyModel(self, energyModel: DataFrame) -> None:
        """Sets the energy model of the beam model."""
        if not isinstance(energyModel, DataFrame):
            error = AttributeError("The energyModel parameter must be a DataFrame.")
            _logger.error(error)
            raise error
        if not {"nomEnergy", "Energy", "dEnergy"}.issubset(energyModel.reset_index().keys()):
            error = AttributeError("The energy definition of the energyModel DataFrame must contain at least the columns ['nomEnergy', 'Energy', 'dEnergy'].")
            _logger.error(error)
            raise error
        if not {"aX", "bX", "cX", "aY", "bY", "cY"}.issubset(energyModel.reset_index().keys()):
            error = AttributeError("The beam envelope (propagation) definition should be based on the sigma squared model so the DataFrame must contain the columns ['aX', 'bX', 'cX', 'aY', 'bY', 'cY'].")
            _logger.error(error)
            raise error
        if "scalingFactor" not in energyModel.reset_index().keys():
            error = AttributeError("The dosimatric callibration of the energy model must contain the 'scalingFactor' column.")
            _logger.error(error)
            raise error

        energyModel = energyModel.copy()
        energyModel = energyModel.reset_index().set_index("nomEnergy")
        energyModel.sort_index(inplace=True)
        self._energyModel = energyModel

    @property
    def nomEnergies(self) -> tuple:
        """Returns the nominal energies the beam model is defined for.

        The nomional energies are usually the ones defined by the TPS. 
        """
        if self._energyModel.empty:
            _logger.warning("The energy model is empty.")

        return tuple(self._energyModel.reset_index().nomEnergy)

    @property
    def interpolation(self) -> str:
        """Returns the interpolation method used for the energy model."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation: Literal['linear', 'nearest', 'spline']) -> None:
        """Sets the interpolation method used for the energy model."""
        if interpolation not in ["linear", "nearest", "spline"]:
            error = AttributeError("The interpolation parameter must be one of ['linear', 'nearest', 'spline'].")
            _logger.error(error)
            raise error
        self._interpolation: Literal['linear', 'nearest', 'spline'] = interpolation

    @property
    def splineOrder(self) -> int:
        """Returns the order of the spline used for the energy model."""
        return self._splineOrder

    @splineOrder.setter
    def splineOrder(self, splineOrder: Annotated[int, Field(strict=True, ge=0, le=5)]) -> None:
        """Sets the order of the spline used for the energy model."""
        if not isinstance(splineOrder, int) or splineOrder < 0 or splineOrder > 5:
            error = AttributeError("The splineOrder parameter must be an integer between 0 and 5.")
            _logger.error(error)
            raise error
        self._splineOrder: Annotated[int, Field(strict=True, ge=0, le=5)] = splineOrder

    def interpolateBeamModel(self, nomEnergy: Numberic | Iterable[Numberic]) -> DataFrame:
        import pandas as pd
        import fredtools as ft

        # validate nominal energy
        if isinstance(nomEnergy, Numberic):
            nomEnergy = [nomEnergy]
        else:
            nomEnergy = list(nomEnergy)

        # check it the beam model was defined
        if self._energyModel is None:
            error = AttributeError("The beam model was not defined. Please set the energy model first.")
            _logger.error(error)
            raise error

        # check if all given nomEnergy are in range of the beam model
        if np.array(nomEnergy).min() < self._energyModel.index.min() or np.array(nomEnergy).max() > self._energyModel.index.max():
            if len(nomEnergy) == 1:
                error = ValueError(f"The nominal energy {nomEnergy[0]} is outside the beam model nominal energy range {self._energyModel.index.min()}-{self._energyModel.index.max()}.")
            else:
                error = ValueError(f"The range of nominal energies for interpolation is {np.array(nomEnergy).min()}-{np.array(nomEnergy).max()} and it is outside the beam model nominal energy range {self._energyModel.index.min()}-{self._energyModel.index.max()}.")
            _logger.error(error)
            raise error

        # interpolate each parameter of the beam model with a given interpolation method
        beamModelEnergyInterp = {}
        beamModelEnergyInterp["nomEnergy"] = nomEnergy
        for key in self._energyModel.keys():
            beamModelEnergyInterp[key] = ft._helper.get1DInterpolator(self._energyModel.index, self._energyModel[key], interpolation=self._interpolation, splineOrder=self._splineOrder)(nomEnergy).tolist()
        beamModelEnergyInterp = pd.DataFrame(beamModelEnergyInterp)

        beamModelEnergyInterp = beamModelEnergyInterp.set_index("nomEnergy")

        return beamModelEnergyInterp

    @property
    def rsModel(self) -> DataFrame:
        """Returns the range shifter model of the beam model."""
        if self._rsModel.empty:
            _logger.warning("The range shifter model is not set.")
        return self._rsModel

    @rsModel.setter
    def rsModel(self, rsModel: DataFrame) -> None:
        """Sets the range shifter model of the beam model."""
        if not isinstance(rsModel, DataFrame):
            error = AttributeError("The rsModel parameter must be a DataFrame.")
            _logger.error(error)
            raise error
        if not {"name", "L", "material"}.issubset(rsModel.reset_index().keys()):
            error = AttributeError("The rsModel DataFrame must contain at least the 'name', 'L', and 'material' columns.")
            _logger.error(error)
            raise error

        rsModel = rsModel.copy()

        if "direction" not in rsModel.keys():
            rsModel["direction"] = [np.eye(3)]
            _logger.debug("The rsModel DataFrame does not contain the 'direction' column. Setting it to identity.")
        if "voxels" not in rsModel.keys():
            rsModel["voxels"] = [np.ones(3, dtype=int)]
            _logger.debug("The rsModel DataFrame does not contain the 'voxels' column. Setting it to ones.")
        if "O" not in rsModel.keys():
            rsModel["O"] = [np.zeros(3)]
            _logger.debug("The rsModel DataFrame does not contain the 'O' column. Setting it to zeros.")

        self._rsModel = rsModel.set_index("name")

    @property
    def materials(self) -> DataFrame:
        """Returns the materials used in the beam model."""
        if self._materials.empty:
            _logger.warning("The materials are not set.")
        return self._materials

    @materials.setter
    def materials(self, materials: DataFrame) -> None:
        """Sets the materials used in the beam model."""
        if not isinstance(materials, DataFrame):
            error = AttributeError("The materials parameter must be a DataFrame.")
            _logger.error(error)
            raise error

        if not {"name", "density", "basedOn"}.issubset(materials.reset_index().keys()):
            error = AttributeError("The materials DataFrame must contain at least the 'name', 'density', and 'basedOn' columns.")
            _logger.error(error)
            raise error

        materials = materials.copy()
        self._materials = materials.set_index("name")

    def displayInfo(self) -> None:
        """Displays information about the beam model."""
        logStr = []
        logStr.append(f"Beam Model: {self.name}")
        logStr.append(f"Machine: {self.machineName} at {self.siteName} ({self.machineVendor} {self.machineModel})")
        logStr.append(f"Creation Time: {self.creationTime}")
        logStr.append(f"Radiation Type: {self.radiationType}")
        logStr.append(f"Spreading device dist. [X/Y]: {self.spreadingDeviceDistance[0]} mm, {self.spreadingDeviceDistance[1]} mm")

        if self.energyModel is not None:
            logStr.append("Energy Model:")
            logStr.append(self.energyModel.to_string())
        else:
            logStr.append("Energy Model: Not set")

        if self.rsModel is not None:
            logStr.append("Range Shifter Model:")
            logStr.append(self.rsModel.to_string())
        else:
            logStr.append("Range Shifter Model: Not set")

        if self.materials is not None:
            logStr.append("Materials:")
            logStr.append(self.materials.to_string())
        else:
            logStr.append("Materials: Not set")

        _logger.info("\n".join(logStr))

    def fromYAML(self, fileName: PathLike) -> None:
        """Loads the beam model from a YAML file.

        Parameters
        ----------
        fileName : PathLike
            The path to the YAML file containing the beam model.
        """
        import numpy as np
        import yaml
        import pandas as pd
        from io import StringIO
        import re
        from fredtools.MonteCarlo.beamModel import twiss2SigmaSquared
        from datetime import datetime
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

        if "Description" not in beamModel.keys():
            beamModel["Description"] = beamModel["BM Description"]
        self.name = beamModel["Description"]["name"]
        self.creationTime = datetime.strptime(beamModel["Description"]["creationTime"], "%Y/%m/%d %H:%M:%S")
        self.radiationType = "proton"

        if "Energy" not in beamModel.keys():
            beamModel["Energy"] = beamModel["BM Energy"]

        beamModelEnergy = beamModel["Energy"].reset_index()
        if not {"nomEnergy", "Energy", "dEnergy", "alphaX", "betaX", "epsilonX", "alphaY", "betaY", "epsilonY", "scalingFactor"}.issubset(beamModelEnergy.keys()):
            error = AttributeError("The energy model imported from YAML must contain the columns ['nomEnergy', 'Energy', 'dEnergy', 'alphaX', 'betaX', 'epsilonX', 'alphaY', 'betaY', 'epsilonY', 'scalingFactor'].")
            _logger.error(error)
            raise error

        aX, bX, cX = twiss2SigmaSquared(beamModelEnergy["epsilonX"], beamModelEnergy["alphaX"], beamModelEnergy["betaX"])
        aY, bY, cY = twiss2SigmaSquared(beamModelEnergy["epsilonY"], beamModelEnergy["alphaY"], beamModelEnergy["betaY"])
        beamModelEnergy = pd.DataFrame({
            "nomEnergy": beamModelEnergy["nomEnergy"],
            "Energy": beamModelEnergy["Energy"],
            "dEnergy": beamModelEnergy["dEnergy"],
            "aX": aX,
            "bX": bX,
            "cX": cX,
            "aY": aY,
            "bY": bY,
            "cY": cY,
            "scalingFactor": beamModelEnergy["scalingFactor"]
        })
        beamModelEnergy.set_index("nomEnergy", inplace=True)
        self.energyModel = beamModelEnergy

        if "RangeShifters" not in beamModel.keys():
            beamModel["RangeShifters"] = beamModel["BM RangeShifters"]
        rsModel = pd.DataFrame(beamModel["RangeShifters"]).T
        rsModel.index.name = "name"
        rsModel.reset_index(inplace=True)
        self.rsModel = rsModel

        if "Materials" not in beamModel.keys():
            beamModel["Materials"] = beamModel["BM Materials"]
        materials = pd.DataFrame(beamModel["Materials"]).T
        materials.index.name = "name"
        materials.rename(columns={"rho": "density"}, inplace=True)
        materials.reset_index(inplace=True)
        self.materials = materials

    def getSigma(self, distance: Numberic, nomEnergy: Numberic | Iterable[Numberic]) -> Tuple[Numberic, Numberic] | Tuple[List[Numberic], List[Numberic]]:
        import numpy as np

        if self._energyModel.empty:
            _logger.warning("The energy model is empty.")
            return (0, 0)

        energyModel: DataFrame = self.interpolateBeamModel(nomEnergy)

        sigmaX = np.sqrt(np.asarray(energyModel["aX"]) + np.asarray(energyModel["bX"]) * float(distance) + np.asarray(energyModel["cX"]) * float(distance) ** 2)
        sigmaY = np.sqrt(np.asarray(energyModel["aY"]) + np.asarray(energyModel["bY"]) * float(distance) + np.asarray(energyModel["cY"]) * float(distance) ** 2)

        return (float(sigmaX[0]), float(sigmaY[0])) if isinstance(nomEnergy, Numberic) else (sigmaX.tolist(), sigmaY.tolist())

    def getGateParams(self, sourceToAxisDistance: Numberic, nomEnergy: Numberic | Iterable[Numberic]) -> DataFrame:
        """ Get the beam parameters for GATE simulation.

        This function calculates the beam parameters for GATE simulation based on the nozzle exit position and nominal energy.
        According to the GATE documentation, the beam propagation parameters are modeled according to the Fermi-Eyges theory 
        (Techniques of Proton Radiotherapy: Transport Theory B. Gottschalk May 1, 2012), that describes the correlated momentum 
        spread of the particle with 4 parameters (each for x and y direction, assuming a beam directed as z):

            - sigma: beam size in [X,Y] directions at the beam production point (`nozzleExit` )
            - divergence: beam divergence in [X,Y] directions
            - emittance: constant area in phase space in [X,Y] directions
            - convergence: beam convergence in [X,Y] directions

        Parameters
        ----------
        nozzleExit : float
            The distance from the nozzle exit to the isocenter in mm.
        nomEnergy : float or iterable of floats
            The nominal energy of the beam in MeV or a list of nominal energies.

        Returns
        -------
        DataFrame
            A DataFrame containing the beam parameters for GATE simulation, including:
            - Energy: nominal energy in [MeV]
            - dEnergy: energy spread in [MeV]
            - sigmaX: beam size in X direction in [mm]
            - sigmaY: beam size in Y direction in [mm]
            - thetaX: beam divergence in X direction in [rad]
            - thetaY: beam divergence in Y direction in [rad]
            - epsilonX: emittance in X direction in [mm * rad]
            - epsilonY: emittance in Y direction in [mm * rad]
            - convergenceX: convergence in X direction (always True)
            - convergenceY: convergence in Y direction (always True)
            - scalingFactor: scaling factor for the beam model (in [p/MU])
        """
        import numpy as np
        import pandas as pd
        import fredtools as ft

        if self._energyModel.empty:
            _logger.warning("The energy model is empty.")
            return pd.DataFrame()

        energyModel: DataFrame = self.interpolateBeamModel(nomEnergy)
        sigmaX, sigmaY = self.getSigma(sourceToAxisDistance, nomEnergy)
        thetaX = np.sqrt(np.asarray(energyModel["cX"]))
        thetaY = np.sqrt(np.asarray(energyModel["cY"]))
        epsilonX, _, _ = ft.MonteCarlo.beamModel.sigmaSquare2Twiss(energyModel["aX"], energyModel["bX"], energyModel["cX"])
        epsilonY, _, _ = ft.MonteCarlo.beamModel.sigmaSquare2Twiss(energyModel["aY"], energyModel["bY"], energyModel["cY"])

        beamModelEnergyGATE = pd.DataFrame({'Energy': energyModel.Energy,  # [MeV]
                                            'dEnergy': energyModel.dEnergy,  # [MeV]
                                            'sigmaX': sigmaX,  # [mm]
                                            'sigmaY': sigmaY,  # [mm]
                                            'thetaX': thetaX,  # [rad]
                                            'thetaY': thetaY,  # [rad]
                                            'epsilonX': np.asarray(epsilonX) * np.pi,  # [mm * rad]
                                            'epsilonY': np.asarray(epsilonY) * np.pi,  # [mm * rad]
                                            'convergenceX': [True] * len(thetaX),
                                            'convergenceY': [True] * len(thetaY),
                                            'scalingFactor': energyModel.scalingFactor,  # [p/MU]
                                            })

        return beamModelEnergyGATE

    def toPickle(self, fileName: PathLike) -> None:
        """Saves the beam model to a pickle file.

        Parameters
        ----------
        fileName : PathLike
            The name of the file to save the beam model to. The file will be saved in binary format.
            If the file already exists, it will be overwritten.
        """
        import pickle as pkl

        with open(fileName, "wb") as f:
            pkl.dump(self, f)

    def fromPickle(self, fileName: PathLike) -> None:
        """Loads the beam model from a pickle file.

        Parameters
        ----------
        fileName : PathLike
            The name of the file to load the beam model from. The file must be in binary format.
        """
        import pickle as pkl
        if not os.path.exists(fileName):
            error = FileNotFoundError(f"The file {fileName} does not exist.")
            _logger.error(error)
            raise error

        with open(fileName, "rb") as f:
            beamModel = pkl.load(f)

        if not isinstance(beamModel, self.__class__):
            error = TypeError(f"The file {fileName} does not contain a valid beamModel object.")
            _logger.error(error)
            raise error

        self.__dict__.update(beamModel.__dict__)


def twiss2SigmaSquared(epsilon: Numberic, alpha: Numberic, beta: Numberic) -> Tuple[Numberic, Numberic, Numberic] | Tuple[List[Numberic], List[Numberic], List[Numberic]]:
    """ Convert Twiss parameters to sigma squared model.

    Convert beam propagation parameters defined as Twiss model to sigma squared model parameters.
    The Twiss model is defined as:

        sigma^2 = epsilon *(beta - 2*alpha*z + ((1 + alpha^2)/beta)*z^2)

    The sigma squared model is defined as:

        sigma^2 = a + b*z + c*z^2

    Parameters
    ----------
        epsilon: Numeric
            Emittance of the beam, preferably in [mm * rad].
        alpha: Numeric
            Alpha parameter of the beam, unitless.
        beta: Numeric
            Beta parameter of the beam, preferably in [mm].

    Returns
    -------
        tuple[Numeric, Numeric, Numeric]
            A tuple containing the parameters (a, b, c) of the sigma squared model.
            The parameters are in the same units as epsilon and beta.
    """

    a = np.asarray(epsilon) * np.asarray(beta)
    b = -2 * np.asarray(epsilon) * np.asarray(alpha)
    c = np.asarray(epsilon) * ((1 + np.asarray(alpha)**2) / np.asarray(beta))

    return (float(np.real(a)), float(b), float(c)) if np.isscalar(a) else (a.tolist(), b.tolist(), c.tolist())


def sigmaSquare2Twiss(a: Numberic | Iterable[Numberic], b: Numberic | Iterable[Numberic], c: Numberic | Iterable[Numberic]) -> Tuple[Numberic, Numberic, Numberic] | Tuple[List[Numberic], List[Numberic], List[Numberic]]:
    """ Convert sigma squared model parameters to Twiss parameters.

    Convert beam propagation parameters defined as sigma squared model to Twiss model parameters.
    The sigma squared model is defined as:

        sigma^2 = a + b*z + c*z^2

    The Twiss model is defined as:

        sigma^2 = epsilon *(beta - 2*alpha*z + ((1 + alpha^2)/beta)*z^2)

    Parameters
    ----------
        a: Numeric
            Parameter a of the sigma squared model, preferably in [mm^2].
        b: Numeric
            Parameter b of the sigma squared model, preferably in [mm].
        c: Numeric
            Parameter c of the sigma squared model, unitless.

    Returns
    -------
        tuple[Numeric, Numeric, Numeric]
            A tuple containing the Twiss parameters (epsilon, alpha, beta).
            The parameters are in the same units as a and b.
    """
    import numpy as np

    epsilonSquared = np.asarray(a) * np.asarray(c) - (np.asarray(b)**2) / 4
    if np.any(epsilonSquared < 0):
        error = ValueError("The calculated epsilon squared is negative. This may indicate that the input parameters do not represent a valid beam propagation model.")
        _logger.error(error)
        raise error

    epsilon = np.real(np.sqrt(epsilonSquared))
    alpha = -np.asarray(b) / (2 * epsilon)
    beta = np.asarray(a) / epsilon

    return (float(np.real(epsilon)), float(alpha), float(beta)) if np.isscalar(epsilon) else (epsilon.tolist(), alpha.tolist(), beta.tolist())


def readBeamModel(fileName: PathLike) -> dict:
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

    return beamModel


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
