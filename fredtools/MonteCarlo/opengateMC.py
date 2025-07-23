from fredtools._typing import *
from fredtools.MonteCarlo.beamModel import beamModel
from fredtools import getLogger
_logger = getLogger(__name__)

try:
    import opengate as gate  # type: ignore

    # get units
    m = gate.g4_units.m
    km = gate.g4_units.km
    cm = gate.g4_units.cm
    mm = gate.g4_units.mm
    um = gate.g4_units.um
    MeV = gate.g4_units.MeV
    eV = gate.g4_units.eV
    deg = gate.g4_units.deg
    mrad = gate.g4_units.mrad
    rad = gate.g4_units.rad
    s = gate.g4_units.s
    gcm3 = gate.g4_units.g_cm3

except ModuleNotFoundError:
    _logger.info("The opengate package is not available and the behaviour of the functions in the opengateMC FREDtools module might be unexpected.\nInstall the opengate first to use the functions in opengateMC module.")


def importBeamModel(beamModel: beamModel, fitOrder: PositiveInt | Literal["auto"] = "auto"):
    """Import a beam model from a fredtools beamModel object to an OpenGATE BeamlineModel.

    The function converts the beam model parameters from a fredtools beamModel object to an OpenGATE BeamlineModel object.
    The beam model parameters include the radiation type, distances from the nozzle to the isocenter, and the beam parameters for different energies.
    The beam parameters include the energy, energy spread, beam size, beam divergence, emittance, and convergence.
    The function also fits the beam parameters to a polynomial of a specified order.
    The fitted coefficients are stored in the OpenGATE BeamlineModel object.

    Parameters
    ----------
    beamModel : fredtools.beamModel
        The beam model to be imported. It must be an instance of fredtools beamModel.
    fitOrder : PositiveInt or Literal["auto"], optional
        The order of the polynomial fit for the beam parameters. If "auto", the fit order is set to the number of energies minus one. (def. "auto")

    Returns
    -------
    BeamlineModel : opengate.contrib.beamlines.ionbeamline.BeamlineModel
        The OpenGATE BeamlineModel object containing the imported beam model parameters.
    """
    import numpy as np
    from opengate.contrib.beamlines.ionbeamline import BeamlineModel  # type: ignore
    import fredtools as ft
    import pandas as pd

    if not isinstance(beamModel, ft.beamModel):
        error = TypeError(f"beamModel must be an instance of ft.beamModel, got {type(beamModel)}")
        _logger.error(error)
        raise error

    BeamlineModel = BeamlineModel()
    BeamlineModel.radiation_types = beamModel.radiationType

    # Nozzle entrance to Isocenter distance
    BeamlineModel.distance_nozzle_iso = beamModel.sourceToAxisDistance  # [mm]
    # SMX to Isocenter distance
    BeamlineModel.distance_stearmag_to_isocenter_x = beamModel.spreadingDeviceDistance[0]
    # SMY to Isocenter distance
    BeamlineModel.distance_stearmag_to_isocenter_y = beamModel.spreadingDeviceDistance[1]

    energyFit = np.linspace(np.min(beamModel.nomEnergies), np.max(beamModel.nomEnergies), 1000)
    beamModelGate = beamModel.getGateParams(sourceToAxisDistance=beamModel.sourceToAxisDistance, nomEnergy=energyFit)

    # beam convergence flag
    if np.unique(beamModelGate.convergenceX).size != 1 or np.unique(beamModelGate.convergenceY).size != 1:
        error = ValueError("Beam convergence must be constant for all energies")
        _logger.error(error)
        raise error

    BeamlineModel.conv_x = int(np.unique(beamModelGate.convergenceX)[0])
    BeamlineModel.conv_y = int(np.unique(beamModelGate.convergenceY)[0])

    # fit order
    if fitOrder == "auto":
        fitOrder = len(beamModelGate.index) - 1
    elif not isinstance(fitOrder, int) or fitOrder < 1:
        error = ValueError(f"fitOrder must be a positive integer, got {fitOrder}")
        _logger.error(error)
        raise error

    BeamlineModel.energy_mean_coeffs = np.polyfit(beamModelGate.index, beamModelGate.Energy, fitOrder).tolist()
    BeamlineModel.energy_spread_coeffs = np.polyfit(beamModelGate.index, beamModelGate.dEnergy, fitOrder).tolist()
    BeamlineModel.sigma_x_coeffs = np.polyfit(beamModelGate.index, beamModelGate.sigmaX, fitOrder).tolist()
    BeamlineModel.sigma_y_coeffs = np.polyfit(beamModelGate.index, beamModelGate.sigmaY, fitOrder).tolist()
    BeamlineModel.theta_x_coeffs = np.polyfit(beamModelGate.index, beamModelGate.thetaX, fitOrder).tolist()
    BeamlineModel.theta_y_coeffs = np.polyfit(beamModelGate.index, beamModelGate.thetaY, fitOrder).tolist()
    BeamlineModel.epsilon_x_coeffs = np.polyfit(beamModelGate.index, beamModelGate.epsilonX, fitOrder).tolist()
    BeamlineModel.epsilon_y_coeffs = np.polyfit(beamModelGate.index, beamModelGate.epsilonY, fitOrder).tolist()

    return BeamlineModel


def addMaterials(sim, beamModel: beamModel):
    """Add materials from a fredtools beamModel to an OpenGATE simulation.

    The function iterates over the materials defined in the beamModel and adds them to the OpenGATE simulation.
    The mean excitation energy (ionization potential) is set for each material based on the properties defined in the beamModel.

    Parameters
    ----------
    sim : opengate.simulation.Simulation
        The OpenGATE simulation object to which the materials will be added.
    beamModel : fredtools.beamModel
        The beam model containing the materials to be added. It must be an instance of fredtools beamModel.
    """
    from opengate import g4_units  # type: ignore

    for materialName, materialProperties in beamModel.materials.iterrows():
        if materialProperties.basedOn == "PMMA":
            _logger.debug(f"Adding material '{materialName}' based on PMMA with density {materialProperties.density} g/cm3 and mean excitation energy {materialProperties.Ipot} eV.")
            sim.volume_manager.material_database.add_material_weights(materialName, ["H", "C", "O"], [0.080541, 0.599846, 0.319613], materialProperties.density * g4_units.g_cm3)
            sim.volume_manager.find_or_build_material(materialName).GetIonisation().SetMeanExcitationEnergy(materialProperties.Ipot * g4_units.eV)

        else:
            error = ValueError(f"Material {materialName} is based on {materialProperties.basedOn}, which is not implemented.")
            _logger.error(error)
            raise error


def generateTimeStamps(spotsInfo: DataFrame, timeSpotDuration: float = 0.1, timeBetweenSpots: float = 0.0, timeBetweenFields: float = 30) -> tuple[list[DateTime], list[DateTime]]:
    """Add start and stop time for each spot in the spotsInfo DataFrame.

    The function calculates the start and stop times for each spot based on the provided duration of a single spot and the time between fields.
    It returns a DataFrame with the added time information.

    Parameters
    ----------
    spotsInfo : DataFrame
        The DataFrame containing spot information.
    timeSpotDuration : float, optional
        The duration of a single spot in seconds (def. 0.1).
    timeBetweenSpots : float, optional
        The time between consecutive spots in seconds (def. 0.0).
    timeBetweenFields : float, optional
        The time between fields in seconds (def. 30).

    Returns
    -------
    tuple[list[DateTime], list[DateTime]]
        A tuple containing two lists: the start times and stop times for each spot.
    """
    from datetime import timedelta

    timesStart = []
    timesStop = []
    current_time = 0

    for _, spotsInfoField in spotsInfo.groupby(by="FDeliveryNo"):
        spotsNo = len(spotsInfoField)
        for i in range(spotsNo):
            timesStart.append(timedelta(seconds=current_time + i * (timeSpotDuration + timeBetweenSpots)))
            timesStop.append(timedelta(seconds=current_time + i * (timeSpotDuration + timeBetweenSpots) + timeSpotDuration))
        current_time += spotsNo * (timeSpotDuration + timeBetweenSpots) + timeBetweenFields

    return timesStart, timesStop


def addIonPencilBeamSources(sim, spotsInfo: DataFrame, beamModel: beamModel, primNo: PositiveInt, CPUNo: PositiveInt) -> DataFrame:

    from datetime import datetime, timedelta
    from scipy.spatial.transform import Rotation
    from fredtools import calcRaysVectors

    if len(spotsInfo.loc[spotsInfo.PBMU == 0]) != 0:
        _logger.debug(f"There are spots with PBMU = 0 ({len(spotsInfo.loc[spotsInfo.PBMU == 0])} out of {len(spotsInfo)}) which will be ignored in the simulation.")
        spotsInfoMC = spotsInfo.loc[spotsInfo.PBMU != 0].copy()
    else:
        spotsInfoMC = spotsInfo.copy()

    # calculate MC parameters for each spot
    spotsInfoMCInterp = beamModel.getGateParams(-beamModel.sourceToAxisDistance, spotsInfoMC.PBnomEnergy).reset_index(drop=True)

    # merge interpolated parameters into spotsInfoMC
    spotsInfoMC.reset_index(drop=True, inplace=True)
    spotsInfoMC = spotsInfoMC.join(spotsInfoMCInterp[["Energy", "dEnergy", "scalingFactor", "sigmaX", "sigmaY", "thetaX", "thetaY", "epsilonX", "epsilonY", "convergenceX", "convergenceY"]])

    # calculate number of primaries for each spot
    spotsInfoMC["PBPrimNo"] = spotsInfoMC.PBMU * spotsInfoMC.scalingFactor

    # calculate gantry rotation for each spot
    spotsInfoMC["PBGantryRotation"] = [Rotation.from_euler("z", FGantryAngle, degrees=True) for FGantryAngle in spotsInfoMC.FGantryAngle]  # gantry rotates around Z axis

    # calculate translation and rotation of each spot
    """The calculation is based on the ray position and direction versor, which assumes the ray basic direction along +Z axis. After recalculation below, the basic beam direction is along +Y axis."""
    PBPos = spotsInfoMC[["PBPosX", "PBPosY"]].copy()
    PBPos["PBPosZ"] = 0
    PBPos = PBPos.to_numpy() * np.array([1, -1, 1])
    raysPosition, raysVersor = calcRaysVectors(PBPos, SAD=beamModel.spreadingDeviceDistance)
    raysTranslation = Rotation.from_euler("x", -90, degrees=True).apply(raysPosition)  # apply rotation to rays translation to get basic beam direction along +Y axis
    raysTranslation = [PBGantryRotation.apply(rayTranslation) for PBGantryRotation, rayTranslation in zip(spotsInfoMC["PBGantryRotation"], raysTranslation)]  # apply gantry rotation to rays translation
    raysRotation = [Rotation.from_euler("x", -90, degrees=True) * Rotation.align_vectors(rayVersor, [[0, 0, 1]])[0] for rayVersor in raysVersor]  # calculate rays rotation from ray direction versor
    raysRotation = spotsInfoMC["PBGantryRotation"] * raysRotation  # apply gantry rotation to rays rotation
    spotsInfoMC["rayTranslation"] = raysTranslation
    spotsInfoMC["rayRotation"] = raysRotation

    # generate IonPencilBeamSource objects

    for idx, spotInfoMC in spotsInfoMC.iterrows():
        PBsource = sim.add_source("IonPencilBeamSource", f"PBsource_{idx}")
        PBsource.attached_to = "world"
        PBsource.n = primNo / CPUNo
        # set energy
        PBsource.energy.type = "gauss"
        PBsource.energy.mono = spotInfoMC.Energy
        PBsource.energy.sigma_gauss = spotInfoMC.dEnergy
        PBsource.particle = beamModel.radiationType
        # PBsource.position.type = "disc"
        # set optics parameters
        PBsource.direction.partPhSp_x = spotInfoMC[["sigmaX", "thetaX", "epsilonX"]].tolist() + ([1] if spotInfoMC["convergenceX"] else [0])
        PBsource.direction.partPhSp_y = spotInfoMC[["sigmaY", "thetaY", "epsilonY"]].tolist() + ([1] if spotInfoMC["convergenceY"] else [0])
        # set position and rotation
        PBsource.position.translation = spotInfoMC.rayTranslation
        PBsource.position.rotation = spotInfoMC.rayRotation.as_matrix()

        # set time parameters
        PBsource.start_time = spotInfoMC.timeStart.total_seconds() * s
        PBsource.end_time = spotInfoMC.timeStop.total_seconds() * s
        # set weight
        PBsource.weight = spotInfoMC.PBPrimNo / primNo

    return spotsInfoMC
