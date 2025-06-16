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
