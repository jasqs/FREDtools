from fredtools._typing import *
# from fredtools import beamModel as beamModelClass
from fredtools import getLogger
_logger = getLogger(__name__)


def importBeamModel(beamModel: beamModelClass, fitOrder: PositiveInt | Literal["auto"] = "auto"):
    import numpy as np
    from opengate.contrib.beamlines.ionbeamline import BeamlineModel
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

    """
    beamModelEnergyFRED.epsilon in [mm * rad]
    beamModelEnergyFRED.beta in [mm]
    beamModelEnergyFRED.alpha in [-]
    """

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


def addMaterials(sim, beamModel: beamModelClass):
    from opengate import g4_units

    for materialName, materialProperties in beamModel.materials.iterrows():
        if materialProperties.basedOn == "PMMA":
            _logger.debug(f"Adding material '{materialName}' based on PMMA with density {materialProperties.density} g/cm3 and mean excitation energy {materialProperties.Ipot} eV.")
            sim.volume_manager.material_database.add_material_weights(materialName, ["H", "C", "O"], [0.080541, 0.599846, 0.319613], materialProperties.density * g4_units.g_cm3)
            sim.volume_manager.find_or_build_material(materialName).GetIonisation().SetMeanExcitationEnergy(materialProperties.Ipot * g4_units.eV)

        else:
            error = ValueError(f"Material {materialName} is based on {materialProperties.basedOn}, which is not implemented.")
            _logger.error(error)
            raise error
