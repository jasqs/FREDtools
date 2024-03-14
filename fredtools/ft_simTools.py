def setFieldsFolderStruct(folderPath, RNfileName, folderName="FRED", overwrite=False, displayInfo=False):
    """Create a folder structure for each field in the treatment plan.

    The function creates a folder structure in a given `folderPath` for each field separately.
    The folder structure is in the form:

        folderPath/folderName:

                    / 1_Field2

                    / 2_Field3

                    / 3_Field1

                    ...

    The number at the beginning of the folder name is the delivery number
    and the number after `Field` is the ID of the field.

    Parameters
    ----------
    folderPath : path
        Path to a folder to create the structure.
    RNfileName : path
        Path to RN dicom file of a treatment plan.
    folderName : string, optional
        Name of the folder to create. (def. 'FRED')
    overwrite : bool, optional
        Determine if the folder should be overwritten.
        If true, then all the data in the existing folder will be
        removed. (def. False)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    path
        Path to created folder structure.
    """
    import shutil
    import os
    import fredtools as ft

    # check if folderPath exists
    if not os.path.exists(folderPath):
        raise FileNotFoundError(f"The folder {folderPath} dose not exist.")

    simFolder = os.path.join(folderPath, folderName)

    # check if simulation folder exists
    if os.path.exists(simFolder) and not overwrite:
        raise FileExistsError(f"The simulation folder {simFolder} already exists.")

    # remove simulation folder if exists and overwrite=True
    if os.path.exists(simFolder) and overwrite:
        shutil.rmtree(simFolder)

    # create simulation folder
    os.mkdir(simFolder)

    # read fields info from RN file and reset index
    fieldsInfo = ft.getRNFields(RNfileName, raiseWarning=False, displayInfo=False)
    fieldsInfo.reset_index(inplace=True)

    # create subfolders for fields
    for _, fieldInfo in fieldsInfo.iterrows():
        os.mkdir(os.path.join(simFolder, f"{fieldInfo.FDeliveryNo:d}_Field{fieldInfo.FNo:d}"))

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Created {:d} field folders in {:s}".format(len(fieldsInfo), simFolder))
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return simFolder


def readFREDStat(fileNameLogOut, displayInfo=False):
    """Read FRED simulation statistics information from the log file.

    The function reads some statistics information from a FRED run.out logfile.
    If some information is unavailable, then a NaN or numpy.np.nan is returned.

    Parameters
    ----------
    fileNameLogOut : string
        A string path to FRED output logfile (usually in out/log/run.out)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dict
        A dictionary with the read data.
    """
    import os
    import re
    import numpy as np
    import fredtools as ft

    def scaleUnit(unit):
        if unit == "ns":
            return 1e9
        if unit == "us":
            return 1e6
        if unit == "ms":
            return 1e3
        if unit == "s":
            return 1
        else:
            return np.nan

    def matchData(lineInclude, matching, startLine=1):
        lineFromFile = ft.getLineFromFile(lineInclude, fileNameLogOut, kind="first", startLine=startLine)
        if not lineFromFile:
            return "NaN"
        value = re.findall(matching, lineFromFile[1])
        if not value:
            return "NaN"
        return value[0]

    # check if file exists
    if not os.path.isfile(fileNameLogOut):
        raise ValueError(f"The file {fileNameLogOut} dose not exist.")

    simInfo = {
        "fredVersion": "NaN",
        "fredVersionDate": "NaN",
        "runConfig": np.nan,
        "runWallclockTime_s": np.nan,
        "primarySimulated": np.nan,
        "trackingRate_prim_s": np.nan,
        "trackTimePerPrimary_us": np.nan,
        "timingInitialization_s": np.nan,
        "timingPrimaryList_s": np.nan,
        "timingDeliveryChecking_s": np.nan,
        "timingGeometryChecking_s": np.nan,
        "timingTracking_s": np.nan,
        "timingWritingOutput_s": np.nan,
        "timingOther_s": np.nan,
    }

    simInfo["fredVersion"] = matchData(r"Version", r"Version\W+([\S+.]+)")
    simInfo["fredVersionDate"] = matchData(r"Version", r"Version.*([0-9]{4}\/[0-9]{2}\/[0-9]{2})")

    # check run config
    runningConfigLine = ft.getLineFromFile(r"Running config", fileNameLogOut, kind="first")
    if runningConfigLine:
        runningConfigTypes = re.findall(r"\w+", re.findall(r"Running config.*\((.*)\)", runningConfigLine[1])[0])
        runningConfigValues = re.findall(r"\d+", re.findall(r"Running config.*:(.*)", runningConfigLine[1])[0])
        for runningConfigType, runningConfigValue in zip(runningConfigTypes, runningConfigValues):
            simInfo["runConfig"+runningConfigType] = int(runningConfigValue)
        if "runConfigGPU" in simInfo.keys() and simInfo["runConfigGPU"] > 0:
            simInfo["runConfig"] = "{:d}xGPU".format(simInfo["runConfigGPU"])
        elif "runConfigPTHREADS" in simInfo.keys():
            simInfo["runConfig"] = "{:d}xCPU".format(simInfo["runConfigPTHREADS"])
        else:
            simInfo["runConfig"] = np.nan

    simInfo["runWallclockTime_s"] = float(matchData(r"Run wallclock time", rf"Run wallclock time:\W+({ft.re_number})"))
    simInfo["primarySimulated"] = float(matchData(r"Number of primary particles", rf"Number of primary particles:\W+({ft.re_number})"))
    simInfo["primarySimulated"] = int(simInfo["primarySimulated"]) if not np.isnan(simInfo["primarySimulated"]) else np.nan

    simInfo["trackingRate_prim_s"] = float(matchData(r"Tracking rate", rf"Tracking rate:\W+({ft.re_number})"))
    simInfo["trackingRate_prim_s"] = int(simInfo["trackingRate_prim_s"]) if not np.isnan(simInfo["trackingRate_prim_s"]) else np.nan

    simInfo["trackTimePerPrimary_us"] = float(matchData(r"Track time per primary", rf"Track time per primary:\W+({ft.re_number})"))
    simInfo["trackTimePerPrimary_us"] /= scaleUnit(matchData(r"Track time per primary", rf"Track time per primary\W+{ft.re_number}\W*(\w+)"))
    simInfo["trackTimePerPrimary_us"] *= 1E6

    timingSummaryStartLine = ft.getLineFromFile(r"^Timing summary", fileNameLogOut, kind="first")
    if timingSummaryStartLine:
        simInfo["timingInitialization_s"] = float(matchData(r"initialization", rf"initialization\W+({ft.re_number})", startLine=timingSummaryStartLine[0]))
        simInfo["timingInitialization_s"] /= scaleUnit(matchData(r"initialization", rf"initialization\W+{ft.re_number}\W*(\w+)", startLine=timingSummaryStartLine[0]))

        simInfo["timingPrimaryList_s"] = float(matchData(r"primary list", rf"primary list\W+({ft.re_number})", startLine=timingSummaryStartLine[0]))
        simInfo["timingPrimaryList_s"] /= scaleUnit(matchData(r"primary list", rf"primary list\W+{ft.re_number}\W*(\w+)", startLine=timingSummaryStartLine[0]))

        simInfo["timingDeliveryChecking_s"] = float(matchData(r"delivery checking", rf"delivery checking\W+({ft.re_number})", startLine=timingSummaryStartLine[0]))
        simInfo["timingDeliveryChecking_s"] /= scaleUnit(matchData(r"delivery checking", rf"delivery checking\W+{ft.re_number}\W*(\w+)", startLine=timingSummaryStartLine[0]))

        simInfo["timingGeometryChecking_s"] = float(matchData(r"geometry checking", rf"geometry checking\W+({ft.re_number})", startLine=timingSummaryStartLine[0]))
        simInfo["timingGeometryChecking_s"] /= scaleUnit(matchData(r"geometry checking", rf"geometry checking\W+{ft.re_number}\W*(\w+)", startLine=timingSummaryStartLine[0]))

        simInfo["timingTracking_s"] = float(matchData(r"tracking", rf"tracking\W+({ft.re_number})", startLine=timingSummaryStartLine[0]))
        simInfo["timingTracking_s"] /= scaleUnit(matchData(r"tracking", rf"tracking\W+{ft.re_number}\W*(\w+)", startLine=timingSummaryStartLine[0]))

        simInfo["timingWritingOutput_s"] = float(matchData(r"writing output", rf"writing output\W+({ft.re_number})", startLine=timingSummaryStartLine[0]))
        simInfo["timingWritingOutput_s"] /= scaleUnit(matchData(r"writing output", rf"writing output\W+{ft.re_number}\W*(\w+)", startLine=timingSummaryStartLine[0]))

        simInfo["timingOther_s"] = float(matchData(r"other", rf"other\W+({ft.re_number})", startLine=timingSummaryStartLine[0]))
        simInfo["timingOther_s"] /= scaleUnit(matchData(r"other", rf"other\W+{ft.re_number}\W*(\w+)", startLine=timingSummaryStartLine[0]))

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# FRED Version: {:s}".format(simInfo["fredVersion"]))
        print("# FRED Version Date: {:s}".format(simInfo["fredVersionDate"]))
        runConfigKeys = [key for key in simInfo.keys() if re.search("runConfig.+", key)]
        if runConfigKeys:
            runConfigValues = [simInfo[runConfigKey] for runConfigKey in runConfigKeys]
            runConfigKeys = [runConfigKey.replace("runConfig", "") for runConfigKey in runConfigKeys]
            print(f"# Run Config ({','.join(runConfigKeys)}): {str(runConfigValues).replace('[','').replace(']','')}")
        print("# Run Config: {}".format(simInfo["runConfig"]))
        print("# Run Wall clock Time: {:.2f} s".format(simInfo["runWallclockTime_s"]))
        print("# Average Track Time Per Primary: {:5f} us".format(simInfo["trackTimePerPrimary_us"]))
        print("# Average Tracking Rate: {:.3E} prim/s".format(simInfo["trackingRate_prim_s"]))
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return simInfo


def readBeamModel(fileName):
    """Read beam model from a YAML file.

    The function reads the beam model parameters from a YAML beam model file.
    The beam model must be defined as a dictionary at least including keys:

        - "BM Description": a dictionary with the beam model descriptions. At least the keys "name" and "creationDate" are required and will be added automatically if not provided.
        - "BM Energy": a pandas DataFrame of beam energetic and propagation parameters, with at least columns:

            -  *Energy parameters*: nominal energy ('nomEnergy'), energy in Monte Carlo ('Energy') and energy spread in Monte Carlo ('dEnergy')
            -  *Dosimetric parameter*: scaling factor to recalculate MU to number of protons ('scalingFactor')
            -  *Optical Parameters*: Alpha, beta and epsilon describing the beam emittance in X/Y directions ('alphaX', 'betaX', 'epsilonX', 'alphaY', 'betaY' and 'epsilonY')

        - "BM RangeShifters": a dictionary with the range shifters and its parameters, like position, thickness, meterial, etc.
        - "BM Materials": a dictionary of the materials and its parameters, like density, composition, etc.

    Additionally, other parameters can be defined.

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

    # load beam model from file
    with open(fileName, "r") as yaml_file:
        beamModel = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    # check if all required sections are present in the beam model
    if not {"BM Description", "BM Energy", "BM RangeShifters", "BM Materials"}.issubset(beamModel.keys()):
        raise ValueError(f"Missing sections in the beam model loaded from {fileName}\nThe beam model must include at least 'BM Description', 'BM Energy', 'BM RangeShifters' and 'BM Materials' sections.")

    # convert all dataFrame-like lists of strings to dataFrame
    for key in beamModel.keys():
        if isinstance(beamModel[key], list):
            if ("row" in beamModel[key][-1]) and ("column" in beamModel[key][-1]):  # the key is a pandas DataFrame
                beamModel[key] = pd.read_csv(StringIO("\n".join(beamModel[key][:-1])), sep='\s+')
                if "nomEnergy" in beamModel[key].columns:
                    beamModel[key].set_index("nomEnergy", inplace=True)

    # validate if required columns exist in BM Energy
    if not {"Energy", "dEnergy", "scalingFactor", "alphaX", "betaX", "epsilonX", "alphaY", "betaY", "epsilonY"}.issubset(beamModel["BM Energy"].columns):
        raise ValueError(f"Missing columns or wrong column names of 'BM Energy' when loading beam model from {fileName}.")
    # validate if there are any missing (None, NaN) values in BM Energy
    if np.any((beamModel["BM Energy"]).isna()):
        raise ValueError(f"Missing values for some records in the 'BM Energy' for the beam model loaded from {fileName}.")

    return beamModel


def writeBeamModel(beamModel, fileName):
    """Write beam model to YAML.

    The function writes the beam model parameters to a beam model file in YAML format.
    The beam model must be defined as a dictionary at least including keys:

        - "BM Description": a dictionary with the beam model descriptions. At least the keys "name" and "creationDate" are required and will be added automatically if not provided.
        - "BM Energy": a pandas DataFrame of beam energetic and propagation parameters, with at least columns:

            -  *Energy parameters*: nominal energy ('nomEnergy'), energy in Monte Carlo ('Energy') and energy spread in Monte Carlo ('dEnergy')
            -  *Dosimetric parameter*: scaling factor to recalculate MU to number of protons ('scalingFactor')
            -  *Optical Parameters*: Alpha, beta and epsilon describing the beam emittance in X/Y directions ('alphaX', 'betaX', 'epsilonX', 'alphaY', 'betaY' and 'epsilonY')

        - "BM RangeShifters": a dictionary with the range shifters and its parameters, like position, thickness, meterial, etc.
        - "BM Materials": a dictionary of the materials and its parameters, like density, composition, etc.

    Additionally, other parameters can be defined as keys and will be saved. If a value of a given key is a pandas DataFrame,
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

    # check if all required sections are present in the beam model
    if not {"BM Energy", "BM RangeShifters", "BM Materials"}.issubset(beamModelSave.keys()):
        raise ValueError(f"Missing sections (keys) in the beam model.\nThe beam model must include at least 'BM Description', 'BM Energy', 'BM RangeShifters' and 'BM Materials' sections.")

    # add "BM Description" to the beam model and add/modify values and/or keys order
    if not "BM Description" in beamModelSave.keys():
        beamModelSave["BM Description"] = {}
    BMDescription = {}
    if "name" in beamModelSave["BM Description"].keys():
        BMDescription["name"] = beamModelSave["BM Description"].pop("name")
    else:
        BMDescription["name"] = os.path.splitext(os.path.basename(fileName))[0]
    if "creationTime" in beamModelSave["BM Description"].keys():
        BMDescription["creationTime"] = beamModelSave["BM Description"].pop("creationTime")
    else:
        BMDescription["creationTime"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    BMDescription.update(beamModelSave["BM Description"])
    beamModelSave["BM Description"] = BMDescription

    # validate if required columns exist in BM Energy
    if not {"Energy", "dEnergy", "scalingFactor", "alphaX", "betaX", "epsilonX", "alphaY", "betaY", "epsilonY"}.issubset(beamModelSave["BM Energy"].columns):
        raise ValueError(f"Missing columns or wrong column names of 'BM Energy' section in the beam model.")
    # validate if there are any missing (None, NaN) values in BM Energy
    if np.any((beamModelSave["BM Energy"]).isna()):
        raise ValueError(f"Missing values for some records in the 'BM Energy' section of the beam model.")

    # beamModelSave["BM Energy"] = beamModelSave["BM Energy"].to_dict()

    # convert all dataFrames to a nicely formated list of strings
    for key in beamModel.keys():
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


def interpolateBeamModel(beamModel, nomEnergy, interpolation="linear", splineOrder=3):
    """Interpolate beam model for a given nominal energy.

    The function interpolates all the beam model parameters for a given nominal energies
    which must be in range of the defined nominal energies in the beam model. The possible
    interpolation methods are 'nearest', 'linear' or 'spline' with order in range 0-5.

    Parameters
    ----------
    beamModel : DataFrame
        Beam model defined in a pandas DataFrame.
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
    if np.isscalar(nomEnergy):
        nomEnergy = [nomEnergy]

    # validate the interpolation method
    if not interpolation.lower() in ["linear", "nearest", "spline"]:
        raise ValueError(f"Interpolation type '{interpolation}' cannot be recognized. Only 'linear', 'nearest' and 'spline' are supported.")

    interpolation = interpolation.lower()

    # set the proper interpolation method for spline
    if interpolation == "spline":
        if splineOrder > 5 or splineOrder < 0:
            raise ValueError(f"Spline order must be in range 0-5.")
        else:
            interpolation = splineOrder

    # check if all given nomEnergy are in range of the beam model
    if np.array(nomEnergy).min() < beamModel.index.min() or np.array(nomEnergy).max() > beamModel.index.max():
        raise ValueError(
            f"The range of nominal energies for interpolation is {np.array(nomEnergy).min()}-{np.array(nomEnergy).max()} and it is outside the beam model nominal energy range {beamModel.index.min()}-{beamModel.index.max()}."
        )

    # interpolate each parameter of the beam model with a given interpolation method
    beamModelEnergyInterp = {}
    beamModelEnergyInterp["nomEnergy"] = nomEnergy
    for key in beamModel.keys():
        beamModelEnergyInterp[key] = interp1d(beamModel.index, beamModel[key], kind=interpolation)(nomEnergy).tolist()
    beamModelEnergyInterp = pd.DataFrame(beamModelEnergyInterp)

    return beamModelEnergyInterp


def getFREDVersions():
    """List the installed FRED varions.

    The function lists the FRED versions installed on the machine.

    Returns
    -------
    List of strings
        List of FRED versions.

    See Also
    --------
    checkFREDVersion : check if the FRED version is installed.
    """
    import subprocess
    import re

    FREDrunCommand = ["fred", "-listVers"]
    runFredProc = subprocess.Popen(r" ".join(FREDrunCommand), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="UTF-8")
    stdout, stderr = runFredProc.communicate()
    runFredProc.wait()
    if stderr or "Error" in stdout:
        raise RuntimeError(stderr if stderr else stdout)
    return [x.strip() for x in stdout.split("\n") if re.findall(r"\d.\d+.\d+", x)]


def checkFREDVersion(version):
    """Check if the FRED version is installed.

    The function validates if the version of FRED, given by the parameter
    is installed on the machine.

    Parameters
    ----------
    version : str
        Version in format #.#.#.

    Returns
    -------
    bool
        True if the version is installed.

    See Also
    --------
    getFREDVersions : list the installed FRED varions.
    """
    import re
    import fredtools as ft

    if not isinstance(version, str) or not re.findall(r"\d.\d+.\d+", version):
        raise ValueError(f"The version must be a string in format #.#.#, for instance '3.58.4'")
    fredVersions = ft.getFREDVersions()

    return any([version in fredVersion for fredVersion in fredVersions])


def getFREDVersion(version=""):
    """Get the full FRED version name.

    The function checks if the `version` of FRED is installed
    and returns its full version name.

    Parameters
    ----------
    version : str
        Version in format #.#.#.

    Returns
    -------
    str
        Full version name returned by FRED.

    See Also
    --------
    getFREDVersions : list the installed FRED varions.
    """
    import subprocess
    import fredtools as ft

    if version and not ft.checkFREDVersion(version):
        raise ValueError(f"No FRED v. {version} installed on the machine.\nAvailable FRED versions:\n" + "\n".join(ft.getFREDVersions()))

    if version:
        FREDrunCommand = ["fred", f"-useVers {version}", "-v"]
    else:
        FREDrunCommand = ["fred", "-v"]
    runFredProc = subprocess.Popen(r" ".join(FREDrunCommand), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="UTF-8")
    stdout, stderr = runFredProc.communicate()
    runFredProc.wait()
    if stderr or "Error" in stdout:
        raise RuntimeError(stderr if stderr else stdout)

    return stdout.split("\n")[0]


def runFRED(fredInpFileName, version="", params=[], displayInfo=False):
    """Run FRED simulation.

    The function runs FRED simulation defined by
    the FRED input file name in the given FRED version.

    Parameters
    ----------
    fredInpFileName : path
        Path string to FRED input file. Usually, it is called `fred.inp`.
    version : str, optional
        Version of FRED in format #.#.#. If no version is given
        then the current version installed is used. (def. "")
    params : str or list of strings, optional
        Additional parameters to FRED engine, for instance
        ["-C", "-V5", "-nogpu"] etc. (def. [])
    displayInfo : bool
        Displays a summary of the function results. (def. False)

    Returns
    -------
    subprocess stdout
        Standard output of the subprocess method in the form of
        list of string lines.

    See Also
    --------
    readFREDStat : read FRED simulation statistics information from logfile.
    checkFREDVersion : check if the FRED version is installed.
    getFREDVersions : list the installed FRED varions.
    """
    import os
    import subprocess
    import fredtools as ft

    # check if the version is available
    if version and not ft.checkFREDVersion(version):
        raise ValueError(f"No FRED v. {version} installed on the machine.\nAvailable FRED versions:\n" + "\n".join(ft.getFREDVersions()))

    # get full fred version name
    fredVersName = ft.getFREDVersion(version).replace("fred", "FRED")

    # check if the fred.inp exists
    if not os.path.exists(fredInpFileName):
        raise ValueError(f"The file '{fredInpFileName}' dose not exist.")

    # get absolute folder name sim. file name
    fredInpFileName = os.path.abspath(fredInpFileName)
    simFolderName = os.path.dirname(fredInpFileName)
    fredInpFileName = os.path.basename(fredInpFileName)

    # run fred sim
    FREDrunCommand = ["fred"]

    FREDrunCommand.append(f"-useVers {version}") if version else None
    params = [params] if isinstance(params, str) else params
    FREDrunCommand.extend(params) if params else None
    FREDrunCommand.append(f"-f {fredInpFileName}")
    FREDrunCommand = r" ".join(FREDrunCommand)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# {fredVersName}")
        print(f"# Running FRED sim. in folder {simFolderName}")
        print(f"# FRED command: {FREDrunCommand}")

    runFredProc = subprocess.Popen(FREDrunCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="UTF-8", cwd=simFolderName + os.sep, universal_newlines=True)
    stdout, stderr = runFredProc.communicate()
    runFredProc.wait()

    if displayInfo:
        # check fred sim
        if "Run wallclock time" in stdout:
            FREDSimStat = ft.readFREDStat(os.path.join(simFolderName, "out/log/run.out"))
            print("# FRED sim. done in {:.0f} s with {:.2E} prim/s".format(FREDSimStat["runWallclockTime_s"], FREDSimStat["trackingRate_prim_s"]))
        else:
            print(f"# ERROR in FRED sim. Check {simFolderName}/out/log/run.out")
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return stdout.splitlines()


def readGATE_HITSActor(fileName):
    """read GATE hits data for active volume.

    The function reads hits results of GATE active volume saved
    to numpy pickle (.npy) or root (.root) file. All the columns
    are read but some of them are renamed:

        -  *ds* is a step length in [cm]
        -  *Edep* is deposited energy in [MeV]
        -  *PDGCode* is the same as PDG encoding [1]_

    Parameters
    ----------
    fileName : path
        Path string to .npy or .root file.

    Returns
    -------
    pandas DataFrame
        Dataframe with the data.

    See Also
    --------
    readFREDStat : read FRED simulation statistics information from logfile.

    References
    ----------
    .. [1] `Monte Carlo Particle Numbering Scheme <https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf>`_
    """
    import pandas as pd
    import numpy as np
    import os
    import uproot

    # read numpy array and convert to DataFrame
    if os.path.splitext(fileName)[-1] in [".npy"]:
        try:
            hitsActor = np.load(fileName)
            hitsActor = pd.DataFrame(hitsActor)
        except ValueError:
            raise ValueError(f"Can not read file {fileName} as numpy pickle.")
    elif os.path.splitext(fileName)[-1] in [".root"]:
        try:
            hitsActor = uproot.open(fileName)
            hitsActor = hitsActor[hitsActor.keys()[0]]
            hitsActor = pd.DataFrame(hitsActor.arrays(hitsActor.keys(), library="np"))
        except ValueError:
            raise ValueError(f"Can not read file {fileName} as root file.")
    else:
        raise ValueError(f"Can not recognise type of the file {fileName}. Only 'root' or 'npy' extentions are possible.")

    # uncapitalize each column name
    def uncapitalize(word):
        if len(word) > 0:
            return word[0].lower() + word[1:]

    columnNames = hitsActor.columns
    columnNames = [uncapitalize(columnName) for columnName in columnNames]
    hitsActor.columns = columnNames

    # rename columns
    hitsActor.rename(columns={"edep": "Edep", "pDGEncoding": "PDGCode"}, inplace=True)

    # convert byte string to string
    for keyName, keyType in hitsActor.dtypes.items():
        if keyType == "object":
            hitsActor[keyName] = hitsActor[keyName].where(hitsActor[keyName].apply(type) != bytes, hitsActor[keyName].str.decode("utf-8"))

    return hitsActor


def readGATE_PSActor(fileName):
    """read GATE hits data for active volume.

    The function reads hits results of GATE active volume saved
    to numpy pickle (.npy) or root (.root) file. All the columns
    are read but some of them are renamed:

        -  *ds* is a step length in [cm]
        -  *Edep* is deposited energy in [MeV]
        -  *PDGCode* is the same as PDG encoding [2]_

    Parameters
    ----------
    fileName : path
        Path string to .npy or .root file.

    Returns
    -------
    pandas DataFrame
        Dataframe with the data.

    See Also
    --------
    readFREDStat : read FRED simulation statistics information from logfile.

    References
    ----------
    .. [2] `Monte Carlo Particle Numbering Scheme <https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf>`_
    """
    import pandas as pd
    import numpy as np
    import os
    import uproot

    # read numpy array and convert to DataFrame
    if os.path.splitext(fileName)[-1] in [".npy"]:
        try:
            psActor = np.load(fileName)
            psActor = pd.DataFrame(psActor)
        except ValueError:
            raise ValueError(f"Can not read file {fileName} as numpy pickle.")
    elif os.path.splitext(fileName)[-1] in [".root"]:
        try:
            psActor = uproot.open(fileName)
            psActor = psActor[psActor.keys()[0]]
            psActor = pd.DataFrame(psActor.arrays(psActor.keys(), library="np"))
        except ValueError:
            raise ValueError(f"Can not read file {fileName} as root file.")
    else:
        raise ValueError(f"Can not recognise type of the file {fileName}. Only 'root' or 'npy' extentions are possible.")

    # uncapitalize each column name
    def uncapitalize(word):
        if len(word) > 0:
            return word[0].lower() + word[1:]

    columnNames = psActor.columns
    columnNames = [uncapitalize(columnName) for columnName in columnNames]
    psActor.columns = columnNames

    # rename columns
    psActor.rename(columns={"ekine": "Ekine", "edep": "Edep", "ekpost": "EkinePost", "ekpre": "EkinePre", "pDGCode": "PDGCode"}, inplace=True)

    # sort columns
    sortOrder = []
    for columnNameScheme in ["ID", "PDG", "Ekine", "Edep", "DEDX", "Length"]:  # sort by name
        sortOrder += np.where([columnNameScheme in columnName for columnName in psActor.columns])[0].tolist()
    for sortIdx in range(psActor.columns.size):  # add missing columns
        if not sortIdx in sortOrder:
            sortOrder.append(sortIdx)
    psActor = psActor[psActor.columns[sortOrder]]

    # convert byte string to string
    for keyName, keyType in psActor.dtypes.items():
        if keyType == "object":
            psActor[keyName] = psActor[keyName].where(psActor[keyName].apply(type) != bytes, psActor[keyName].str.decode("utf-8"))

    return psActor


def readGATEStat(fileNameLogOut, displayInfo=False):
    """Read GATE simulation statistics information from Simulation Statistic Actor.

    The function reads some statistics information from the
    GATE Simulation Statistic Actor output [3]_.

    Parameters
    ----------
    fileNameLogOut : string
        A string path to GATE  Simulation Statistic Actor output.
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    dict
        A dictionary with the read data.

    References
    ----------
    .. [3] `GATE manual for Simulation Statistic Actor <https://opengate.readthedocs.io/en/latest/tools_to_interact_with_the_simulation_actors.html?highlight=%20Simulation%20Statistic%20Actor#id7>`_
    """
    import re
    import fredtools as ft

    simStat = {}
    with open(fileNameLogOut, "r") as file_h:
        for line in file_h.readlines():
            key = re.findall(r"# (\w+)", line)[0]
            if key in ["StartDate", "EndDate", "TPS", "SPS"]:
                continue
            value = re.findall(r"=\s+(.+)", line)[0]
            if "Number" in key:
                simStat[key] = int(value)
            else:
                simStat[key] = float(value)
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Number of events: {:d}".format(simStat["NumberOfEvents"]))
        print("# Elapsed Time (total):   {:2f} s".format(simStat["ElapsedTime"]))
        print("# Elapsed Time (no init): {:2f} s".format(simStat["ElapsedTimeWoInit"]))
        print("# Average Tracking Rate:  {:.3E} prim/s".format(simStat["PPS"]))
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return simStat
