from fredtools._typing import *
from fredtools import getLogger
_logger = getLogger(__name__)


def setFieldsFolderStruct(folderPath: PathLike, RNfileName: PathLike, folderName: str = "FRED", overwrite: bool = False, displayInfo: bool = False) -> PathLike:
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
        _logger.info("Created {:d} field folders in {:s}".format(len(fieldsInfo), simFolder))

    return simFolder


def readFREDStat(fileName: PathLike, displayInfo: bool = False) -> DottedDict:
    """Read FRED simulation statistics information from the log file.

    The function reads some statistics information from a FRED run.out logfile.
    If some information is unavailable, then a NaN or numpy.np.nan is returned.

    Parameters
    ----------
    fileName : string
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

    def scaleUnit(unit: str) -> Numberic:
        match unit:
            case "ns":
                return 1e9
            case "us":
                return 1e6
            case "ms":
                return 1e3
            case "s":
                return 1
            case _:
                return np.nan

    def matchData(lineInclude, matching, startLine=1):
        lineFromFile = ft.getLineFromFile(lineInclude, fileName, kind="first", startLine=startLine)
        if not lineFromFile:
            return "NaN"
        value = re.findall(matching, lineFromFile[1])
        if not value:
            return "NaN"
        return value[0]

    # check if file exists
    if not os.path.isfile(fileName):
        error = FileNotFoundError(f"The file {fileName} dose not exist.")
        _logger.error(error)
        raise error

    simInfo = DottedDict({"fredVersion": "NaN",
                          "fredVersionDate": "NaN",
                          "runConfig": "NaN",
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
                          })

    simInfo.fredVersion = matchData(r"Version", r"Version\W+([\S+.]+)")
    simInfo.fredVersionDate = matchData(r"Version", r"Version.*([0-9]{4}\/[0-9]{2}\/[0-9]{2})")

    # check run config
    runningConfigLine = ft.getLineFromFile(r"Running config", fileName, kind="first")
    if runningConfigLine:
        runningConfigTypes = re.findall(r"\w+", re.findall(r"Running config.*\((.*)\)", runningConfigLine[1])[0])
        runningConfigValues = re.findall(r"\d+", re.findall(r"Running config.*:(.*)", runningConfigLine[1])[0])
        for runningConfigType, runningConfigValue in zip(runningConfigTypes, runningConfigValues):
            simInfo["runConfig" + runningConfigType] = int(runningConfigValue)
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

    timingSummaryStartLine = ft.getLineFromFile(r"^Timing summary", fileName, kind="first")
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
        strLog = ["FRED simulation logging:"]
        strLog.append("FRED Version:                   {:s}".format(simInfo["fredVersion"]))
        strLog.append("FRED Version Date:              {:s}".format(simInfo["fredVersionDate"]))
        runConfigKeys = [key for key in simInfo.keys() if re.search("runConfig.+", key)]
        if runConfigKeys:
            runConfigValues = [simInfo[runConfigKey] for runConfigKey in runConfigKeys]
            runConfigKeys = [runConfigKey.replace("runConfig", "") for runConfigKey in runConfigKeys]
            strLog.append(f"Run Config ({','.join(runConfigKeys)}):      {str(runConfigValues).replace('[','').replace(']','')}")
        strLog.append("Run Config:                     {}".format(simInfo["runConfig"]))
        strLog.append("Run Wall clock Time:            {:.2f} s".format(simInfo["runWallclockTime_s"]))
        strLog.append("Average Track Time Per Primary: {:5f} us".format(simInfo["trackTimePerPrimary_us"]))
        strLog.append("Average Tracking Rate:          {:.3E} prim/s".format(simInfo["trackingRate_prim_s"]))
        _logger.info("\n\t".join(strLog))

    return simInfo


def getFREDVersions() -> List[str]:
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


def checkFREDVersion(version: str) -> bool:
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
        error = ValueError(f"The version must be a string in format #.#.#, for instance '3.71.0'")
        _logger.error(error)
        raise error
    fredVersions = ft.getFREDVersions()

    return any([version in fredVersion for fredVersion in fredVersions])


def getFREDVersion(version: str = "") -> str:
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
        error = ValueError(f"No FRED v. {version} installed on the machine.\nAvailable FRED versions:\n" + "\n".join(ft.getFREDVersions()))
        _logger.error(error)
        raise error

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


def runFRED(fileName: PathLike, version: str = "", params: Iterable[str] = [], displayInfo: bool = False) -> List[str]:
    """Run FRED simulation.

    The function runs FRED simulation defined by
    the FRED input file name in the given FRED version.

    Parameters
    ----------
    fileName : path
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
    if not os.path.exists(fileName):
        raise ValueError(f"The file '{fileName}' dose not exist.")

    # get absolute folder name sim. file name
    fileName = os.path.abspath(fileName)
    simFolderName = os.path.dirname(fileName)
    fileName = os.path.basename(fileName)

    # run fred sim
    FREDrunCommand = ["fred"]

    FREDrunCommand.append(f"-useVers {version}") if version else None
    params = [params] if isinstance(params, str) else params
    FREDrunCommand.extend(params) if params else None
    FREDrunCommand.append(f"-f {fileName}")
    FREDrunCommand = r" ".join(FREDrunCommand)

    if displayInfo:
        _logger.info(f"Running FRED sim. in folder {simFolderName}")
        _logger.debug(f"{fredVersName}")

    runFredProc = subprocess.Popen(FREDrunCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="UTF-8", cwd=simFolderName + os.sep, universal_newlines=True)
    stdout, stderr = runFredProc.communicate()
    runFredProc.wait()

    if displayInfo:
        # check fred sim
        if "Run wallclock time" in stdout:
            FREDSimStat = ft.readFREDStat(os.path.join(simFolderName, "out/log/run.out"))
            _logger.info("FRED sim. done in {:.0f} s with {:.2E} prim/s".format(FREDSimStat["runWallclockTime_s"], FREDSimStat["trackingRate_prim_s"]))
        else:
            _logger.error(f"ERROR in FRED sim. Check {simFolderName}/out/log/run.out")

    return stdout.splitlines()
