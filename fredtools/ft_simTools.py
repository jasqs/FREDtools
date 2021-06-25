import os
import warnings
import fredtools as ft


def setFieldsFolderStruct(folderPath, RNfileName, folderName="FRED", displayInfo=False):
    """Create folder structure for each field in treatment plan

    The function creates a folder structure in a given `folderPath` for each field separately.
    The folder structure is in form:

        folderPath/folderName:
                    |- F1
                    |- F2
                    ...

    Parameters
    ----------
    folderPath : path
        Path to folder to create the structure.
    RNfileName : path
        Path to RN dicom file of a treatment plan.
    folderName : string
        Name of the folder to create (def. 'FRED')
    displayInfo : bool
        Displays a summary of the function results (def. False).

    Returns
    -------
    path
        Path to created folder structure.

    """
    SimFolder = os.path.join(folderPath, folderName)
    if not os.path.exists(SimFolder):
        os.mkdir(SimFolder)
    else:
        warnings.warn("Warning: {:s} simulation folder already exists.".format(folderName))

    # create subfolders for fields
    planInfo = ft.getRNInfo(RNfileName, displayInfo=False)
    for fieldNo in planInfo["fieldsNumber"]:
        if not os.path.exists(os.path.join(SimFolder, "F{:d}".format(fieldNo))):
            os.mkdir(os.path.join(SimFolder, "F{:d}".format(fieldNo)))
    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# Created {:d} field folders in {:s}".format(planInfo["numberOfBeams"], SimFolder))
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return SimFolder


def readFREDStat(fileNameLogOut, displayInfo=False):
    """Read FRED simulation statistics information from logfile.

    The function reads some statistics information from a FRED run.out logfile.
    If some information os not available, then a NaN or numpy.nan is returned.

    Parameters
    ----------
    fileNameLogOut : string
        A string path to FRED outou logfile (usually in out/log/run.out)
    displayInfo : bool, optional
        Displays a summary of the function results (def. False).

    Returns
    -------
    dict
        A dictionary with the read data.
    """
    import os
    import re
    from numpy import nan

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
            return nan

    # check if file exists
    if not os.path.isfile(fileNameLogOut):
        raise ValueError(f"The file {fileNameLogOut} dose not exist.")

    simInfo = {
        "fredVersion": "NaN",
        "fredVersionDate": "NaN",
        "runConfig": "NaN",
        "runConfigMPI": nan,
        "runConfigTHREADS": nan,
        "runConfigGPU": nan,
        "runWallclockTime_s": nan,
        "primarySimulated": nan,
        "trackingRate_prim_s": nan,
        "trackTimePerPrimary_us": nan,
        "timingInitialization_s": nan,
        "timingPBSkimming_s": nan,
        "timingPrimaryList_s": nan,
        "timingGeometryChecking_s": nan,
        "timingTracking_s": nan,
        "timingWritingOutput_s": nan,
        "timingOther_s": nan,
    }

    with open(fileNameLogOut) as f:
        for num, line in enumerate(f, 1):

            # FRED Version and relase date
            Version_re = re.search("Version\W+([\S+.]+)", line)
            VersionDate_re = re.search("Version.*([0-9]{4}\/[0-9]{2}\/[0-9]{2})", line)
            if Version_re:
                simInfo["fredVersion"] = Version_re.group(1)
            if VersionDate_re:
                simInfo["fredVersionDate"] = VersionDate_re.group(1)

            # configuration fo the run
            RunningConfig_re = re.search("Running config.*([0-9]+)\,([0-9]+)\,([0-9]+)", line)
            if RunningConfig_re:
                simInfo["runConfigMPI"] = int(RunningConfig_re.group(1))
                simInfo["runConfigTHREADS"] = int(RunningConfig_re.group(2))
                simInfo["runConfigGPU"] = int(RunningConfig_re.group(3))
                if simInfo["runConfigGPU"] == 0:
                    simInfo["runConfig"] = "CPUx{:d}".format(simInfo["runConfigTHREADS"])
                else:
                    simInfo["runConfig"] = "GPUx{:d}".format(simInfo["runConfigGPU"])

            # total run time
            RunWallclockTime_re = re.findall("Run wallclock time:\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)", line)
            if RunWallclockTime_re:
                simInfo["runWallclockTime_s"] = float(RunWallclockTime_re[0])

            # total number of primaries simulated
            PrimarySimulated_re = re.findall("Number of primary particles\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)", line)
            if PrimarySimulated_re:
                simInfo["primarySimulated"] = int(float(PrimarySimulated_re[0]))

            # Average Tracking Rate (prim/s)
            TrackingRate_re = re.findall("Tracking rate\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)", line)
            if TrackingRate_re:
                simInfo["trackingRate_prim_s"] = float(TrackingRate_re[0])

            # Average Track time per prim
            TrackTimePerPrimary_re = re.findall("Track time per primary\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)\W+(.+)", line)
            if TrackTimePerPrimary_re:
                simInfo["trackTimePerPrimary_us"] = float(TrackTimePerPrimary_re[0][0]) / scaleUnit(TrackTimePerPrimary_re[0][1]) * 1e6

            # Timing: initialization
            TimingInitialization_re = re.findall("\W+initialization\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)\W+([A-Za-z]+)", line)
            if TimingInitialization_re:
                simInfo["timingInitialization_s"] = float(TimingInitialization_re[0][0]) / scaleUnit(TimingInitialization_re[0][1])
            # Timing: PB skimming
            TimingPBSkimming_re = re.findall("\W+PB skimming\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)\W+([A-Za-z]+)", line)
            if TimingPBSkimming_re:
                simInfo["timingPBSkimming_s"] = float(TimingPBSkimming_re[0][0]) / scaleUnit(TimingPBSkimming_re[0][1])
            # Timing: primary list
            TimingPrimaryList_re = re.findall("\W+primary list\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)\W+([A-Za-z]+)", line)
            if TimingPrimaryList_re:
                simInfo["timingPrimaryList_s"] = float(TimingPrimaryList_re[0][0]) / scaleUnit(TimingPrimaryList_re[0][1])
            # Timing: geometry checking
            TimingGeometryChecking_re = re.findall("\W+geometry checking\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)\W+([A-Za-z]+)", line)
            if TimingGeometryChecking_re:
                simInfo["timingGeometryChecking_s"] = float(TimingGeometryChecking_re[0][0]) / scaleUnit(TimingGeometryChecking_re[0][1])
            # Timing: tracking
            TimingTracking_re = re.findall("\W+tracking\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)\W+([A-Za-z]+)", line)
            if TimingTracking_re:
                simInfo["timingTracking_s"] = float(TimingTracking_re[0][0]) / scaleUnit(TimingTracking_re[0][1])
            # Timing: writing output
            TimingWritingOutput_re = re.findall("\W+writing output\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)\W+([A-Za-z]+)", line)
            if TimingWritingOutput_re:
                simInfo["timingWritingOutput_s"] = float(TimingWritingOutput_re[0][0]) / scaleUnit(TimingWritingOutput_re[0][1])
            # Timing: other
            TimingOther_re = re.findall("\W+other\W+([-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?)\W+([A-Za-z]+)", line)
            if TimingOther_re:
                simInfo["timingOther_s"] = float(TimingOther_re[0][0]) / scaleUnit(TimingOther_re[0][1])

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print("# FRED Version: {:s}".format(simInfo["fredVersion"]))
        print("# FRED Version Date: {:s}".format(simInfo["fredVersionDate"]))
        print("# Run Config (MPI,THREADS,GPU): {:d},{:d},{:d}".format(simInfo["runConfigMPI"], simInfo["runConfigTHREADS"], simInfo["runConfigGPU"]))
        print("# Run Config: {:s}".format(simInfo["runConfig"]))
        print("# Run Wall clock Time: {:.2f} s".format(simInfo["runWallclockTime_s"]))
        print("# Average Track Time Per Primary: {:5f} us".format(simInfo["trackTimePerPrimary_us"]))
        print("# Average Tracking Rate: {:.3E} prim/s".format(simInfo["trackingRate_prim_s"]))
        print("#" * len(f"### {ft._currentFuncName()} ###"))
    return simInfo
