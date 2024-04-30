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
        print(f"### {ft.currentFuncName()} ###")
        print("# Number of events: {:d}".format(simStat["NumberOfEvents"]))
        print("# Elapsed Time (total):   {:2f} s".format(simStat["ElapsedTime"]))
        print("# Elapsed Time (no init): {:2f} s".format(simStat["ElapsedTimeWoInit"]))
        print("# Average Tracking Rate:  {:.3E} prim/s".format(simStat["PPS"]))
        print("#" * len(f"### {ft.currentFuncName()} ###"))
    return simStat
