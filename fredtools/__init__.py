from .ImgIO import *
from .ImgAnalyse import *
from .ImgManipulate import *
from .MonteCarlo import *
from .BraggPeak import *
from .GammaIndex import *
from .Miscellaneous import *
from .helper import *


_version = [0, 8, 1]
__version__ = ".".join(map(str, _version))


def _checkJupyterMode():
    """Check if the FREDtools was loaded from jupyter"""
    try:
        if get_ipython().config["IPKernelApp"]:
            return True
    except:
        return False


def _checkMatplotlibBackend():
    import matplotlib

    if "inline" in matplotlib.get_backend():
        return "inline"
    elif "ipympl" in matplotlib.get_backend():
        return "ipympl"
    else:
        return "unknown"


def _currentFuncName(n=0):
    """Get name of the function where the currentFuncName() is called.
    currentFuncName(1) get the name of the caller.
    """
    import sys
    return sys._getframe(n + 1).f_code.co_name
