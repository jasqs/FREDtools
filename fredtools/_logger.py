# from typing import Literal
import logging


class customFormatterINFO(logging.Formatter):

    grey = "\x1b[38;21m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG:
        grey + '%(levelname)-8s: %(name)s.%(funcName)s: %(message)s' + reset,
        logging.INFO:
        grey + '%(funcName)s: %(message)s' + reset,
        logging.WARNING:
        yellow + '%(levelname)-8s: %(name)s.%(funcName)s: %(message)s' + reset,
        logging.ERROR:
        red + '%(levelname)-8s: %(name)s.%(funcName)s: %(message)s' + reset,
        logging.CRITICAL:
        bold_red + '%(levelname)-8s: %(name)s.%(funcName)s: %(message)s' + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class customFormatterDEBUG(logging.Formatter):

    grey = "\x1b[38;21m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG:
        grey + '%(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
        logging.INFO:
        grey + '%(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
        logging.WARNING:
        yellow + '%(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
        logging.ERROR:
        red + '%(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
        logging.CRITICAL:
        bold_red + '%(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def getConsoleLogHandler() -> logging.Handler:
    import sys
    consoleHandler = logging.StreamHandler(sys.stdout)
    return consoleHandler


def getFileLogHandler(fileName: str) -> logging.Handler:
    fileHandler = logging.FileHandler(fileName)
    # fileHandler.setFormatter(customFormatterDEBUG())
    return fileHandler


def getLogger(name: str | None = None) -> logging.Logger:
    """Get logger of a given name.

    The function is a wrapper for `logging.getLogger(name)` with a NullHandler added to handlers.

    Parameters
    ----------
    name : str or None, optional
        Name of the logger. If not providef or None, then the caller function name will be used. (def. None)
    """
    from fredtools._helper import currentFuncName
    if not name:
        name = currentFuncName(1)

    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())

    return logger


def configureLogging(level: int = logging.WARNING, console: bool = True, fileName: str | None = None, force: bool = True) -> None:
    """Configure logging if not configured.

    The function configures logging for the FREDtools library only if not configured already.

    Parameters
    ----------
    level : int, optional
        Logger level. It is recommended to use levels defined in logging, such as logging.DEBUG, logging.WARNING, etc. (def. logging.WARNING)
    console : bool, optional
        Determine if the standard console output handler should be added. (def. True)
    fileName : str or None, optional
        File name for logging or None if no logging to file is requested. (def. None)
    force : bool, optional
        Determine if any existing handlers attached to the root logger should be removed and closed before
        carrying out the configuration (def. True)
    """
    import sys

    loggingHandlers = []
    if console:
        loggingHandler = getConsoleLogHandler()

        if level <= logging.DEBUG:
            loggingHandler.setFormatter(customFormatterDEBUG())
        else:
            loggingHandler.setFormatter(customFormatterINFO())

        loggingHandlers.append(loggingHandler)

    if fileName:
        loggingHandler = getFileLogHandler(fileName)
        loggingHandler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)-0.3d] %(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s',
                                                      datefmt="%Y-%m-%d %H:%M:%S"))

        loggingHandlers.append(loggingHandler)

    logging.basicConfig(level=level,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=loggingHandlers,
                        force=force)
