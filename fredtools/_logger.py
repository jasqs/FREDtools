import logging
from typing import Literal


class customFormatterINFO(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG:
        grey + '%(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
        logging.INFO:
        grey + '%(funcName)s: %(message)s' + reset,
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


def getConsoleLogHandler():
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(customFormatterINFO())
    return consoleHandler


def getFileLogHandler(fileName: str):
    fileHandler = logging.FileHandler(fileName)
    fileHandler.setFormatter(customFormatterDEBUG())
    return fileHandler


def _getLogger(name: str) -> logging.Logger:
    import sys
    loggingStreamHandler = logging.StreamHandler(sys.stdout)
    loggingStreamHandler.setFormatter(customFormatterINFO())

    logging.basicConfig(level=logging.WARNING,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[loggingStreamHandler],
                        force=False)

    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    # logger.propagate = False

    return logger


def loggerDecorator(func):
    import functools
    import inspect

    @functools.wraps(func)
    def decorator(*args, **kwargs):
        import fredtools as ft
        # actions before the function call
        logger = ft._getLogger(func.__module__)
        loggerLevelCaller = logger.getEffectiveLevel()
        bindArgs = inspect.signature(func).bind(*args, **kwargs).arguments
        # print(bindArgs)
        # print(loggerLevelCaller)

        displayInfo = ("displayInfo" in bindArgs.keys() and bindArgs["displayInfo"] == True) or "displayImageInfo" == func.__name__
        # print(displayInfo)

        if displayInfo and loggerLevelCaller > ft._logger.logging.INFO:
            # print("changing level to info")
            logger.setLevel(ft._logger.logging.INFO)

        # function call
        funcOut = func(*args, **kwargs)

        # actions after the function call
        logger.setLevel(loggerLevelCaller)
        # print(ft._logger.logging.getLevelName(logger.getEffectiveLevel()))
        del logger

        return funcOut
    return decorator


def configureLogging(level: int = logging.WARNING) -> None:
    import sys

    loggingStreamHandler = logging.StreamHandler(sys.stdout)
    if level <= logging.DEBUG:
        loggingStreamHandler.setFormatter(customFormatterDEBUG())
    else:
        loggingStreamHandler.setFormatter(customFormatterINFO())

    logging.basicConfig(level=level,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[loggingStreamHandler],
                        force=True)
