import logging


class customFormatterINFO(logging.Formatter):
    """
    Console log formatter with a compact layout for INFO messages.

    The formatter colours the log messages with ANSI escape sequences based on the log level,
    using the per-level formatting strings defined in the FORMATS dictionary. INFO records are
    formatted compactly as 'funcName: message', i.e. without the level name and logger name,
    whereas all the other levels are formatted as 'LEVEL: name.funcName: message'.
    No line numbers are included for any level. This formatter is selected by `configureLogging`
    for the console handler when the logging level is above logging.DEBUG.
    """
    grey = "\x1b[38;21m"
    blue = "\x1b[34m"
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
    """
    Console log formatter with a verbose layout including line numbers.

    The formatter colours the log messages with ANSI escape sequences based on the log level,
    using the per-level formatting strings defined in the FORMATS dictionary. All the levels,
    including INFO, are formatted as 'LEVEL: name.funcName:lineno: message', i.e. prefixed with
    the level name and with the line number appended to the function name. This formatter is
    selected by `configureLogging` for the console handler when the logging level is
    logging.DEBUG or lower.
    """
    grey = "\x1b[38;21m"
    blue = "\x1b[34m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG:
        blue + '%(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
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
    """
    Returns a console log handler that directs log messages to the standard output.

    Returns
    -------
    logging.Handler
        The console log handler.
    """
    import sys
    consoleHandler = logging.StreamHandler(sys.stdout)
    return consoleHandler


def getFileLogHandler(fileName: str) -> logging.Handler:
    """
    Returns a file log handler for the specified file name.

    Parameters
    ----------
    fileName : str
        The name of the file to log to.

    Returns
    -------
    logging.Handler
        The file log handler.
    """
    fileHandler = logging.FileHandler(fileName)
    # fileHandler.setFormatter(customFormatterDEBUG())
    return fileHandler


def getLogger(name: str | None = None) -> logging.Logger:
    """Get a logger with a NullHandler attached.

    The function returns a logger with a NullHandler attached. The NullHandler is attached to the logger
    to suppress any logging messages if no handler is attached to the logger.

    Parameters
    ----------
    name : str or None, optional
        Name of the logger. If not provided, the name of the calling function is used. (def. None)

    Returns
    -------
    logger : logging.Logger
        Logger object with a NullHandler attached.
    """
    # from fredtools._helper import currentFuncName
    if not name:
        name = currentFuncName(1)

    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())

    return logger


def configureLogging(level: int = logging.WARNING, console: bool = True, fileName: str | None = None, force: bool = True) -> None:
    """Configure logging for the FREDtools library.

    The function configures logging for the FREDtools library. By default (`force=True`),
    any existing handlers attached to the root logger are removed and closed before
    the configuration is carried out, i.e. the logging is always reconfigured.
    If `force` is False, the configuration is applied only if the root logger
    has no handlers configured already.

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


def currentFuncName(n=0):
    """Get name of the function where the currentFuncName() is called.

    currentFuncName(1) gets the name of the caller.
    """
    import sys
    return sys._getframe(n + 1).f_code.co_name
