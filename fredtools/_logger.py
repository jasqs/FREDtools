import logging


def _getLogger(name: str) -> logging.Logger:
    import sys

    class customFormatter(logging.Formatter):
        grey = "\x1b[38;21m"
        yellow = "\x1b[33m"
        red = "\x1b[31m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        FORMATS = {
            logging.DEBUG:
            grey + '%(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
            logging.INFO:
            # grey + '# %(message)s' + reset,
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

    loggingStreamHandler = logging.StreamHandler(sys.stdout)
    loggingStreamHandler.setFormatter(customFormatter())

    logging.basicConfig(level=logging.WARNING,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[loggingStreamHandler],
                        force=False)

    logger = logging.getLogger(name)

    return logger


def _loggerDecorator(func):
    import functools
    import inspect

    @functools.wraps(func)
    def decorator(*args, **kwargs):
        # from fredtools._logger import *
        import fredtools as ft
        from fredtools import LOG_LEVEL
        global LOG_LEVEL

        logger = ft._getLogger(func.__module__)
        # logger.setLevel(LOG_LEVEL)
        loggerLevel = logger.getEffectiveLevel()

        bindArgs = inspect.signature(func).bind(*args, **kwargs).arguments
        # print(bindArgs)
        if "displayInfo" in bindArgs.keys() and bindArgs["displayInfo"]:

            # print("decorator: ", func.__module__)
            # logger.warning("Something is happening before the function is called.")
            # print("Something is happening before the function is called.")

            logger.setLevel(ft._logger.logging.INFO)

            print("effective loging level: ", ft._logger.logging.getLevelName(logger.getEffectiveLevel()))

        funcOut = func(*args, **kwargs)

        if "displayInfo" in bindArgs.keys() and bindArgs["displayInfo"]:
            # print("Something is happening after the function is called.")
            logger.setLevel(loggerLevel)
            # print(ft._logger.logging.getLevelName(logger.getEffectiveLevel()))
            del logger

        return funcOut
    return decorator


# # print("XXXXXXXXXXXXXXXXXXXXXXXXX")
# logger = _getLogger(__name__)
# logger.info(f"fredtools version: {ft.__version__}")
