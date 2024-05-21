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
            grey + '[%(asctime)s.%(msecs)-0.3d] %(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
            logging.INFO:
            grey + '# %(message)s' + reset,
            logging.WARNING:
            yellow + '[%(asctime)s.%(msecs)-0.3d] %(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
            logging.ERROR:
            red + '[%(asctime)s.%(msecs)-0.3d] %(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset,
            logging.CRITICAL:
            bold_red + '[%(asctime)s.%(msecs)-0.3d] %(levelname)-8s: %(name)s.%(funcName)s:%(lineno)d: %(message)s' + reset
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    loggingStreamHandler = logging.StreamHandler(sys.stdout)
    loggingStreamHandler.setFormatter(customFormatter())

    logging.basicConfig(level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[loggingStreamHandler],
                        force=False)

    return logging.getLogger(name)
