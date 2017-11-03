import os
import logging.config
import logging

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/'))

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s - %(name)s - %(levelname)s] - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": "info.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf8"
        },
    },
    # "loggers": {
    #     "my_module": {
    #         "level": "ERROR",
    #         "handlers": ["console"],
    #         "propagate": "no"
    #     }
    # },

    "root": {
        "level": "DEBUG",
        "handlers": ["console", "info_file_handler"]
    }
})

logging.getLogger(__name__).debug('Logging initialized')