import logging
import logging.config
from datetime import datetime


SHORT_FORMAT = '[%(asctime)s:%(levelname)-8s] %(message)s'

DATE_FORMAT = '%Y/%m/%d %H:%M:%S'

LOG_FILE_NAME = f"log_{datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')}.txt"

DEFAULT_LOGGER = {
    'version': 1,
    'root': {
        'level': 'NOTSET',
        'handlers': []
    },
    'loggers': {
        'main': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'brief',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'full',
            'level': 'INFO',
            'filename': LOG_FILE_NAME
        }
    },
    'formatters': {
        'full': {'format': SHORT_FORMAT, 'datefmt': DATE_FORMAT},
        'brief': {'format': SHORT_FORMAT, 'datefmt': DATE_FORMAT}
    }
}


def get_logger(name='main', level=logging.INFO):
    logging.config.dictConfig(DEFAULT_LOGGER)
    log = logging.getLogger(name)
    log.setLevel(level)
    return log