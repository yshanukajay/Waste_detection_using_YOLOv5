from wasteDetection.logger import logging
from wasteDetection.exception import AppException
import sys


try:
    result = 10 / 0
except Exception as e:
    logging.info("An exception occurred")
    raise AppException(e, sys) 