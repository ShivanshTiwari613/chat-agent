#filepath: app/utils/logger.py

import logging
from os import name
import sys
from venv import logger
from config.settings import settings

def setup_logger(name: str = "AI_AGENT"):
    """
    Sets up a centralized logger with format and level defined in settings.
    """
    logger = logging.getLogger(name)

    # prevent adding multiple handlers if function is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Simple, clean format
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level from settings (e.g., INFO, DEBUG)
        logger.setLevel(settings.LOG_LEVEL.upper())
    
    return logger