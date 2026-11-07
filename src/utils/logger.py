"""Logging utility for the Classroom Face Recognition System."""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "ClassroomRecognition", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard logger instance.
    
    Args:
        name: Logger name.
        log_file: Optional file path to log to.
        level: Logging verbosity level.
        
    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Optional file handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger

