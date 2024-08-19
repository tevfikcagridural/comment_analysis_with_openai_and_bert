from logging import Logger
import logging

def log_handler(logging_name: str) ->Logger:
    """
    Create a logger for API logs with file handler.

    Args:
        - logging_name (str): The name of the logger

    Returns:
        A configured logger instance.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.INFO)

    # Define the logger handlers
    file_handler = logging.FileHandler(f'data/interim/logs/{logging_name}.log', mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    return logger
