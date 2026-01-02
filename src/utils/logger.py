"""
Configuration du logging
"""
import logging
import sys
from datetime import datetime

def setup_logger(name: str = 'house_price_prediction', level: int = logging.INFO):
    """
    Configure le logger pour le projet
    
    Args:
        name: Nom du logger
        level: Niveau de logging
        
    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Éviter la duplication des handlers
    if logger.handlers:
        return logger
    
    # Format des messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler pour fichier
    file_handler = logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger