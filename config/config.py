"""
Configuration du projet
"""
import os

# Chemins
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Paramètres des modèles
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    'linear_regression': {
        'fit_intercept': True,
        'normalize': False
    }
}

# Paramètres de preprocessing
PREPROCESSING_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'scaling_method': 'standard'
}

# Créer les dossiers s'ils n'existent pas
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)