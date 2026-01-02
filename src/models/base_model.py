"""
Classe de base pour tous les modèles
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Classe abstraite pour tous les modèles"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Entraîne le modèle"""
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Fait des prédictions"""
        pass
    
    def get_feature_importance(self) -> dict:
        """Retourne l'importance des features si disponible"""
        return None