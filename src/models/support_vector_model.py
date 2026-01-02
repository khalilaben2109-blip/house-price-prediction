"""
Modèle Support Vector Regression (SVR)
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

class SupportVectorModel(BaseModel):
    """Modèle Support Vector Regression"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        super().__init__()
        self.model = SVR(
            kernel=kernel,
            C=C,
            gamma=gamma
        )
        self.scaler = StandardScaler()
        self.is_scaled = False
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Entraîne le modèle SVR
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
        """
        # SVR nécessite une normalisation supplémentaire
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        self.is_scaled = True
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions
        
        Args:
            X_test: Features de test
            
        Returns:
            np.ndarray: Prédictions
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        # Appliquer la même normalisation
        if self.is_scaled:
            X_scaled = self.scaler.transform(X_test)
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X_test)
    
    def get_feature_importance(self) -> dict:
        """
        SVR ne fournit pas d'importance des features directement
        
        Returns:
            dict: None pour SVR
        """
        return None