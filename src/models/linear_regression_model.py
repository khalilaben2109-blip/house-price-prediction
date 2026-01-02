"""
Modèle de régression linéaire
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    """Modèle de régression linéaire"""
    
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Entraîne le modèle de régression linéaire
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
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
        
        return self.model.predict(X_test)
    
    def get_coefficients(self) -> dict:
        """
        Retourne les coefficients du modèle
        
        Returns:
            dict: Coefficients et intercept
        """
        if not self.is_trained:
            return None
        
        return {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_
        }