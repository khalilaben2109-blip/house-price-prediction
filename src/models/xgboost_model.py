"""
Modèle XGBoost pour la régression
"""
import pandas as pd
import numpy as np
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """Modèle XGBoost pour la régression"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42):
        super().__init__()
        
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                objective='reg:squarederror'
            )
        except ImportError:
            print("⚠️  XGBoost non installé, utilisation de RandomForest à la place")
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Entraîne le modèle XGBoost
        
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
    
    def get_feature_importance(self) -> dict:
        """
        Retourne l'importance des features
        
        Returns:
            dict: Importance des features
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return {
                'feature_importance': self.model.feature_importances_
            }
        return None