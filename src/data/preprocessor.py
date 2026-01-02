"""
Module pour le preprocessing des données
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple

class DataPreprocessor:
    """Classe pour le preprocessing des données"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prépare les données pour l'entraînement
        
        Args:
            X: Features
            y: Target
            test_size: Taille du set de test
            random_state: Seed pour la reproductibilité
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Gestion des valeurs manquantes
        X_clean = self._handle_missing_values(X)
        
        # Encodage des variables catégorielles
        X_encoded = self._encode_categorical_features(X_clean)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state
        )
        
        # Normalisation
        X_train_scaled = self._scale_features(X_train, fit=True)
        X_test_scaled = self._scale_features(X_test, fit=False)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Gère les valeurs manquantes"""
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]
        
        if X_numeric.isnull().sum().sum() > 0:
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X_numeric),
                columns=numeric_columns,
                index=X.index
            )
            X[numeric_columns] = X_imputed
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode les variables catégorielles"""
        X_encoded = X.copy()
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X_encoded[col] = self.label_encoders[col].fit_transform(X[col])
        
        return X_encoded
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Normalise les features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)