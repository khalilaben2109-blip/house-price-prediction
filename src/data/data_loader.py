"""
Module pour charger les datasets
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from typing import Tuple, Optional
import os
import sys

# Ajouter le chemin pour importer le database_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataLoader:
    """Classe pour charger différents datasets"""
    
    def __init__(self, use_database: bool = False, data_source: str = 'mixed'):
        """
        Initialise le DataLoader
        
        Args:
            use_database: Si True, utilise la base de données, sinon génère des données
            data_source: 'synthetic', 'california', 'online', 'mixed'
        """
        self.use_database = use_database
        self.data_source = data_source
        
        if use_database:
            try:
                from database.database_manager import DatabaseManager
                self.db_manager = DatabaseManager()
            except ImportError as e:
                print(f"⚠️  Impossible d'importer le gestionnaire de base de données: {e}")
                print("📊 Utilisation des données alternatives à la place")
                self.use_database = False
    
    def load_boston_housing(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge le dataset des prix des maisons selon la source configurée
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        if self.use_database:
            return self._load_from_database()
        else:
            return self._load_from_source()
    
    def _load_from_database(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge les données depuis la base de données
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        try:
            print("📊 Chargement des données depuis la base de données...")
            X, y = self.db_manager.load_properties_data()
            
            # Afficher les statistiques de la base
            stats = self.db_manager.get_database_stats()
            print(f"✅ Base de données connectée:")
            print(f"   📋 Propriétés: {stats['properties']}")
            print(f"   🔮 Prédictions: {stats['predictions']}")
            print(f"   🤖 Modèles entraînés: {stats['trained_models']}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement depuis la base de données: {e}")
            print("📊 Basculement vers les données alternatives")
            return self._load_from_source()
    
    def _load_from_source(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge les données selon la source configurée
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        from data.data_generator import DataGenerator
        
        generator = DataGenerator()
        
        if self.data_source == 'synthetic':
            return generator.generate_synthetic_housing_data(n_samples=1000, complexity='medium')
        elif self.data_source == 'california':
            return generator.load_california_housing()
        elif self.data_source == 'online':
            online_data = generator.load_online_housing_data()
            if online_data:
                return online_data
            else:
                print("🔄 Basculement vers données synthétiques")
                return generator.generate_synthetic_housing_data(n_samples=1000, complexity='medium')
        elif self.data_source == 'mixed':
            return generator.generate_mixed_dataset(n_samples=1500)
        else:
            # Fallback vers les données synthétiques originales
            return self._generate_original_synthetic_data()
    
    def _generate_original_synthetic_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Génère le dataset synthétique original (pour compatibilité)
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        print("📊 Génération de données synthétiques (format original)...")
        
        # Génération de données synthétiques
        np.random.seed(42)
        X, y = make_regression(
            n_samples=506,  # Même taille que le dataset Boston original
            n_features=13,
            noise=0.1,
            random_state=42
        )
        
        # Noms des features similaires au dataset Boston
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                        'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='MEDV')
        
        # Normaliser les prix pour qu'ils ressemblent à des prix de maisons
        y_series = (y_series - y_series.min()) / (y_series.max() - y_series.min()) * 40 + 10
        
        print("✅ Données synthétiques générées (format original)")
        return X_df, y_series
    
    def get_available_sources(self) -> list:
        """
        Retourne la liste des sources de données disponibles
        
        Returns:
            list: Sources disponibles
        """
        return ['synthetic', 'california', 'online', 'mixed', 'original']
    
    def set_data_source(self, source: str):
        """
        Change la source de données
        
        Args:
            source: Nouvelle source ('synthetic', 'california', 'online', 'mixed')
        """
        if source in self.get_available_sources():
            self.data_source = source
            print(f"📊 Source de données changée vers: {source}")
        else:
            print(f"❌ Source inconnue: {source}")
            print(f"Sources disponibles: {self.get_available_sources()}")
    
    def save_prediction_to_db(self, model_name: str, predicted_price: float, 
                             actual_price: Optional[float] = None,
                             model_version: str = "1.0", confidence_score: Optional[float] = None):
        """
        Sauvegarde une prédiction dans la base de données
        
        Args:
            model_name: Nom du modèle utilisé
            predicted_price: Prix prédit
            actual_price: Prix réel (si connu)
            model_version: Version du modèle
            confidence_score: Score de confiance
        """
        if self.use_database and hasattr(self, 'db_manager'):
            try:
                prediction_id = self.db_manager.save_prediction(
                    property_id=None,
                    model_name=model_name,
                    predicted_price=predicted_price,
                    actual_price=actual_price,
                    model_version=model_version,
                    confidence_score=confidence_score
                )
                print(f"💾 Prédiction sauvegardée avec ID: {prediction_id}")
                return prediction_id
            except Exception as e:
                print(f"❌ Erreur lors de la sauvegarde: {e}")
        else:
            print("⚠️  Base de données non disponible, prédiction non sauvegardée")
    
    def save_model_results_to_db(self, model_name: str, model_version: str,
                                metrics: dict, hyperparameters: dict,
                                training_samples: int, test_samples: int):
        """
        Sauvegarde les résultats d'un modèle dans la base de données
        """
        if self.use_database and hasattr(self, 'db_manager'):
            try:
                model_id = self.db_manager.save_model_results(
                    model_name=model_name,
                    model_version=model_version,
                    metrics=metrics,
                    hyperparameters=hyperparameters,
                    training_samples=training_samples,
                    test_samples=test_samples
                )
                print(f"💾 Résultats du modèle sauvegardés avec ID: {model_id}")
                return model_id
            except Exception as e:
                print(f"❌ Erreur lors de la sauvegarde: {e}")
        else:
            print("⚠️  Base de données non disponible, résultats non sauvegardés")
    
    def get_database_stats(self) -> dict:
        """Récupère les statistiques de la base de données"""
        if self.use_database and hasattr(self, 'db_manager'):
            return self.db_manager.get_database_stats()
        return {}
    
    def load_kaggle_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge un dataset depuis un fichier CSV Kaggle
        
        Args:
            filepath: Chemin vers le fichier CSV
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        df = pd.read_csv(filepath)
        # Adapter selon la structure du dataset Kaggle
        X = df.drop('price', axis=1)  # Supposer que 'price' est la colonne target
        y = df['price']
        
        return X, y
    
    def load_kaggle_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge un dataset depuis un fichier CSV Kaggle
        
        Args:
            filepath: Chemin vers le fichier CSV
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        df = pd.read_csv(filepath)
        # Adapter selon la structure du dataset Kaggle
        X = df.drop('price', axis=1)  # Supposer que 'price' est la colonne target
        y = df['price']
        
        return X, y