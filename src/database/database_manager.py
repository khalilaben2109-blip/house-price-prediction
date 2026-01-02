"""
Gestionnaire de base de données pour le projet
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import os
from datetime import datetime
import logging

class DatabaseManager:
    """Gestionnaire de connexions et opérations de base de données"""
    
    def __init__(self, db_path: str = "data/house_prices.db"):
        self.db_path = db_path
        self.connection = None
        self.logger = logging.getLogger(__name__)
        
        # Créer le dossier data s'il n'existe pas
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialiser la base de données
        self._initialize_database()
    
    def connect(self) -> sqlite3.Connection:
        """Établit une connexion à la base de données"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
            self.logger.info(f"Connexion établie à la base de données: {self.db_path}")
            return self.connection
        except Exception as e:
            self.logger.error(f"Erreur de connexion à la base de données: {e}")
            raise
    
    def disconnect(self):
        """Ferme la connexion à la base de données"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Connexion fermée")
    
    def _initialize_database(self):
        """Initialise la structure de la base de données"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Table des propriétés immobilières
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS properties (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crim REAL,           -- Taux de criminalité
                    zn REAL,             -- Proportion de terrains résidentiels
                    indus REAL,          -- Proportion d'acres commerciales
                    chas INTEGER,        -- Variable Charles River (0 ou 1)
                    nox REAL,            -- Concentration d'oxydes nitriques
                    rm REAL,             -- Nombre moyen de pièces
                    age REAL,            -- Proportion de logements anciens
                    dis REAL,            -- Distance aux centres d'emploi
                    rad INTEGER,         -- Accessibilité aux autoroutes
                    tax REAL,            -- Taux de taxe foncière
                    ptratio REAL,        -- Ratio élèves/enseignants
                    b REAL,              -- Proportion de population noire
                    lstat REAL,          -- Pourcentage de population défavorisée
                    medv REAL,           -- Prix médian (target)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table des prédictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    property_id INTEGER,
                    model_name TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_version TEXT,
                    confidence_score REAL,
                    FOREIGN KEY (property_id) REFERENCES properties (id)
                )
            """)
            
            # Table des modèles entraînés
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trained_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rmse REAL,
                    mae REAL,
                    r2_score REAL,
                    mse REAL,
                    hyperparameters TEXT,  -- JSON des hyperparamètres
                    training_samples INTEGER,
                    test_samples INTEGER
                )
            """)
            
            # Table des logs d'entraînement
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    log_level TEXT,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES trained_models (id)
                )
            """)
            
            conn.commit()
            self.logger.info("Base de données initialisée avec succès")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
            raise
        finally:
            self.disconnect()
    
    def insert_sample_data(self, num_samples: int = 506):
        """Insère des données d'exemple dans la base de données"""
        from sklearn.datasets import make_regression
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Vérifier si des données existent déjà
            cursor.execute("SELECT COUNT(*) FROM properties")
            count = cursor.fetchone()[0]
            
            if count > 0:
                self.logger.info(f"La base contient déjà {count} propriétés")
                return
            
            # Générer des données synthétiques
            np.random.seed(42)
            X, y = make_regression(
                n_samples=num_samples,
                n_features=13,
                noise=0.1,
                random_state=42
            )
            
            # Normaliser les prix pour qu'ils ressemblent à des prix de maisons
            y = (y - y.min()) / (y.max() - y.min()) * 40 + 10
            
            # Noms des colonnes
            feature_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
                           'rad', 'tax', 'ptratio', 'b', 'lstat']
            
            # Insérer les données
            for i in range(num_samples):
                values = [float(X[i, j]) for j in range(13)] + [float(y[i])]
                placeholders = ', '.join(['?'] * 14)
                
                cursor.execute(f"""
                    INSERT INTO properties ({', '.join(feature_names)}, medv)
                    VALUES ({placeholders})
                """, values)
            
            conn.commit()
            self.logger.info(f"Insertion de {num_samples} propriétés réussie")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Erreur lors de l'insertion des données: {e}")
            raise
        finally:
            self.disconnect()
    
    def load_properties_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge les données des propriétés depuis la base de données
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        conn = self.connect()
        
        try:
            query = """
                SELECT crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv
                FROM properties
                ORDER BY id
            """
            
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                self.logger.warning("Aucune donnée trouvée, insertion de données d'exemple")
                self.disconnect()
                self.insert_sample_data()
                conn = self.connect()
                df = pd.read_sql_query(query, conn)
            
            # Séparer features et target
            feature_columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
                             'rad', 'tax', 'ptratio', 'b', 'lstat']
            
            X = df[feature_columns]
            y = df['medv']
            
            self.logger.info(f"Données chargées: {len(df)} propriétés")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {e}")
            raise
        finally:
            self.disconnect()
    
    def save_prediction(self, property_id: Optional[int], model_name: str, 
                       predicted_price: float, actual_price: Optional[float] = None,
                       model_version: str = "1.0", confidence_score: Optional[float] = None):
        """
        Sauvegarde une prédiction dans la base de données
        
        Args:
            property_id: ID de la propriété (None pour nouvelle prédiction)
            model_name: Nom du modèle utilisé
            predicted_price: Prix prédit
            actual_price: Prix réel (si connu)
            model_version: Version du modèle
            confidence_score: Score de confiance
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO predictions (property_id, model_name, predicted_price, 
                                       actual_price, model_version, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (property_id, model_name, predicted_price, actual_price, 
                  model_version, confidence_score))
            
            conn.commit()
            prediction_id = cursor.lastrowid
            self.logger.info(f"Prédiction sauvegardée avec ID: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Erreur lors de la sauvegarde de la prédiction: {e}")
            raise
        finally:
            self.disconnect()
    
    def save_model_results(self, model_name: str, model_version: str, 
                          metrics: Dict[str, float], hyperparameters: Dict[str, Any],
                          training_samples: int, test_samples: int):
        """
        Sauvegarde les résultats d'un modèle entraîné
        
        Args:
            model_name: Nom du modèle
            model_version: Version du modèle
            metrics: Dictionnaire des métriques (RMSE, MAE, R2, MSE)
            hyperparameters: Hyperparamètres du modèle
            training_samples: Nombre d'échantillons d'entraînement
            test_samples: Nombre d'échantillons de test
        """
        import json
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO trained_models (model_name, model_version, rmse, mae, r2_score, mse,
                                          hyperparameters, training_samples, test_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_name, model_version, metrics.get('RMSE'), metrics.get('MAE'),
                  metrics.get('R2'), metrics.get('MSE'), json.dumps(hyperparameters),
                  training_samples, test_samples))
            
            conn.commit()
            model_id = cursor.lastrowid
            self.logger.info(f"Résultats du modèle sauvegardés avec ID: {model_id}")
            return model_id
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
            raise
        finally:
            self.disconnect()
    
    def get_model_history(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Récupère l'historique des modèles entraînés
        
        Args:
            model_name: Nom du modèle (optionnel, pour filtrer)
            
        Returns:
            pd.DataFrame: Historique des modèles
        """
        conn = self.connect()
        
        try:
            query = """
                SELECT model_name, model_version, training_date, rmse, mae, r2_score, mse,
                       training_samples, test_samples
                FROM trained_models
            """
            
            params = []
            if model_name:
                query += " WHERE model_name = ?"
                params.append(model_name)
            
            query += " ORDER BY training_date DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de l'historique: {e}")
            raise
        finally:
            self.disconnect()
    
    def get_predictions_history(self, limit: int = 100) -> pd.DataFrame:
        """
        Récupère l'historique des prédictions
        
        Args:
            limit: Nombre maximum de prédictions à récupérer
            
        Returns:
            pd.DataFrame: Historique des prédictions
        """
        conn = self.connect()
        
        try:
            query = """
                SELECT p.id, p.model_name, p.predicted_price, p.actual_price,
                       p.prediction_date, p.model_version, p.confidence_score
                FROM predictions p
                ORDER BY p.prediction_date DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=[limit])
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des prédictions: {e}")
            raise
        finally:
            self.disconnect()
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Récupère les statistiques de la base de données
        
        Returns:
            Dict[str, int]: Statistiques (nombre de propriétés, prédictions, modèles)
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Nombre de propriétés
            cursor.execute("SELECT COUNT(*) FROM properties")
            stats['properties'] = cursor.fetchone()[0]
            
            # Nombre de prédictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            stats['predictions'] = cursor.fetchone()[0]
            
            # Nombre de modèles entraînés
            cursor.execute("SELECT COUNT(*) FROM trained_models")
            stats['trained_models'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des statistiques: {e}")
            raise
        finally:
            self.disconnect()

# Fonction utilitaire pour obtenir une instance du gestionnaire
def get_database_manager() -> DatabaseManager:
    """Retourne une instance du gestionnaire de base de données"""
    return DatabaseManager()