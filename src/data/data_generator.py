"""
Générateur de données diversifiées pour tester les modèles
"""
import pandas as pd
import numpy as np
import requests
from typing import Tuple, Optional
from sklearn.datasets import make_regression, fetch_california_housing
import warnings

class DataGenerator:
    """Générateur de datasets diversifiés"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_synthetic_housing_data(self, n_samples: int = 1000, 
                                      complexity: str = 'medium') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Génère des données synthétiques de maisons avec différents niveaux de complexité
        
        Args:
            n_samples: Nombre d'échantillons
            complexity: 'simple', 'medium', 'complex'
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        print(f"🏗️ Génération de {n_samples} propriétés synthétiques (niveau: {complexity})")
        
        if complexity == 'simple':
            n_features = 8
            noise = 0.05
        elif complexity == 'medium':
            n_features = 13
            noise = 0.1
        else:  # complex
            n_features = 20
            noise = 0.15
        
        # Génération de base
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=self.random_state
        )
        
        # Noms des features selon la complexité
        if complexity == 'simple':
            feature_names = ['surface', 'chambres', 'age', 'distance_centre', 
                           'score_quartier', 'garage', 'jardin', 'etage']
        elif complexity == 'medium':
            feature_names = ['surface', 'chambres', 'salles_bain', 'age', 'distance_centre',
                           'score_quartier', 'garage', 'jardin', 'etage', 'balcon',
                           'cave', 'ascenseur', 'chauffage']
        else:  # complex
            feature_names = ['surface', 'chambres', 'salles_bain', 'age', 'distance_centre',
                           'score_quartier', 'garage', 'jardin', 'etage', 'balcon',
                           'cave', 'ascenseur', 'chauffage', 'isolation', 'securite',
                           'transport_public', 'commerces', 'ecoles', 'hopitaux', 'pollution']
        
        # Créer le DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Normaliser et ajuster les valeurs pour qu'elles soient réalistes
        for i, col in enumerate(X_df.columns):
            if 'surface' in col:
                X_df[col] = np.abs(X_df[col]) * 50 + 50  # 50-500 m²
            elif 'chambres' in col:
                X_df[col] = np.abs(X_df[col]) % 6 + 1  # 1-6 chambres
            elif 'age' in col:
                X_df[col] = np.abs(X_df[col]) * 30 + 5  # 5-95 ans
            elif 'distance' in col:
                X_df[col] = np.abs(X_df[col]) * 20 + 1  # 1-40 km
            elif any(word in col for word in ['score', 'quartier', 'securite']):
                X_df[col] = (X_df[col] - X_df[col].min()) / (X_df[col].max() - X_df[col].min()) * 10  # 0-10
            else:
                # Variables binaires ou catégorielles
                X_df[col] = (X_df[col] > X_df[col].median()).astype(int)
        
        # Ajuster les prix de manière réaliste
        y_series = pd.Series(y, name='prix')
        
        # Formule plus réaliste basée sur les features principales
        realistic_price = (
            X_df['surface'] * 3000 +  # 3000€/m²
            X_df['chambres'] * 15000 +  # 15000€ par chambre
            (100 - X_df['age']) * 500 +  # Dépréciation avec l'âge
            X_df['score_quartier'] * 10000 +  # Impact du quartier
            np.random.normal(0, 20000, len(X_df))  # Variabilité
        )
        
        # Mélanger avec les données générées pour garder de la complexité
        y_final = 0.7 * realistic_price + 0.3 * ((y - y.min()) / (y.max() - y.min()) * 200000 + 100000)
        y_series = pd.Series(y_final, name='prix')
        
        print(f"✅ Données générées: prix moyen {y_series.mean():.0f}€, écart-type {y_series.std():.0f}€")
        return X_df, y_series
    
    def load_california_housing(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge le dataset California Housing de scikit-learn
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target
        """
        print("🏖️ Chargement du dataset California Housing...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                housing = fetch_california_housing()
            
            X = pd.DataFrame(housing.data, columns=housing.feature_names)
            y = pd.Series(housing.target * 100000, name='prix')  # Convertir en euros
            
            print(f"✅ California Housing chargé: {len(X)} propriétés")
            print(f"   Prix moyen: {y.mean():.0f}€, écart-type: {y.std():.0f}€")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement California Housing: {e}")
            print("🔄 Basculement vers données synthétiques")
            return self.generate_synthetic_housing_data(n_samples=20640, complexity='medium')
    
    def load_online_housing_data(self) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Tente de charger des données immobilières depuis une source en ligne
        
        Returns:
            Optional[Tuple[pd.DataFrame, pd.Series]]: Features et target ou None
        """
        print("🌐 Tentative de chargement de données en ligne...")
        
        # URLs de datasets publics
        urls = [
            "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
            "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        ]
        
        for i, url in enumerate(urls):
            try:
                print(f"   Essai {i+1}: {url.split('/')[-1]}")
                
                df = pd.read_csv(url)
                
                if len(df) > 100:  # Vérifier que le dataset est valide
                    # Identifier la colonne de prix (plusieurs noms possibles)
                    price_columns = ['medv', 'price', 'median_house_value', 'target', 'y']
                    price_col = None
                    
                    for col in price_columns:
                        if col in df.columns:
                            price_col = col
                            break
                    
                    if price_col:
                        X = df.drop(columns=[price_col])
                        y = df[price_col]
                        
                        # Nettoyer les données
                        X = X.select_dtypes(include=[np.number])  # Garder seulement les colonnes numériques
                        
                        if len(X.columns) >= 5:  # Au moins 5 features
                            print(f"✅ Données en ligne chargées: {len(df)} échantillons")
                            print(f"   Features: {list(X.columns)}")
                            print(f"   Prix moyen: {y.mean():.2f}, écart-type: {y.std():.2f}")
                            
                            return X, pd.Series(y.values, name='prix')
                
            except Exception as e:
                print(f"   ❌ Échec: {e}")
                continue
        
        print("❌ Impossible de charger des données en ligne")
        return None
    
    def generate_mixed_dataset(self, n_samples: int = 1500) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Génère un dataset mixte combinant plusieurs sources
        
        Args:
            n_samples: Nombre total d'échantillons souhaités
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features et target combinées
        """
        print(f"🎭 Génération d'un dataset mixte ({n_samples} échantillons)")
        
        datasets = []
        
        # 1. Essayer les données en ligne
        online_data = self.load_online_housing_data()
        if online_data:
            X_online, y_online = online_data
            if len(X_online) > 0:
                # Prendre un échantillon
                sample_size = min(len(X_online), n_samples // 3)
                idx = np.random.choice(len(X_online), sample_size, replace=False)
                datasets.append((X_online.iloc[idx], y_online.iloc[idx], "online"))
        
        # 2. California Housing
        try:
            X_cal, y_cal = self.load_california_housing()
            sample_size = min(len(X_cal), n_samples // 3)
            idx = np.random.choice(len(X_cal), sample_size, replace=False)
            datasets.append((X_cal.iloc[idx], y_cal.iloc[idx], "california"))
        except:
            pass
        
        # 3. Données synthétiques pour compléter
        remaining_samples = n_samples - sum(len(X) for X, y, source in datasets)
        if remaining_samples > 0:
            X_synth, y_synth = self.generate_synthetic_housing_data(
                n_samples=remaining_samples, 
                complexity='complex'
            )
            datasets.append((X_synth, y_synth, "synthetic"))
        
        if not datasets:
            # Fallback: tout synthétique
            return self.generate_synthetic_housing_data(n_samples=n_samples, complexity='medium')
        
        # Combiner tous les datasets
        print("🔄 Combinaison des sources de données...")
        
        # Trouver les colonnes communes
        all_columns = set()
        for X, y, source in datasets:
            all_columns.update(X.columns)
        
        # Prendre les colonnes les plus communes (au moins dans 2 sources)
        column_counts = {}
        for X, y, source in datasets:
            for col in X.columns:
                column_counts[col] = column_counts.get(col, 0) + 1
        
        common_columns = [col for col, count in column_counts.items() if count >= 1]
        common_columns = common_columns[:15]  # Limiter à 15 features max
        
        # Standardiser et combiner
        combined_X_list = []
        combined_y_list = []
        
        for X, y, source in datasets:
            # Sélectionner et réordonner les colonnes
            available_cols = [col for col in common_columns if col in X.columns]
            X_subset = X[available_cols].copy()
            
            # Ajouter les colonnes manquantes avec des valeurs par défaut
            for col in common_columns:
                if col not in X_subset.columns:
                    X_subset[col] = np.random.normal(0, 1, len(X_subset))
            
            # Réordonner les colonnes
            X_subset = X_subset[common_columns]
            
            combined_X_list.append(X_subset)
            combined_y_list.append(y)
            
            print(f"   📊 {source}: {len(X_subset)} échantillons")
        
        # Concaténer
        final_X = pd.concat(combined_X_list, ignore_index=True)
        final_y = pd.concat(combined_y_list, ignore_index=True)
        
        # Normaliser les prix pour qu'ils soient cohérents
        final_y = (final_y - final_y.min()) / (final_y.max() - final_y.min()) * 400000 + 100000
        final_y.name = 'prix'
        
        print(f"✅ Dataset mixte créé: {len(final_X)} échantillons, {len(final_X.columns)} features")
        print(f"   Prix moyen: {final_y.mean():.0f}€, écart-type: {final_y.std():.0f}€")
        
        return final_X, final_y
    
    def get_dataset_info(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Retourne des informations sur le dataset
        
        Args:
            X: Features
            y: Target
            
        Returns:
            dict: Informations sur le dataset
        """
        return {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': list(X.columns),
            'price_mean': y.mean(),
            'price_std': y.std(),
            'price_min': y.min(),
            'price_max': y.max(),
            'missing_values': X.isnull().sum().sum(),
            'data_types': X.dtypes.value_counts().to_dict()
        }