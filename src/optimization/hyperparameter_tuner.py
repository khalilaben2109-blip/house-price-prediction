"""
Module d'optimisation des hyperparamètres
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Tuple

class HyperparameterTuner:
    """Classe pour l'optimisation des hyperparamètres"""
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'neg_mean_squared_error'):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.best_params = {}
        self.best_scores = {}
    
    def tune_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          method: str = 'grid') -> Dict[str, Any]:
        """
        Optimise les hyperparamètres du Random Forest
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            method: 'grid' ou 'random'
            
        Returns:
            Dict: Meilleurs paramètres trouvés
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf = RandomForestRegressor(random_state=42)
        
        if method == 'grid':
            search = GridSearchCV(
                rf, param_grid, cv=self.cv_folds, 
                scoring=self.scoring, n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                rf, param_grid, cv=self.cv_folds, 
                scoring=self.scoring, n_jobs=-1, verbose=1,
                n_iter=50, random_state=42
            )
        
        search.fit(X_train, y_train)
        
        self.best_params['Random Forest'] = search.best_params_
        self.best_scores['Random Forest'] = -search.best_score_
        
        print(f"Meilleurs paramètres Random Forest: {search.best_params_}")
        print(f"Meilleur score (RMSE): {np.sqrt(-search.best_score_):.4f}")
        
        return search.best_params_
    
    def tune_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        La régression linéaire n'a pas d'hyperparamètres à optimiser
        
        Returns:
            Dict: Paramètres par défaut
        """
        default_params = {'fit_intercept': True}
        self.best_params['Linear Regression'] = default_params
        
        # Calculer le score avec validation croisée
        from sklearn.model_selection import cross_val_score
        lr = LinearRegression()
        scores = cross_val_score(lr, X_train, y_train, cv=self.cv_folds, 
                                scoring=self.scoring)
        self.best_scores['Linear Regression'] = -scores.mean()
        
        print(f"Score Linear Regression (RMSE): {np.sqrt(-scores.mean()):.4f}")
        
        return default_params
    
    def get_optimized_models(self) -> Dict[str, Any]:
        """
        Retourne les modèles avec les meilleurs hyperparamètres
        
        Returns:
            Dict: Modèles optimisés
        """
        models = {}
        
        if 'Random Forest' in self.best_params:
            models['Random Forest Optimized'] = RandomForestRegressor(
                **self.best_params['Random Forest'], random_state=42
            )
        
        if 'Linear Regression' in self.best_params:
            models['Linear Regression'] = LinearRegression(
                **self.best_params['Linear Regression']
            )
        
        return models
    
    def save_results(self, filepath: str = 'data/processed/hyperparameter_results.csv'):
        """
        Sauvegarde les résultats d'optimisation
        
        Args:
            filepath: Chemin de sauvegarde
        """
        results_data = []
        for model_name in self.best_params:
            row = {'Model': model_name, 'RMSE': np.sqrt(self.best_scores[model_name])}
            row.update(self.best_params[model_name])
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filepath, index=False)
        print(f"Résultats sauvegardés dans {filepath}")