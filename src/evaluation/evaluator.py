"""
Module d'évaluation des modèles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any

class ModelEvaluator:
    """Classe pour évaluer les performances des modèles"""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Évalue les performances d'un modèle
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            
        Returns:
            Dict[str, float]: Métriques d'évaluation
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred)
        }
        
        return metrics
    
    def compare_models(self, results: Dict[str, Dict[str, float]]):
        """
        Compare les performances de plusieurs modèles
        
        Args:
            results: Dictionnaire des résultats par modèle
        """
        df_results = pd.DataFrame(results).T
        
        print("=== COMPARAISON DES MODÈLES ===")
        print(df_results.round(4))
        
        # Graphique de comparaison
        self._plot_comparison(df_results)
        
        # Meilleur modèle
        best_model = df_results['RMSE'].idxmin()
        print(f"\nMeilleur modèle (RMSE le plus bas): {best_model}")
        print(f"RMSE: {df_results.loc[best_model, 'RMSE']:.4f}")
    
    def _plot_comparison(self, df_results: pd.DataFrame):
        """Crée un graphique de comparaison des modèles"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Comparaison des Performances des Modèles', fontsize=16)
        
        metrics = ['RMSE', 'MAE', 'R2', 'MSE']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            df_results[metric].plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
            ax.set_title(f'{metric}')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_name: str):
        """
        Graphique des prédictions vs valeurs réelles
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            model_name: Nom du modèle
        """
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title(f'{model_name} - Prédictions vs Réalité')
        
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Prédictions')
        plt.ylabel('Résidus')
        plt.title(f'{model_name} - Analyse des Résidus')
        
        plt.tight_layout()
        plt.show()