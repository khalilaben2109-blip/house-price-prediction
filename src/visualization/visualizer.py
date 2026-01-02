"""
Module de visualisation des données et résultats
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any

class DataVisualizer:
    """Classe pour créer des visualisations"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_data_distribution(self, X: pd.DataFrame, y: pd.Series):
        """
        Visualise la distribution des données
        
        Args:
            X: Features
            y: Target
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Analyse Exploratoire des Données', fontsize=16)
        
        # Distribution du target
        axes[0, 0].hist(y, bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution des Prix des Maisons')
        axes[0, 0].set_xlabel('Prix (en milliers de $)')
        axes[0, 0].set_ylabel('Fréquence')
        
        # Boxplot du target
        axes[0, 1].boxplot(y)
        axes[0, 1].set_title('Boxplot des Prix')
        axes[0, 1].set_ylabel('Prix (en milliers de $)')
        
        # Matrice de corrélation
        corr_matrix = pd.concat([X, y], axis=1).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 0], fmt='.2f')
        axes[1, 0].set_title('Matrice de Corrélation')
        
        # Features les plus importantes (corrélation avec target)
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        axes[1, 1].barh(range(len(correlations)), correlations.values)
        axes[1, 1].set_yticks(range(len(correlations)))
        axes[1, 1].set_yticklabels(correlations.index)
        axes[1, 1].set_title('Corrélation des Features avec le Prix')
        axes[1, 1].set_xlabel('Corrélation Absolue')
        
        plt.tight_layout()
        plt.savefig('data/processed/data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]):
        """
        Compare visuellement les performances des modèles
        
        Args:
            results: Résultats des modèles
        """
        df_results = pd.DataFrame(results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparaison des Performances des Modèles', fontsize=16)
        
        metrics = ['RMSE', 'MAE', 'R2', 'MSE']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = df_results[metric].plot(kind='bar', ax=ax, color=colors[i])
            ax.set_title(f'{metric}')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs sur les barres
            for bar in bars.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('data/processed/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str):
        """
        Analyse détaillée des prédictions
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            model_name: Nom du modèle
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analyse des Prédictions - {model_name}', fontsize=16)
        
        # Prédictions vs Réalité
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Valeurs Réelles')
        axes[0, 0].set_ylabel('Prédictions')
        axes[0, 0].set_title('Prédictions vs Réalité')
        
        # Analyse des résidus
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Prédictions')
        axes[0, 1].set_ylabel('Résidus')
        axes[0, 1].set_title('Analyse des Résidus')
        
        # Distribution des résidus
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange')
        axes[1, 0].set_title('Distribution des Résidus')
        axes[1, 0].set_xlabel('Résidus')
        axes[1, 0].set_ylabel('Fréquence')
        
        # Q-Q plot des résidus
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot des Résidus')
        
        plt.tight_layout()
        plt.savefig(f'data/processed/{model_name.lower().replace(" ", "_")}_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()