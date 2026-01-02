"""
Script principal pour l'entraînement et l'évaluation des modèles
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.linear_regression_model import LinearRegressionModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.gradient_boosting_model import GradientBoostingModel
from models.support_vector_model import SupportVectorModel
from evaluation.evaluator import ModelEvaluator
from visualization.visualizer import DataVisualizer
from utils.logger import setup_logger

def main():
    logger = setup_logger()
    logger.info("Début de l'entraînement des modèles")
    
    # Chargement des données avec source mixte
    data_loader = DataLoader(use_database=False, data_source='mixed')
    X, y = data_loader.load_boston_housing()
    logger.info(f"Données chargées: {X.shape[0]} échantillons, {X.shape[1]} features")
    
    # Visualisation des données
    visualizer = DataVisualizer()
    logger.info("Création des visualisations exploratoires")
    visualizer.plot_data_distribution(X, y)
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(X, y)
    logger.info(f"Données préparées: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Modèles (5 algorithmes)
    models = {
        'Linear Regression': LinearRegressionModel(),
        'Random Forest': RandomForestModel(),
        'XGBoost': XGBoostModel(),
        'Gradient Boosting': GradientBoostingModel(),
        'Support Vector': SupportVectorModel()
    }
    
    # Entraînement et évaluation
    evaluator = ModelEvaluator()
    results = {}
    predictions_dict = {}
    
    for name, model in models.items():
        logger.info(f"Entraînement du modèle: {name}")
        try:
            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            results[name] = evaluator.evaluate(y_test, predictions)
            predictions_dict[name] = predictions
            
            # Analyse détaillée des prédictions
            visualizer.plot_predictions_analysis(y_test.values, predictions, name)
        except Exception as e:
            logger.error(f"Erreur avec le modèle {name}: {e}")
            continue
    
    # Comparaison des modèles
    logger.info("Comparaison des modèles")
    evaluator.compare_models(results)
    visualizer.plot_model_comparison(results)
    
    # Sauvegarde des résultats
    import pandas as pd
    results_df = pd.DataFrame(results).T
    results_df.to_csv('data/processed/model_results.csv')
    logger.info("Résultats sauvegardés dans data/processed/model_results.csv")
    
    logger.info("Entraînement terminé avec succès!")

if __name__ == "__main__":
    main()