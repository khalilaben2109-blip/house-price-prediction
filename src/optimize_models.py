"""
Script d'optimisation des hyperparamètres
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from optimization.hyperparameter_tuner import HyperparameterTuner
from evaluation.evaluator import ModelEvaluator
from utils.logger import setup_logger

def main():
    logger = setup_logger()
    logger.info("Début de l'optimisation des hyperparamètres")
    
    # Chargement et préparation des données
    data_loader = DataLoader()
    X, y = data_loader.load_boston_housing()
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(X, y)
    
    # Optimisation des hyperparamètres
    tuner = HyperparameterTuner(cv_folds=5)
    
    logger.info("Optimisation du Random Forest...")
    tuner.tune_random_forest(X_train, y_train, method='random')
    
    logger.info("Évaluation de la Linear Regression...")
    tuner.tune_linear_regression(X_train, y_train)
    
    # Évaluation des modèles optimisés
    logger.info("Évaluation des modèles optimisés")
    optimized_models = tuner.get_optimized_models()
    
    evaluator = ModelEvaluator()
    results = {}
    
    for name, model in optimized_models.items():
        logger.info(f"Test du modèle optimisé: {name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = evaluator.evaluate(y_test, predictions)
    
    # Affichage des résultats
    evaluator.compare_models(results)
    
    # Sauvegarde
    tuner.save_results()
    logger.info("Optimisation terminée!")

if __name__ == "__main__":
    main()