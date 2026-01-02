"""
Démonstration du projet avec base de données
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.linear_regression_model import LinearRegressionModel
from models.random_forest_model import RandomForestModel
from evaluation.evaluator import ModelEvaluator
from database.database_manager import DatabaseManager
from utils.logger import setup_logger
import pandas as pd

def main():
    print("=" * 70)
    print("🏠 DÉMONSTRATION AVEC BASE DE DONNÉES 🏠")
    print("=" * 70)
    
    logger = setup_logger()
    
    # 1. Initialisation de la base de données
    print("\n💾 1. INITIALISATION DE LA BASE DE DONNÉES")
    print("-" * 50)
    
    try:
        db_manager = DatabaseManager()
        print("✅ Base de données SQLite initialisée")
        print(f"📍 Emplacement: {db_manager.db_path}")
        
        # Insérer des données d'exemple si nécessaire
        db_manager.insert_sample_data()
        
        # Afficher les statistiques
        stats = db_manager.get_database_stats()
        print(f"📊 Statistiques de la base:")
        print(f"   📋 Propriétés: {stats['properties']}")
        print(f"   🔮 Prédictions: {stats['predictions']}")
        print(f"   🤖 Modèles entraînés: {stats['trained_models']}")
        
    except Exception as e:
        print(f"❌ Erreur d'initialisation de la base de données: {e}")
        return
    
    # 2. Chargement des données depuis la base
    print("\n📊 2. CHARGEMENT DES DONNÉES DEPUIS LA BASE")
    print("-" * 50)
    
    # Utiliser le DataLoader avec base de données
    data_loader = DataLoader(use_database=True)
    X, y = data_loader.load_boston_housing()
    
    print(f"✅ Données chargées: {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"   Prix moyen: {y.mean():.2f}k$, Écart-type: {y.std():.2f}k$")
    
    # 3. Preprocessing
    print("\n🔧 3. PREPROCESSING DES DONNÉES")
    print("-" * 50)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(X, y)
    print(f"✅ Division train/test: {X_train.shape[0]} / {X_test.shape[0]} échantillons")
    
    # 4. Entraînement des modèles avec sauvegarde en base
    print("\n🤖 4. ENTRAÎNEMENT ET SAUVEGARDE DES MODÈLES")
    print("-" * 50)
    
    models = {
        'Linear Regression': LinearRegressionModel(),
        'Random Forest': RandomForestModel()
    }
    
    evaluator = ModelEvaluator()
    results = {}
    
    for name, model in models.items():
        print(f"   Entraînement: {name}...")
        
        # Entraîner le modèle
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluator.evaluate(y_test, predictions)
        results[name] = metrics
        
        # Sauvegarder les résultats en base
        hyperparams = {}
        if hasattr(model.model, 'get_params'):
            hyperparams = model.model.get_params()
        
        model_id = data_loader.save_model_results_to_db(
            model_name=name,
            model_version="1.0",
            metrics=metrics,
            hyperparameters=hyperparams,
            training_samples=len(X_train),
            test_samples=len(X_test)
        )
        
        print(f"   ✅ {name} - RMSE: {metrics['RMSE']:.4f} (ID: {model_id})")
    
    # 5. Sauvegarde de prédictions d'exemple
    print("\n🔮 5. SAUVEGARDE DE PRÉDICTIONS D'EXEMPLE")
    print("-" * 50)
    
    # Faire quelques prédictions et les sauvegarder
    best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
    best_model = models[best_model_name]
    
    # Prédictions sur quelques échantillons de test
    sample_predictions = best_model.predict(X_test[:5])
    sample_actual = y_test.iloc[:5]
    
    for i, (pred, actual) in enumerate(zip(sample_predictions, sample_actual)):
        confidence = 1.0 - abs(pred - actual) / actual  # Score de confiance simple
        
        prediction_id = data_loader.save_prediction_to_db(
            model_name=best_model_name,
            predicted_price=pred,
            actual_price=actual,
            model_version="1.0",
            confidence_score=max(0, confidence)
        )
        
        print(f"   🎯 Prédiction {i+1}: {pred:.2f}k$ (réel: {actual:.2f}k$) - ID: {prediction_id}")
    
    # 6. Consultation de l'historique
    print("\n📈 6. HISTORIQUE DES MODÈLES ET PRÉDICTIONS")
    print("-" * 50)
    
    # Historique des modèles
    model_history = db_manager.get_model_history()
    if not model_history.empty:
        print("🤖 Historique des modèles entraînés:")
        print(model_history[['model_name', 'training_date', 'rmse', 'r2_score']].to_string(index=False))
    
    print()
    
    # Historique des prédictions
    predictions_history = db_manager.get_predictions_history(limit=10)
    if not predictions_history.empty:
        print("🔮 Dernières prédictions:")
        print(predictions_history[['model_name', 'predicted_price', 'actual_price', 'prediction_date']].to_string(index=False))
    
    # 7. Statistiques finales
    print("\n📊 7. STATISTIQUES FINALES DE LA BASE")
    print("-" * 50)
    
    final_stats = db_manager.get_database_stats()
    print(f"📋 Total propriétés: {final_stats['properties']}")
    print(f"🔮 Total prédictions: {final_stats['predictions']}")
    print(f"🤖 Total modèles entraînés: {final_stats['trained_models']}")
    
    # Meilleur modèle
    best_rmse = results[best_model_name]['RMSE']
    best_r2 = results[best_model_name]['R2']
    
    print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
    print(f"   📈 RMSE: {best_rmse:.4f}")
    print(f"   📈 R²: {best_r2:.4f}")
    print(f"   📈 Précision: {best_r2*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("🎉 DÉMONSTRATION AVEC BASE DE DONNÉES TERMINÉE ! 🎉")
    print("=" * 70)
    print(f"💾 Base de données disponible: {db_manager.db_path}")
    print("🔍 Vous pouvez explorer la base avec un outil SQLite")

if __name__ == "__main__":
    main()