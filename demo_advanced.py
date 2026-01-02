"""
Démonstration avancée avec 5 modèles et données diversifiées
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import DataLoader
from data.data_generator import DataGenerator
from data.preprocessor import DataPreprocessor
from models.linear_regression_model import LinearRegressionModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.gradient_boosting_model import GradientBoostingModel
from models.support_vector_model import SupportVectorModel
from evaluation.evaluator import ModelEvaluator
from visualization.visualizer import DataVisualizer
from utils.logger import setup_logger
import pandas as pd
import numpy as np
import time

def main():
    print("=" * 80)
    print("🚀 DÉMONSTRATION AVANCÉE - 5 MODÈLES & DONNÉES DIVERSIFIÉES 🚀")
    print("=" * 80)
    
    logger = setup_logger()
    
    # 1. Test de différentes sources de données
    print("\n📊 1. TEST DE DIFFÉRENTES SOURCES DE DONNÉES")
    print("-" * 60)
    
    data_sources = ['mixed', 'california', 'synthetic']
    datasets = {}
    
    for source in data_sources:
        print(f"\n🔍 Test de la source: {source}")
        try:
            data_loader = DataLoader(use_database=False, data_source=source)
            X, y = data_loader.load_boston_housing()
            
            # Informations sur le dataset
            generator = DataGenerator()
            info = generator.get_dataset_info(X, y)
            
            datasets[source] = {
                'X': X, 'y': y, 'info': info,
                'loader': data_loader
            }
            
            print(f"✅ {source.capitalize()}: {info['n_samples']} échantillons, {info['n_features']} features")
            print(f"   Prix: {info['price_mean']:.0f}€ ± {info['price_std']:.0f}€")
            
        except Exception as e:
            print(f"❌ Erreur avec {source}: {e}")
    
    # Choisir le meilleur dataset (le plus grand)
    if datasets:
        best_source = max(datasets.keys(), key=lambda k: datasets[k]['info']['n_samples'])
        X, y = datasets[best_source]['X'], datasets[best_source]['y']
        data_loader = datasets[best_source]['loader']
        
        print(f"\n🏆 Dataset sélectionné: {best_source}")
        print(f"   📊 {len(X)} échantillons, {len(X.columns)} features")
        print(f"   💰 Prix moyen: {y.mean():.0f}€")
    else:
        print("❌ Aucun dataset disponible, arrêt de la démonstration")
        return
    
    # 2. Visualisation des données
    print("\n📈 2. VISUALISATION DES DONNÉES")
    print("-" * 60)
    
    try:
        visualizer = DataVisualizer()
        print("✅ Génération des graphiques d'exploration...")
        # visualizer.plot_data_distribution(X, y)  # Commenté pour éviter l'affichage
    except Exception as e:
        print(f"⚠️  Erreur de visualisation: {e}")
    
    # 3. Preprocessing
    print("\n🔧 3. PREPROCESSING DES DONNÉES")
    print("-" * 60)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(X, y, test_size=0.2)
    
    print(f"✅ Données préparées:")
    print(f"   🎯 Entraînement: {X_train.shape[0]} échantillons")
    print(f"   🧪 Test: {X_test.shape[0]} échantillons")
    print(f"   📊 Features: {X_train.shape[1]}")
    
    # 4. Entraînement des 5 modèles
    print("\n🤖 4. ENTRAÎNEMENT DE 5 MODÈLES AVANCÉS")
    print("-" * 60)
    
    models = {
        'Linear Regression': LinearRegressionModel(),
        'Random Forest': RandomForestModel(n_estimators=100),
        'XGBoost': XGBoostModel(n_estimators=100),
        'Gradient Boosting': GradientBoostingModel(n_estimators=100),
        'Support Vector': SupportVectorModel(kernel='rbf', C=1.0)
    }
    
    evaluator = ModelEvaluator()
    results = {}
    predictions_dict = {}
    training_times = {}
    
    print("🚀 Entraînement en cours...")
    
    for name, model in models.items():
        print(f"\n   🔄 {name}...")
        
        start_time = time.time()
        
        try:
            # Entraîner le modèle
            model.train(X_train, y_train)
            
            # Faire des prédictions
            predictions = model.predict(X_test)
            
            # Évaluer
            metrics = evaluator.evaluate(y_test, predictions)
            
            # Stocker les résultats
            results[name] = metrics
            predictions_dict[name] = predictions
            training_times[name] = time.time() - start_time
            
            print(f"   ✅ {name}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f} ({training_times[name]:.2f}s)")
            
            # Sauvegarder en base si disponible
            if hasattr(data_loader, 'save_model_results_to_db'):
                hyperparams = {}
                if hasattr(model.model, 'get_params'):
                    hyperparams = model.model.get_params()
                
                data_loader.save_model_results_to_db(
                    model_name=name,
                    model_version="2.0",
                    metrics=metrics,
                    hyperparameters=hyperparams,
                    training_samples=len(X_train),
                    test_samples=len(X_test)
                )
            
        except Exception as e:
            print(f"   ❌ Erreur avec {name}: {e}")
            # Continuer avec les autres modèles
            continue
    
    # 5. Comparaison des résultats
    print("\n📊 5. COMPARAISON DES PERFORMANCES")
    print("-" * 60)
    
    if results:
        # Créer un DataFrame des résultats
        results_df = pd.DataFrame(results).T
        results_df['Training_Time'] = [training_times.get(model, 0) for model in results_df.index]
        
        print("\n=== TABLEAU COMPLET DES RÉSULTATS ===")
        print(results_df.round(4))
        
        # Analyse des performances
        best_rmse = results_df['RMSE'].idxmin()
        best_r2 = results_df['R2'].idxmax()
        fastest = results_df['Training_Time'].idxmin()
        
        print(f"\n🏆 ANALYSE DES PERFORMANCES:")
        print(f"   🎯 Meilleur RMSE: {best_rmse} ({results_df.loc[best_rmse, 'RMSE']:.4f})")
        print(f"   📈 Meilleur R²: {best_r2} ({results_df.loc[best_r2, 'R2']:.4f})")
        print(f"   ⚡ Plus rapide: {fastest} ({results_df.loc[fastest, 'Training_Time']:.2f}s)")
        
        # Graphiques de comparaison
        try:
            evaluator.compare_models(results)
            # visualizer.plot_model_comparison(results)  # Commenté pour éviter l'affichage
        except Exception as e:
            print(f"⚠️  Erreur de visualisation: {e}")
    
    # 6. Analyse détaillée du meilleur modèle
    print("\n🔍 6. ANALYSE DÉTAILLÉE DU MEILLEUR MODÈLE")
    print("-" * 60)
    
    if results:
        best_model_name = results_df['RMSE'].idxmin()
        best_predictions = predictions_dict[best_model_name]
        best_model = models[best_model_name]
        
        print(f"🏆 Modèle sélectionné: {best_model_name}")
        
        # Importance des features (si disponible)
        feature_importance = best_model.get_feature_importance()
        if feature_importance and 'feature_importance' in feature_importance:
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importance['feature_importance']
            }).sort_values('Importance', ascending=False)
            
            print("\n📊 Top 5 Features les plus importantes:")
            for i, row in importance_df.head().iterrows():
                print(f"   {row['Feature']}: {row['Importance']:.4f}")
        
        # Quelques prédictions d'exemple
        print(f"\n🎯 Exemples de prédictions ({best_model_name}):")
        for i in range(min(5, len(best_predictions))):
            actual = y_test.iloc[i]
            predicted = best_predictions[i]
            error = abs(actual - predicted)
            error_pct = (error / actual) * 100
            
            print(f"   Propriété {i+1}: {predicted:.0f}€ (réel: {actual:.0f}€, erreur: {error_pct:.1f}%)")
            
            # Sauvegarder quelques prédictions
            if hasattr(data_loader, 'save_prediction_to_db'):
                confidence = max(0, 1 - (error / actual))
                data_loader.save_prediction_to_db(
                    model_name=best_model_name,
                    predicted_price=predicted,
                    actual_price=actual,
                    model_version="2.0",
                    confidence_score=confidence
                )
    
    # 7. Recommandations
    print("\n💡 7. RECOMMANDATIONS")
    print("-" * 60)
    
    if results:
        print("📋 Analyse des résultats:")
        
        # Analyser les performances
        avg_rmse = results_df['RMSE'].mean()
        avg_r2 = results_df['R2'].mean()
        
        if avg_r2 > 0.8:
            print("   ✅ Excellentes performances globales (R² > 0.8)")
        elif avg_r2 > 0.6:
            print("   ✅ Bonnes performances globales (R² > 0.6)")
        else:
            print("   ⚠️  Performances moyennes, considérer:")
            print("      • Plus de données d'entraînement")
            print("      • Feature engineering avancé")
            print("      • Hyperparameter tuning plus poussé")
        
        # Recommandations par modèle
        if 'XGBoost' in results and results['XGBoost']['R2'] > 0.7:
            print("   🚀 XGBoost recommandé pour la production")
        elif 'Random Forest' in results and results['Random Forest']['R2'] > 0.7:
            print("   🌲 Random Forest recommandé (bon compromis performance/interprétabilité)")
        elif 'Linear Regression' in results and results['Linear Regression']['R2'] > 0.8:
            print("   📈 Linear Regression surprenamment efficace (données linéaires)")
    
    # 8. Sauvegarde des résultats
    print("\n💾 8. SAUVEGARDE DES RÉSULTATS")
    print("-" * 60)
    
    if results:
        # Sauvegarder les résultats détaillés
        results_df.to_csv('data/processed/advanced_model_results.csv')
        print("✅ Résultats sauvegardés dans data/processed/advanced_model_results.csv")
        
        # Créer un rapport de synthèse
        summary = {
            'dataset_source': best_source,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'best_model': best_model_name,
            'best_rmse': results_df.loc[best_model_name, 'RMSE'],
            'best_r2': results_df.loc[best_model_name, 'R2'],
            'avg_performance': avg_r2,
            'models_tested': len(results)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('data/processed/advanced_summary.csv', index=False)
        print("✅ Résumé sauvegardé dans data/processed/advanced_summary.csv")
    
    print("\n" + "=" * 80)
    print("🎉 DÉMONSTRATION AVANCÉE TERMINÉE AVEC SUCCÈS ! 🎉")
    print("=" * 80)
    
    if results:
        print(f"🏆 Meilleur modèle: {best_model_name}")
        print(f"📊 Dataset utilisé: {best_source} ({len(X)} échantillons)")
        print(f"🎯 Performance: RMSE={results_df.loc[best_model_name, 'RMSE']:.4f}, R²={results_df.loc[best_model_name, 'R2']:.4f}")
        print(f"🚀 {len(results)} modèles testés avec succès")

if __name__ == "__main__":
    main()