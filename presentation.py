"""
Script de présentation automatique du projet
"""
import sys
import os
import time
import subprocess
from pathlib import Path

def print_banner(text, char="=", width=80):
    """Affiche un banner stylisé"""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")

def print_step(step_num, title, description=""):
    """Affiche une étape de la présentation"""
    print(f"🔹 ÉTAPE {step_num}: {title}")
    if description:
        print(f"   {description}")
    print()

def wait_for_user(message="Appuyez sur Entrée pour continuer..."):
    """Attend l'input utilisateur"""
    input(f"⏸️  {message}")

def run_command(command, description=""):
    """Exécute une commande avec affichage"""
    if description:
        print(f"🔧 {description}")
    print(f"💻 Commande: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Succès!")
            if result.stdout:
                print(f"📤 Sortie: {result.stdout[:200]}...")
        else:
            print("❌ Erreur!")
            if result.stderr:
                print(f"🚨 Erreur: {result.stderr[:200]}...")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

def check_requirements():
    """Vérifie les prérequis"""
    print_step(1, "VÉRIFICATION DES PRÉREQUIS")
    
    # Vérifier Python
    try:
        import sys
        python_version = sys.version.split()[0]
        print(f"✅ Python {python_version} détecté")
    except:
        print("❌ Python non trouvé")
        return False
    
    # Vérifier les modules principaux
    required_modules = ['pandas', 'numpy', 'sklearn', 'matplotlib']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} installé")
        except ImportError:
            print(f"❌ {module} manquant")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n🚨 Modules manquants: {', '.join(missing_modules)}")
        print("💡 Exécutez: pip install -r requirements.txt")
        return False
    
    return True

def show_project_structure():
    """Affiche la structure du projet"""
    print_step(2, "STRUCTURE DU PROJET")
    
    structure = """
    house-price-prediction/
    ├── 🏠 app.py                     # Interface web Streamlit
    ├── 🎯 demo.py                    # Démonstration complète
    ├── 📋 presentation.py            # Ce script de présentation
    ├── 📄 README.md                  # Documentation principale
    ├── 📦 requirements.txt           # Dépendances Python
    ├── 🚫 .gitignore                # Fichiers à ignorer
    │
    ├── 📁 src/                       # Code source principal
    │   ├── 📊 data/                  # Gestion des données
    │   ├── 🤖 models/                # Modèles ML
    │   ├── 📈 evaluation/            # Évaluation
    │   ├── 🎨 visualization/         # Visualisations
    │   ├── ⚙️ optimization/          # Optimisation
    │   ├── 📄 reports/               # Génération de rapports
    │   └── 🛠️ utils/                 # Utilitaires
    │
    ├── 📓 notebooks/                 # Jupyter notebooks
    ├── 🧪 tests/                     # Tests unitaires
    ├── ⚙️ config/                    # Configuration
    ├── 💾 data/                      # Données (raw/processed)
    ├── 🤖 models/                    # Modèles sauvegardés
    └── 📋 logs/                      # Fichiers de log
    """
    
    print(structure)

def demo_basic_functionality():
    """Démonstration des fonctionnalités de base"""
    print_step(3, "DÉMONSTRATION DES FONCTIONNALITÉS DE BASE")
    
    print("🔹 Test des modèles de base...")
    run_command("python src/main.py", "Entraînement des modèles Linear Regression et Random Forest")
    
    wait_for_user("Voulez-vous voir les tests unitaires ?")
    
    print("🔹 Exécution des tests unitaires...")
    run_command("python tests/test_models.py", "Validation des composants")

def demo_advanced_features():
    """Démonstration des fonctionnalités avancées"""
    print_step(4, "FONCTIONNALITÉS AVANCÉES")
    
    print("🔹 Optimisation des hyperparamètres...")
    run_command("python src/optimize_models.py", "Recherche des meilleurs paramètres")
    
    wait_for_user("Voulez-vous générer un rapport PDF ?")
    
    print("🔹 Génération du rapport PDF...")
    run_command("python src/reports/report_generator.py", "Création du rapport automatique")

def demo_web_interface():
    """Démonstration de l'interface web"""
    print_step(5, "INTERFACE WEB INTERACTIVE")
    
    print("🌐 Lancement de l'interface web Streamlit...")
    print("📱 L'interface sera accessible à: http://localhost:8501")
    print("🔧 Fonctionnalités disponibles:")
    print("   • Exploration interactive des données")
    print("   • Entraînement de modèles en temps réel")
    print("   • Prédictions personnalisées")
    print("   • Optimisation des hyperparamètres")
    print("   • Visualisations dynamiques")
    
    wait_for_user("Appuyez sur Entrée pour lancer Streamlit (Ctrl+C pour arrêter)")
    
    try:
        subprocess.run("streamlit run app.py", shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Interface web fermée")

def show_results_summary():
    """Affiche un résumé des résultats"""
    print_step(6, "RÉSUMÉ DES RÉSULTATS")
    
    # Lire les résultats s'ils existent
    results_file = Path("data/processed/model_results.csv")
    if results_file.exists():
        try:
            import pandas as pd
            results_df = pd.read_csv(results_file, index_col=0)
            
            print("📊 PERFORMANCES DES MODÈLES:")
            print(results_df.round(4))
            
            best_model = results_df['RMSE'].idxmin()
            best_rmse = results_df.loc[best_model, 'RMSE']
            best_r2 = results_df.loc[best_model, 'R2']
            
            print(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
            print(f"📈 RMSE: {best_rmse:.4f}")
            print(f"📈 R²: {best_r2:.4f} ({best_r2*100:.2f}% de précision)")
            
        except Exception as e:
            print(f"❌ Erreur lors de la lecture des résultats: {e}")
    else:
        print("📋 Aucun résultat trouvé. Exécutez d'abord la démonstration complète.")

def show_next_steps():
    """Affiche les prochaines étapes"""
    print_step(7, "PROCHAINES ÉTAPES ET EXTENSIONS")
    
    next_steps = """
    🚀 EXTENSIONS POSSIBLES:
    
    📊 Données et Features:
    • Intégrer de vrais datasets immobiliers (Kaggle, APIs)
    • Feature engineering avancé (nouvelles variables)
    • Gestion des données temporelles
    
    🤖 Modèles:
    • Ajouter XGBoost, LightGBM, CatBoost
    • Réseaux de neurones (TensorFlow/PyTorch)
    • Ensemble methods avancés
    
    🔧 MLOps:
    • Pipeline CI/CD avec GitHub Actions
    • Monitoring des modèles en production
    • A/B testing des modèles
    • Versioning des modèles avec MLflow
    
    🌐 Déploiement:
    • API REST avec FastAPI
    • Application mobile
    • Dashboard en temps réel
    • Intégration cloud (AWS, GCP, Azure)
    
    📈 Analytics:
    • Explainability (SHAP, LIME)
    • Détection de drift des données
    • Alertes automatiques
    • Rapports automatisés
    """
    
    print(next_steps)

def main():
    """Fonction principale de présentation"""
    print_banner("🏠 PRÉSENTATION DU PROJET PRÉDICTION PRIX MAISONS 🏠", "🏠", 80)
    
    print("👋 Bienvenue dans la présentation interactive du projet!")
    print("🎯 Ce script vous guidera à travers toutes les fonctionnalités.")
    print("⏱️  Durée estimée: 10-15 minutes")
    
    wait_for_user("Prêt à commencer ?")
    
    # Étape 1: Vérification des prérequis
    if not check_requirements():
        print("🚨 Veuillez installer les dépendances avant de continuer.")
        return
    
    wait_for_user()
    
    # Étape 2: Structure du projet
    show_project_structure()
    wait_for_user()
    
    # Étape 3: Fonctionnalités de base
    demo_basic_functionality()
    wait_for_user()
    
    # Étape 4: Fonctionnalités avancées
    demo_advanced_features()
    wait_for_user()
    
    # Étape 5: Interface web
    response = input("🌐 Voulez-vous lancer l'interface web ? (o/n): ")
    if response.lower() in ['o', 'oui', 'y', 'yes']:
        demo_web_interface()
    
    # Étape 6: Résumé des résultats
    show_results_summary()
    wait_for_user()
    
    # Étape 7: Prochaines étapes
    show_next_steps()
    
    # Conclusion
    print_banner("🎉 PRÉSENTATION TERMINÉE 🎉", "🎉", 80)
    
    print("✅ Vous avez découvert toutes les fonctionnalités du projet!")
    print("📚 Consultez le README.md pour plus de détails")
    print("🌐 Lancez 'streamlit run app.py' pour l'interface web")
    print("📄 Générez des rapports avec 'python src/reports/report_generator.py'")
    print("🤖 Entraînez les modèles avec 'python demo.py'")
    
    print("\n💡 N'hésitez pas à explorer le code et à l'adapter à vos besoins!")
    print("🙏 Merci d'avoir suivi cette présentation!")

if __name__ == "__main__":
    main()