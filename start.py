"""
Script de démarrage rapide du projet
"""
import subprocess
import sys
import os
from pathlib import Path

def print_logo():
    """Affiche le logo du projet"""
    logo = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🏠 PRÉDICTION DES PRIX DES MAISONS 🏠                    ║
    ║                                                              ║
    ║    🤖 Machine Learning Project                               ║
    ║    📊 Linear Regression + Random Forest                     ║
    ║    🎯 Interface Web Interactive                              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(logo)

def check_installation():
    """Vérifie l'installation des dépendances"""
    print("🔍 Vérification des dépendances...")
    
    required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'streamlit']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n🚨 Packages manquants: {', '.join(missing)}")
        print("💡 Installation automatique...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return False
    
    return True

def show_menu():
    """Affiche le menu principal"""
    print("\n" + "="*60)
    print("🎯 QUE VOULEZ-VOUS FAIRE ?")
    print("="*60)
    
    options = {
        "1": "🚀 Démonstration complète (demo.py)",
        "2": "🌐 Interface web interactive (Streamlit)",
        "3": "📊 Entraînement des modèles de base",
        "4": "⚙️ Optimisation des hyperparamètres", 
        "5": "📄 Génération de rapport PDF",
        "6": "🧪 Tests unitaires",
        "7": "📋 Présentation guidée",
        "8": "💾 Démonstration avec base de données",
        "9": "🗄️ Interface de gestion de base de données",
        "10": "🚀 Démonstration avancée (5 modèles + données diversifiées)",
        "11": "📚 Ouvrir la documentation",
        "12": "❌ Quitter"
    }
    
    for key, value in options.items():
        print(f"  {key}. {value}")
    
    print("="*60)
    return input("👉 Votre choix (1-9): ").strip()

def run_option(choice):
    """Exécute l'option choisie"""
    commands = {
        "1": "python demo.py",
        "2": "streamlit run app.py",
        "3": "python src/main.py",
        "4": "python src/optimize_models.py",
        "5": "python src/reports/report_generator.py",
        "6": "python tests/test_models.py",
        "7": "python presentation.py",
        "8": "python demo_database.py",
        "9": "streamlit run database_app.py --server.port 8502",
        "10": "python demo_advanced.py",
        "11": "start README.md" if os.name == 'nt' else "open README.md"
    }
    
    if choice in commands:
        print(f"\n🔧 Exécution: {commands[choice]}")
        print("-" * 50)
        
        try:
            if choice == "2":  # Streamlit principal
                print("🌐 Lancement de l'interface web...")
                print("📱 Accessible sur: http://localhost:8501")
                print("🛑 Appuyez sur Ctrl+C pour arrêter")
            elif choice == "9":  # Interface base de données
                print("🗄️ Lancement de l'interface de gestion de base de données...")
                print("📱 Accessible sur: http://localhost:8502")
                print("🛑 Appuyez sur Ctrl+C pour arrêter")
            
            subprocess.run(commands[choice], shell=True)
            
        except KeyboardInterrupt:
            print("\n🛑 Arrêté par l'utilisateur")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    elif choice == "12":
        print("👋 Au revoir!")
        return False
    
    else:
        print("❌ Option invalide!")
    
    return True

def show_quick_info():
    """Affiche des informations rapides"""
    info = """
    📋 INFORMATIONS RAPIDES:
    
    🎯 Objectif: Prédire les prix des maisons avec ML
    📊 Dataset: 506 échantillons, 13 features
    🤖 Modèles: Linear Regression, Random Forest
    📈 Métriques: RMSE, MAE, R², MSE
    
    🚀 Démarrage rapide:
    • Option 1: Démonstration complète automatique
    • Option 2: Interface web interactive
    • Option 7: Présentation guidée pas à pas
    
    📚 Documentation complète dans README.md
    """
    print(info)

def main():
    """Fonction principale"""
    print_logo()
    
    # Vérification des dépendances
    if not check_installation():
        print("\n🔄 Redémarrez le script après l'installation")
        return
    
    show_quick_info()
    
    # Boucle principale
    while True:
        choice = show_menu()
        
        if not run_option(choice):
            break
        
        if choice != "9":
            input("\n⏸️  Appuyez sur Entrée pour revenir au menu...")

if __name__ == "__main__":
    main()