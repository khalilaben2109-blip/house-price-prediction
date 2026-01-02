"""
Script d'initialisation automatique pour GitHub
"""
import subprocess
import os
import sys

def run_command(command, description=""):
    """Exécute une commande avec gestion d'erreur"""
    if description:
        print(f"🔧 {description}")
    
    print(f"💻 Commande: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Succès!")
            if result.stdout.strip():
                print(f"📤 {result.stdout.strip()}")
        else:
            print("❌ Erreur!")
            if result.stderr.strip():
                print(f"🚨 {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    
    print()
    return True

def init_github_repo():
    """Initialise le repository GitHub"""
    print("🚀 INITIALISATION DU REPOSITORY GITHUB")
    print("=" * 50)
    
    # Demander le nom d'utilisateur GitHub
    username = input("👤 Entrez votre nom d'utilisateur GitHub: ").strip()
    if not username:
        print("❌ Nom d'utilisateur requis!")
        return False
    
    repo_name = "house-price-prediction"
    
    print(f"\n📋 Configuration:")
    print(f"   👤 Utilisateur: {username}")
    print(f"   📁 Repository: {repo_name}")
    print(f"   🔗 URL: https://github.com/{username}/{repo_name}")
    
    confirm = input("\n✅ Confirmer l'initialisation ? (o/n): ").lower()
    if confirm not in ['o', 'oui', 'y', 'yes']:
        print("❌ Initialisation annulée")
        return False
    
    print("\n🔄 Initialisation en cours...")
    
    # Vérifier si Git est installé
    if not run_command("git --version", "Vérification de Git"):
        print("❌ Git n'est pas installé. Installez Git d'abord.")
        return False
    
    # Initialiser Git si nécessaire
    if not os.path.exists('.git'):
        if not run_command("git init", "Initialisation du repository Git"):
            return False
    
    # Configuration Git (optionnel)
    email = input("📧 Email Git (optionnel, Entrée pour ignorer): ").strip()
    if email:
        run_command(f'git config user.email "{email}"', "Configuration email Git")
    
    name = input("👤 Nom Git (optionnel, Entrée pour ignorer): ").strip()
    if name:
        run_command(f'git config user.name "{name}"', "Configuration nom Git")
    
    # Ajouter tous les fichiers
    if not run_command("git add .", "Ajout de tous les fichiers"):
        return False
    
    # Commit initial
    commit_message = """🎉 Initial commit: Complete ML house price prediction project

✨ Features:
- 5 ML algorithms (Linear Regression, Random Forest, XGBoost, Gradient Boosting, SVR)
- Interactive Streamlit web interfaces
- SQLite database integration
- 98.7% accuracy on mixed datasets
- Professional architecture and documentation"""
    
    if not run_command(f'git commit -m "{commit_message}"', "Commit initial"):
        return False
    
    # Ajouter l'origine GitHub
    origin_url = f"https://github.com/{username}/{repo_name}.git"
    if not run_command(f"git remote add origin {origin_url}", "Ajout de l'origine GitHub"):
        # Peut-être que l'origine existe déjà
        run_command(f"git remote set-url origin {origin_url}", "Mise à jour de l'origine GitHub")
    
    # Créer la branche main
    run_command("git branch -M main", "Configuration de la branche main")
    
    # Pousser vers GitHub
    print("🚀 Poussée vers GitHub...")
    print("⚠️  Si c'est la première fois, vous devrez peut-être vous authentifier")
    
    if run_command("git push -u origin main", "Poussée vers GitHub"):
        print("\n" + "=" * 50)
        print("🎉 SUCCÈS ! PROJET PUBLIÉ SUR GITHUB !")
        print("=" * 50)
        print(f"🔗 Votre projet est maintenant disponible à:")
        print(f"   https://github.com/{username}/{repo_name}")
        print()
        print("📋 Prochaines étapes recommandées:")
        print("   1. Allez sur GitHub et vérifiez que tout est correct")
        print("   2. Ajoutez une description au repository")
        print("   3. Ajoutez des topics (machine-learning, python, streamlit, etc.)")
        print("   4. Créez une release v1.0.0")
        print("   5. Partagez votre projet !")
        
        return True
    else:
        print("\n❌ Erreur lors de la poussée vers GitHub")
        print("💡 Vérifiez:")
        print("   - Que le repository existe sur GitHub")
        print("   - Vos permissions d'accès")
        print("   - Votre authentification Git")
        return False

def create_github_repository_instructions():
    """Affiche les instructions pour créer le repository sur GitHub"""
    print("\n📋 INSTRUCTIONS POUR CRÉER LE REPOSITORY SUR GITHUB:")
    print("=" * 60)
    print("1. 🌐 Allez sur https://github.com")
    print("2. ➕ Cliquez sur 'New repository'")
    print("3. 📝 Nom: house-price-prediction")
    print("4. 📄 Description: 🏠 Advanced ML project for house price prediction with 5 algorithms, web interfaces, and 98.7% accuracy")
    print("5. 🔓 Public (recommandé pour portfolio)")
    print("6. ✅ Cochez 'Add a README file'")
    print("7. ⚖️ Choisissez 'MIT License'")
    print("8. 🚀 Cliquez 'Create repository'")
    print("9. 🔄 Revenez ici et relancez ce script")
    print("=" * 60)

def main():
    """Fonction principale"""
    print("🏠 HOUSE PRICE PREDICTION - GITHUB SETUP")
    print("=" * 50)
    
    # Vérifier si on est dans le bon dossier
    if not os.path.exists('src') or not os.path.exists('app.py'):
        print("❌ Ce script doit être exécuté dans le dossier du projet")
        print("📁 Assurez-vous d'être dans le dossier contenant 'src/' et 'app.py'")
        return
    
    print("✅ Dossier de projet détecté")
    
    choice = input("\n❓ Le repository GitHub existe-t-il déjà ? (o/n): ").lower()
    
    if choice not in ['o', 'oui', 'y', 'yes']:
        create_github_repository_instructions()
        input("\n⏸️  Appuyez sur Entrée après avoir créé le repository sur GitHub...")
    
    # Initialiser le repository
    if init_github_repo():
        print("\n🎊 Félicitations ! Votre projet est maintenant sur GitHub !")
    else:
        print("\n😞 Échec de l'initialisation. Consultez les messages d'erreur ci-dessus.")

if __name__ == "__main__":
    main()