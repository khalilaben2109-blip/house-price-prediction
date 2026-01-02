# 🤝 Guide de Contribution

Merci de votre intérêt pour contribuer au projet **Prédiction des Prix des Maisons** ! 

## 🎯 Comment Contribuer

### 🐛 Signaler des Bugs
1. Vérifiez que le bug n'a pas déjà été signalé
2. Créez une issue avec le template "Bug Report"
3. Incluez les informations suivantes :
   - Version de Python
   - Système d'exploitation
   - Étapes pour reproduire le bug
   - Comportement attendu vs observé
   - Logs d'erreur si disponibles

### ✨ Proposer des Fonctionnalités
1. Créez une issue avec le template "Feature Request"
2. Décrivez clairement la fonctionnalité souhaitée
3. Expliquez pourquoi elle serait utile
4. Proposez une implémentation si possible

### 🔧 Contribuer au Code

#### Prérequis
- Python 3.8+
- Git
- Connaissance de base en machine learning

#### Processus de Développement
1. **Fork** le repository
2. **Clone** votre fork localement
```bash
git clone https://github.com/votre-username/house-price-prediction.git
cd house-price-prediction
```

3. **Créez une branche** pour votre fonctionnalité
```bash
git checkout -b feature/nouvelle-fonctionnalite
```

4. **Installez** les dépendances de développement
```bash
pip install -r requirements.txt
```

5. **Développez** votre fonctionnalité
   - Suivez les conventions de code existantes
   - Ajoutez des tests pour votre code
   - Documentez vos fonctions

6. **Testez** votre code
```bash
python tests/test_models.py
python demo.py  # Test complet
```

7. **Committez** vos changements
```bash
git add .
git commit -m "feat: ajouter nouvelle fonctionnalité"
```

8. **Poussez** vers votre fork
```bash
git push origin feature/nouvelle-fonctionnalite
```

9. **Créez une Pull Request**

## 📝 Standards de Code

### Style de Code
- Suivre PEP 8 pour Python
- Utiliser des noms de variables descriptifs
- Commenter le code complexe
- Docstrings pour toutes les fonctions publiques

### Structure des Commits
Utiliser le format [Conventional Commits](https://www.conventionalcommits.org/) :

```
type(scope): description

[corps optionnel]

[footer optionnel]
```

**Types :**
- `feat`: nouvelle fonctionnalité
- `fix`: correction de bug
- `docs`: documentation
- `style`: formatage, pas de changement de code
- `refactor`: refactoring du code
- `test`: ajout ou modification de tests
- `chore`: tâches de maintenance

**Exemples :**
```
feat(models): ajouter support pour XGBoost
fix(preprocessing): corriger la gestion des valeurs manquantes
docs(readme): mettre à jour les instructions d'installation
```

### Tests
- Ajouter des tests pour toute nouvelle fonctionnalité
- Maintenir une couverture de test élevée
- Tester les cas limites et d'erreur

### Documentation
- Mettre à jour le README si nécessaire
- Documenter les nouvelles APIs
- Ajouter des exemples d'utilisation

## 🏗️ Architecture du Projet

### Structure des Dossiers
```
src/
├── data/           # Gestion des données
├── models/         # Modèles ML
├── evaluation/     # Évaluation des modèles
├── visualization/  # Visualisations
├── optimization/   # Optimisation des hyperparamètres
├── reports/        # Génération de rapports
└── utils/          # Utilitaires
```

### Conventions de Nommage
- **Classes** : PascalCase (`LinearRegressionModel`)
- **Fonctions/Variables** : snake_case (`train_model`)
- **Constantes** : UPPER_CASE (`MAX_ITERATIONS`)
- **Fichiers** : snake_case (`data_loader.py`)

## 🎨 Domaines de Contribution

### 🤖 Machine Learning
- Nouveaux algorithmes (XGBoost, Neural Networks, SVM)
- Techniques d'ensemble avancées
- Feature engineering automatique
- AutoML integration

### 📊 Visualisations
- Nouveaux types de graphiques
- Dashboards interactifs
- Visualisations 3D
- Animations

### 🌐 Interface Utilisateur
- Améliorations Streamlit
- Interface mobile
- API REST
- Interface en ligne de commande

### 🔧 Infrastructure
- Pipeline CI/CD
- Containerisation Docker
- Déploiement cloud
- Monitoring et logging

### 📚 Documentation
- Tutoriels
- Exemples d'utilisation
- Traductions
- Vidéos explicatives

## 🏆 Reconnaissance

Les contributeurs seront reconnus de plusieurs façons :
- Mention dans le README
- Badge de contributeur
- Invitation à rejoindre l'équipe de maintenance
- Recommandations LinkedIn

## 📞 Support

- **Issues GitHub** : Pour les bugs et fonctionnalités
- **Discussions** : Pour les questions générales
- **Email** : [votre-email] pour les questions privées

## 📄 Licence

En contribuant, vous acceptez que vos contributions soient sous licence MIT.

---

**Merci de contribuer à rendre ce projet encore meilleur ! 🚀**