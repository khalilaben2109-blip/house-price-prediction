"""
Application Web Interactive pour la Prédiction des Prix des Maisons
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Configuration de la page
st.set_page_config(
    page_title="🏠 Prédiction Prix Maisons",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter le chemin src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.linear_regression_model import LinearRegressionModel
from models.random_forest_model import RandomForestModel
from evaluation.evaluator import ModelEvaluator
from optimization.hyperparameter_tuner import HyperparameterTuner

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Charge les données avec mise en cache"""
    data_loader = DataLoader()
    return data_loader.load_boston_housing()

def train_models(X_train, y_train, X_test, y_test):
    """Entraîne les modèles et retourne les résultats"""
    models = {
        'Linear Regression': LinearRegressionModel(),
        'Random Forest': RandomForestModel()
    }
    
    evaluator = ModelEvaluator()
    results = {}
    predictions = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Entraînement: {name}...')
        model.train(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = evaluator.evaluate(y_test, pred)
        predictions[name] = pred
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text('Entraînement terminé!')
    return results, predictions

def main():
    # En-tête principal
    st.markdown('<h1 class="main-header">🏠 Prédiction des Prix des Maisons</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Configuration")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📍 Navigation",
        ["🏠 Accueil", "📊 Exploration des Données", "🤖 Modèles ML", "📈 Prédictions", "⚙️ Optimisation"]
    )
    
    # Chargement des données
    if 'data_loaded' not in st.session_state:
        with st.spinner('Chargement des données...'):
            X, y = load_data()
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.data_loaded = True
    
    X, y = st.session_state.X, st.session_state.y
    
    if page == "🏠 Accueil":
        show_home_page(X, y)
    elif page == "📊 Exploration des Données":
        show_data_exploration(X, y)
    elif page == "🤖 Modèles ML":
        show_models_page(X, y)
    elif page == "📈 Prédictions":
        show_predictions_page(X, y)
    elif page == "⚙️ Optimisation":
        show_optimization_page(X, y)

def show_home_page(X, y):
    """Page d'accueil"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Échantillons</h3>
            <h2>{X.shape[0]}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔢 Features</h3>
            <h2>{X.shape[1]}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Prix Moyen</h3>
            <h2>{y.mean():.1f}k$</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Objectif du Projet</h3>
        <p>Ce projet utilise des algorithmes de machine learning pour prédire les prix des maisons 
        en analysant 13 caractéristiques différentes. Nous comparons les performances de la 
        régression linéaire et du Random Forest.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Aperçu des données
    st.subheader("📋 Aperçu des Données")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Features (5 premières lignes)**")
        st.dataframe(X.head(), use_container_width=True)
    
    with col2:
        st.write("**Statistiques du Prix**")
        stats_df = pd.DataFrame({
            'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max'],
            'Valeur (k$)': [y.mean(), y.median(), y.std(), y.min(), y.max()]
        })
        st.dataframe(stats_df, use_container_width=True)

def show_data_exploration(X, y):
    """Page d'exploration des données"""
    st.header("📊 Exploration des Données")
    
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Corrélations", "📋 Statistiques"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution du prix
            fig = px.histogram(y, nbins=30, title="Distribution des Prix des Maisons")
            fig.update_layout(xaxis_title="Prix (k$)", yaxis_title="Fréquence")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boxplot du prix
            fig = px.box(y=y, title="Boxplot des Prix")
            fig.update_layout(yaxis_title="Prix (k$)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des features
        st.subheader("Distribution des Features")
        selected_features = st.multiselect(
            "Sélectionnez les features à visualiser:",
            X.columns.tolist(),
            default=X.columns[:4].tolist()
        )
        
        if selected_features:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=selected_features[:4]
            )
            
            for i, feature in enumerate(selected_features[:4]):
                row = i // 2 + 1
                col = i % 2 + 1
                fig.add_trace(
                    go.Histogram(x=X[feature], name=feature, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title_text="Distribution des Features Sélectionnées")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Matrice de corrélation
        st.subheader("🔗 Matrice de Corrélation")
        
        corr_data = pd.concat([X, y], axis=1)
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matrice de Corrélation",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top corrélations avec le prix
        st.subheader("🎯 Top Corrélations avec le Prix")
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title="Corrélation Absolue avec le Prix"
        )
        fig.update_layout(xaxis_title="Corrélation Absolue", yaxis_title="Features")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📋 Statistiques Descriptives")
        st.dataframe(X.describe(), use_container_width=True)
        
        # Informations sur les données
        st.subheader("ℹ️ Informations sur le Dataset")
        info_data = {
            'Feature': X.columns,
            'Type': [X[col].dtype for col in X.columns],
            'Valeurs Manquantes': [X[col].isnull().sum() for col in X.columns],
            'Valeurs Uniques': [X[col].nunique() for col in X.columns]
        }
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)

def show_models_page(X, y):
    """Page des modèles ML"""
    st.header("🤖 Modèles de Machine Learning")
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    test_size = st.sidebar.slider("Taille du set de test", 0.1, 0.4, 0.2, 0.05)
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(X, y, test_size=test_size)
    
    st.markdown(f"""
    <div class="success-box">
        <h4>✅ Données Préparées</h4>
        <p>Entraînement: {X_train.shape[0]} échantillons | Test: {X_test.shape[0]} échantillons</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Entraîner les Modèles", type="primary"):
        results, predictions = train_models(X_train, y_train, X_test, y_test)
        
        # Affichage des résultats
        st.subheader("📊 Résultats des Modèles")
        
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Graphique de comparaison
        metrics = ['RMSE', 'MAE', 'R2', 'MSE']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics
        )
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Bar(
                    x=results_df.index,
                    y=results_df[metric],
                    name=metric,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Comparaison des Performances")
        st.plotly_chart(fig, use_container_width=True)
        
        # Meilleur modèle
        best_model = results_df['RMSE'].idxmin()
        best_rmse = results_df.loc[best_model, 'RMSE']
        best_r2 = results_df.loc[best_model, 'R2']
        
        st.markdown(f"""
        <div class="success-box">
            <h4>🏆 Meilleur Modèle: {best_model}</h4>
            <p>RMSE: {best_rmse:.4f} | R²: {best_r2:.4f} | Précision: {best_r2*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

def show_predictions_page(X, y):
    """Page des prédictions"""
    st.header("📈 Analyse des Prédictions")
    
    # Interface de prédiction manuelle
    st.subheader("🎯 Prédiction Personnalisée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Configurez les caractéristiques de la maison:**")
        
        # Créer des sliders pour chaque feature
        feature_values = {}
        for i, feature in enumerate(X.columns[:7]):  # Première moitié
            try:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                mean_val = float(X[feature].mean())
                
                feature_values[feature] = st.slider(
                    f"{feature}",
                    min_val, max_val, mean_val,
                    key=f"slider_{feature}"
                )
            except Exception as e:
                st.warning(f"Erreur avec la feature {feature}: {e}")
                feature_values[feature] = 0.0
    
    with col2:
        st.write("**Continuez la configuration:**")
        
        for feature in X.columns[7:]:  # Deuxième moitié
            try:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                mean_val = float(X[feature].mean())
                
                feature_values[feature] = st.slider(
                    f"{feature}",
                    min_val, max_val, mean_val,
                    key=f"slider_{feature}"
                )
            except Exception as e:
                st.warning(f"Erreur avec la feature {feature}: {e}")
                feature_values[feature] = 0.0
    
    if st.button("💡 Prédire le Prix", type="primary"):
        # Préparer les données pour la prédiction
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(X, y)
        
        # Entraîner un modèle rapide
        model = LinearRegressionModel()
        model.train(X_train, y_train)
        
        # Créer le vecteur de prédiction
        input_data = pd.DataFrame([feature_values])
        
        # Normaliser avec le même preprocessor
        input_scaled = preprocessor.scaler.transform(input_data)
        input_df = pd.DataFrame(input_scaled, columns=X.columns)
        
        # Prédiction
        prediction = model.predict(input_df)[0]
        
        st.markdown(f"""
        <div class="success-box">
            <h3>🏠 Prix Prédit: {prediction:.2f}k$</h3>
            <p>Basé sur les caractéristiques saisies</p>
        </div>
        """, unsafe_allow_html=True)

def show_optimization_page(X, y):
    """Page d'optimisation"""
    st.header("⚙️ Optimisation des Hyperparamètres")
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Optimisation Automatique</h4>
        <p>Cette section permet d'optimiser automatiquement les hyperparamètres 
        des modèles pour améliorer leurs performances.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration de l'optimisation
    col1, col2 = st.columns(2)
    
    with col1:
        cv_folds = st.selectbox("Nombre de folds (CV)", [3, 5, 10], index=1)
        search_method = st.selectbox("Méthode de recherche", ["Random Search", "Grid Search"])
    
    with col2:
        n_iter = st.slider("Nombre d'itérations (Random Search)", 10, 100, 50)
        optimize_rf = st.checkbox("Optimiser Random Forest", value=True)
    
    if st.button("🚀 Lancer l'Optimisation", type="primary"):
        # Préparation des données
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(X, y)
        
        # Optimisation
        tuner = HyperparameterTuner(cv_folds=cv_folds)
        
        with st.spinner('Optimisation en cours...'):
            if optimize_rf:
                method = 'random' if search_method == "Random Search" else 'grid'
                best_params = tuner.tune_random_forest(X_train, y_train, method=method)
                
                st.subheader("🎯 Meilleurs Paramètres Trouvés")
                st.json(best_params)
                
                # Test du modèle optimisé
                optimized_models = tuner.get_optimized_models()
                
                if optimized_models:
                    evaluator = ModelEvaluator()
                    results = {}
                    
                    for name, model in optimized_models.items():
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        results[name] = evaluator.evaluate(y_test, predictions)
                    
                    st.subheader("📊 Résultats Optimisés")
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df.round(4), use_container_width=True)
                    
                    # Graphique des améliorations
                    fig = px.bar(
                        results_df.reset_index(),
                        x='index',
                        y=['RMSE', 'MAE'],
                        title="Performance des Modèles Optimisés",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()