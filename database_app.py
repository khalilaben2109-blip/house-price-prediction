"""
Interface Streamlit pour la gestion de la base de données
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Configuration de la page
st.set_page_config(
    page_title="🗄️ Gestion Base de Données",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter le chemin src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.database_manager import DatabaseManager
from data.data_loader import DataLoader

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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_database_manager():
    """Charge le gestionnaire de base de données avec mise en cache"""
    return DatabaseManager()

def main():
    # En-tête principal
    st.markdown('<h1 class="main-header">🗄️ Gestion de la Base de Données</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Configuration")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📍 Navigation",
        ["🏠 Tableau de Bord", "📊 Données", "🤖 Modèles", "🔮 Prédictions", "⚙️ Administration"]
    )
    
    # Initialisation de la base de données
    try:
        db_manager = load_database_manager()
        
        # Vérifier et insérer des données si nécessaire
        stats = db_manager.get_database_stats()
        if stats['properties'] == 0:
            with st.spinner('Initialisation de la base de données...'):
                db_manager.insert_sample_data()
                st.success("✅ Base de données initialisée avec des données d'exemple")
        
    except Exception as e:
        st.error(f"❌ Erreur de connexion à la base de données: {e}")
        return
    
    if page == "🏠 Tableau de Bord":
        show_dashboard(db_manager)
    elif page == "📊 Données":
        show_data_page(db_manager)
    elif page == "🤖 Modèles":
        show_models_page(db_manager)
    elif page == "🔮 Prédictions":
        show_predictions_page(db_manager)
    elif page == "⚙️ Administration":
        show_admin_page(db_manager)

def show_dashboard(db_manager):
    """Tableau de bord principal"""
    st.header("🏠 Tableau de Bord")
    
    # Statistiques générales
    stats = db_manager.get_database_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏘️ Propriétés</h3>
            <h2>{stats['properties']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔮 Prédictions</h3>
            <h2>{stats['predictions']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 Modèles</h3>
            <h2>{stats['trained_models']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphiques du tableau de bord
    col1, col2 = st.columns(2)
    
    with col1:
        # Historique des modèles
        model_history = db_manager.get_model_history()
        if not model_history.empty:
            fig = px.bar(
                model_history.groupby('model_name')['rmse'].mean().reset_index(),
                x='model_name',
                y='rmse',
                title="Performance Moyenne des Modèles (RMSE)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun modèle entraîné trouvé")
    
    with col2:
        # Évolution des prédictions
        predictions_history = db_manager.get_predictions_history(limit=50)
        if not predictions_history.empty:
            predictions_history['prediction_date'] = pd.to_datetime(predictions_history['prediction_date'])
            
            fig = px.line(
                predictions_history,
                x='prediction_date',
                y='predicted_price',
                color='model_name',
                title="Évolution des Prédictions dans le Temps"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune prédiction trouvée")

def show_data_page(db_manager):
    """Page de gestion des données"""
    st.header("📊 Gestion des Données")
    
    tab1, tab2, tab3 = st.tabs(["📋 Propriétés", "📈 Analyse", "➕ Ajouter"])
    
    with tab1:
        st.subheader("📋 Liste des Propriétés")
        
        # Charger les données
        data_loader = DataLoader(use_database=True)
        X, y = data_loader.load_boston_housing()
        
        # Combiner X et y pour l'affichage
        df_display = pd.concat([X, y], axis=1)
        
        # Identifier la colonne de prix
        price_column = y.name if hasattr(y, 'name') and y.name else 'prix'
        
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            price_range = st.slider(
                f"Filtrer par {price_column}",
                float(y.min()),
                float(y.max()),
                (float(y.min()), float(y.max()))
            )
        
        with col2:
            # Trouver une colonne appropriée pour le filtrage
            room_column = None
            possible_room_columns = ['RM', 'rm', 'chambres', 'rooms', 'bedrooms']
            
            for col in possible_room_columns:
                if col in X.columns:
                    room_column = col
                    break
            
            if room_column:
                num_rooms_range = st.slider(
                    f"Filtrer par {room_column}",
                    float(X[room_column].min()),
                    float(X[room_column].max()),
                    (float(X[room_column].min()), float(X[room_column].max()))
                )
            else:
                # Utiliser la première colonne numérique comme fallback
                numeric_cols = X.select_dtypes(include=[float, int]).columns
                if len(numeric_cols) > 1:
                    fallback_col = numeric_cols[1]
                    num_rooms_range = st.slider(
                        f"Filtrer par {fallback_col}",
                        float(X[fallback_col].min()),
                        float(X[fallback_col].max()),
                        (float(X[fallback_col].min()), float(X[fallback_col].max()))
                    )
                else:
                    num_rooms_range = None
        
        # Appliquer les filtres
        filtered_df = df_display[
            (df_display[price_column] >= price_range[0]) & 
            (df_display[price_column] <= price_range[1])
        ]
        
        # Appliquer le filtre de la deuxième colonne si disponible
        if room_column and num_rooms_range:
            filtered_df = filtered_df[
                (filtered_df[room_column] >= num_rooms_range[0]) & 
                (filtered_df[room_column] <= num_rooms_range[1])
            ]
        
        st.write(f"📊 {len(filtered_df)} propriétés affichées sur {len(df_display)} total")
        st.dataframe(filtered_df, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Analyse des Données")
        
        # Graphiques d'analyse
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des prix
            fig = px.histogram(y, nbins=30, title="Distribution des Prix")
            fig.update_layout(xaxis_title="Prix (k$)", yaxis_title="Fréquence")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Corrélation prix vs deuxième feature numérique
            numeric_cols = X.select_dtypes(include=[float, int]).columns
            if len(numeric_cols) > 1:
                second_feature = numeric_cols[1]
                fig = px.scatter(
                    x=X[second_feature], y=y,
                    title=f"Prix vs {second_feature}",
                    labels={'x': second_feature, 'y': 'Prix (€)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pas assez de features numériques pour le graphique de corrélation")
        
        # Matrice de corrélation
        st.subheader("🔗 Matrice de Corrélation")
        corr_data = pd.concat([X.select_dtypes(include=[float, int]), y], axis=1)
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matrice de Corrélation",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("➕ Ajouter une Nouvelle Propriété")
        
        with st.form("add_property"):
            st.write("Saisissez les caractéristiques de la propriété:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                crim = st.number_input("CRIM (Taux de criminalité)", value=0.1)
                zn = st.number_input("ZN (Zone résidentielle)", value=0.0)
                indus = st.number_input("INDUS (Zone industrielle)", value=5.0)
                chas = st.selectbox("CHAS (Rivière Charles)", [0, 1])
                nox = st.number_input("NOX (Pollution)", value=0.5)
                rm = st.number_input("RM (Nombre de pièces)", value=6.0)
                age = st.number_input("AGE (Âge du bâtiment)", value=50.0)
            
            with col2:
                dis = st.number_input("DIS (Distance centres emploi)", value=3.0)
                rad = st.number_input("RAD (Accessibilité autoroutes)", value=5.0)
                tax = st.number_input("TAX (Taxe foncière)", value=300.0)
                ptratio = st.number_input("PTRATIO (Ratio élèves/prof)", value=15.0)
                b = st.number_input("B (Population)", value=350.0)
                lstat = st.number_input("LSTAT (Population défavorisée)", value=10.0)
                medv = st.number_input("MEDV (Prix en k$)", value=25.0)
            
            submitted = st.form_submit_button("➕ Ajouter la Propriété")
            
            if submitted:
                # Ici, vous pourriez ajouter la logique pour insérer en base
                st.success("✅ Propriété ajoutée avec succès!")
                st.info("💡 Fonctionnalité d'ajout à implémenter selon vos besoins")

def show_models_page(db_manager):
    """Page de gestion des modèles"""
    st.header("🤖 Gestion des Modèles")
    
    # Historique des modèles
    model_history = db_manager.get_model_history()
    
    if model_history.empty:
        st.info("Aucun modèle entraîné trouvé. Lancez d'abord une session d'entraînement.")
        
        if st.button("🚀 Lancer un Entraînement de Démonstration"):
            with st.spinner("Entraînement en cours..."):
                # Lancer le script de démonstration avec base de données
                import subprocess
                result = subprocess.run(["python", "demo_database.py"], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("✅ Entraînement terminé avec succès!")
                    st.rerun()
                else:
                    st.error(f"❌ Erreur lors de l'entraînement: {result.stderr}")
    else:
        # Afficher l'historique
        st.subheader("📊 Historique des Modèles Entraînés")
        st.dataframe(model_history, use_container_width=True)
        
        # Graphiques de performance
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                model_history,
                x='model_name',
                y='rmse',
                color='model_version',
                title="Performance des Modèles (RMSE)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                model_history,
                x='model_name',
                y='r2_score',
                color='model_version',
                title="Score R² des Modèles"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_predictions_page(db_manager):
    """Page de gestion des prédictions"""
    st.header("🔮 Gestion des Prédictions")
    
    # Historique des prédictions
    predictions_history = db_manager.get_predictions_history(limit=100)
    
    if predictions_history.empty:
        st.info("Aucune prédiction trouvée.")
    else:
        # Filtres
        col1, col2 = st.columns(2)
        
        with col1:
            model_filter = st.selectbox(
                "Filtrer par modèle",
                ["Tous"] + list(predictions_history['model_name'].unique())
            )
        
        with col2:
            limit = st.slider("Nombre de prédictions à afficher", 10, 100, 50)
        
        # Appliquer les filtres
        filtered_predictions = predictions_history.head(limit)
        if model_filter != "Tous":
            filtered_predictions = filtered_predictions[
                filtered_predictions['model_name'] == model_filter
            ]
        
        # Afficher les prédictions
        st.subheader(f"📋 Dernières Prédictions ({len(filtered_predictions)})")
        st.dataframe(filtered_predictions, use_container_width=True)
        
        # Graphiques d'analyse
        if not filtered_predictions.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Prédictions vs réalité
                valid_predictions = filtered_predictions.dropna(subset=['actual_price'])
                if not valid_predictions.empty:
                    fig = px.scatter(
                        valid_predictions,
                        x='actual_price',
                        y='predicted_price',
                        color='model_name',
                        title="Prédictions vs Prix Réels"
                    )
                    fig.add_shape(
                        type="line",
                        x0=valid_predictions['actual_price'].min(),
                        y0=valid_predictions['actual_price'].min(),
                        x1=valid_predictions['actual_price'].max(),
                        y1=valid_predictions['actual_price'].max(),
                        line=dict(color="red", dash="dash")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution des prédictions
                fig = px.histogram(
                    filtered_predictions,
                    x='predicted_price',
                    color='model_name',
                    title="Distribution des Prix Prédits"
                )
                st.plotly_chart(fig, use_container_width=True)

def show_admin_page(db_manager):
    """Page d'administration"""
    st.header("⚙️ Administration de la Base de Données")
    
    # Informations sur la base
    st.subheader("ℹ️ Informations sur la Base de Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📍 **Emplacement**: {db_manager.db_path}")
        
        # Taille du fichier
        if os.path.exists(db_manager.db_path):
            size_mb = os.path.getsize(db_manager.db_path) / (1024 * 1024)
            st.info(f"💾 **Taille**: {size_mb:.2f} MB")
    
    with col2:
        stats = db_manager.get_database_stats()
        st.info(f"📊 **Statistiques**:")
        st.write(f"- Propriétés: {stats['properties']}")
        st.write(f"- Prédictions: {stats['predictions']}")
        st.write(f"- Modèles: {stats['trained_models']}")
    
    # Actions d'administration
    st.subheader("🔧 Actions d'Administration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Réinitialiser les Données"):
            if st.checkbox("Confirmer la réinitialisation"):
                try:
                    # Supprimer le fichier de base de données
                    if os.path.exists(db_manager.db_path):
                        os.remove(db_manager.db_path)
                    
                    # Recréer la base
                    new_db = DatabaseManager()
                    new_db.insert_sample_data()
                    
                    st.success("✅ Base de données réinitialisée")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    with col2:
        if st.button("📤 Exporter les Données"):
            try:
                data_loader = DataLoader(use_database=True)
                X, y = data_loader.load_boston_housing()
                
                df_export = pd.concat([X, y], axis=1)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="💾 Télécharger CSV",
                    data=csv,
                    file_name="house_prices_export.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Erreur d'export: {e}")
    
    with col3:
        if st.button("🔍 Analyser la Base"):
            st.info("💡 Analyse de la base de données en cours...")
            
            # Analyser les tables
            conn = db_manager.connect()
            cursor = conn.cursor()
            
            try:
                # Lister les tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                st.write("📋 **Tables disponibles**:")
                for table in tables:
                    st.write(f"- {table[0]}")
                
                # Analyser chaque table
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    st.write(f"  📊 {table_name}: {count} enregistrements")
                
            finally:
                db_manager.disconnect()

if __name__ == "__main__":
    main()