import os
import sys
import subprocess
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# Permet d'importer preprocessing.py depuis le dossier src/
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from preprocessing import preprocess_data, ORDINAL_MAPPINGS, ONE_HOT_COLS, LEAKY_CHURN_FEATURES, LEAKY_CLUSTERING_FEATURES

# --- Chemins relatifs au projet ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'train_test')

app = Flask(__name__)


def load_artifacts():
    global churn_pipeline, clustering_artifacts, kmeans_model, regression_pipeline
    global cluster_label_mapping, reverse_cluster_label_mapping, processed_columns, dashboard_metrics

    churn_pipeline = joblib.load(os.path.join(MODELS_DIR, 'churn_pipeline.pkl'))
    clustering_artifacts = joblib.load(os.path.join(MODELS_DIR, 'clustering_preprocessor_pipeline.pkl'))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    regression_pipeline = joblib.load(os.path.join(MODELS_DIR, 'regression_pipeline.pkl'))
    cluster_label_mapping = joblib.load(os.path.join(MODELS_DIR, 'cluster_label_mapping.pkl'))
    try:
        reverse_cluster_label_mapping = joblib.load(os.path.join(MODELS_DIR, 'reverse_cluster_label_mapping.pkl'))
    except FileNotFoundError:
        reverse_cluster_label_mapping = {label: cid for cid, label in cluster_label_mapping.items()}
    processed_columns = joblib.load(os.path.join(MODELS_DIR, 'processed_columns.pkl'))
    try:
        dashboard_metrics = joblib.load(os.path.join(MODELS_DIR, 'dashboard_metrics.pkl'))
    except FileNotFoundError:
        dashboard_metrics = {}
    print("[✓] Tous les modèles et pipelines chargés avec succès.")


try:
    load_artifacts()
except FileNotFoundError as e:
    print(f"[!] Erreur de chargement des modèles: {e}. Assurez-vous d'avoir exécuté train_model.py.")
    exit()

# --- Mappings pour l'affichage des segments (basé sur le ré-étiquetage)
# Ces mappings sont maintenant dynamiques et basés sur cluster_label_mapping
# Les valeurs sont des exemples et devraient être ajustées après analyse des nouveaux clusters
DISPLAY_CLUSTER_INFO = {
    'VIP':          ('Clients VIP / Fidèles',   'success',  'fa-star',          'Très faible'),
    'Fidele':       ('Clients Fidèles',         'primary',  'fa-heart',         'Faible'),
    'Occasionnel':  ('Clients Occasionnels',    'warning',  'fa-clock',         'Modéré'),
    'A Risque':     ('Clients à Risque',        'danger',   'fa-exclamation-triangle', 'Élevé'),
}

# --- HELPERS ---

def get_processed_df_from_form(form_data: dict) -> pd.DataFrame:
    """
    Construit un DataFrame à partir des données du formulaire et le pré-traite
    en utilisant la fonction preprocess_data de src/preprocessing.py.
    """
    # Nettoyer le formulaire : chaînes vides -> NaN
    clean_form = {
        key: (np.nan if value is None or str(value).strip() == '' else value)
        for key, value in form_data.items()
    }

    # Créer un DataFrame avec une seule ligne à partir des données du formulaire
    # (gardé en commentaire ici pour transparence, mais le DataFrame final est créé
    # via le DataFrame des colonnes originales ci-dessous)

    # Assurer que toutes les colonnes attendues sont présentes, même si vides
    # et dans le bon ordre pour preprocess_data
    # Pour cela, il faut récupérer la liste des colonnes brutes attendues
    # C'est une simplification, idéalement il faudrait un mapping complet
    # des champs du formulaire vers les colonnes brutes.
    # Pour l'exemple, on va prendre les colonnes du df processed et enlever celles créées
    # par feature engineering pour simuler les colonnes brutes.
    # C'est une partie délicate qui dépend fortement de la structure de votre formulaire
    # et de la fonction preprocess_data.

    # Pour simplifier, on va créer un DataFrame avec les colonnes de `processed_columns`
    # et remplir avec les données du formulaire, le reste sera NaN.
    # C'est une approche qui nécessite que le formulaire fournisse des noms de colonnes
    # qui correspondent aux colonnes *après* le prétraitement initial mais *avant* l'imputation/scaling.
    # C'est pourquoi l'utilisation de pipelines est cruciale.

    # Ici, nous allons simuler la création d'un DataFrame avec les colonnes originales
    # que preprocess_data attend, puis le passer à preprocess_data.
    # Il est IMPÉRATIF que les noms des champs du formulaire correspondent aux noms des colonnes
    # attendues par preprocess_data AVANT l'encodage.

    # Pour cet exemple, je vais créer un DataFrame avec les colonnes du fichier CSV original
    # et remplir avec les données du formulaire. Les colonnes non présentes dans le formulaire
    # seront NaN et gérées par le pipeline.
    # Ceci est une simplification. Dans un vrai projet, il faudrait un mapping précis.

    # Charger un échantillon du DataFrame original pour obtenir les colonnes brutes
    # C'est une solution temporaire pour avoir la liste des colonnes originales
    # Idéalement, cette liste devrait être définie explicitement ou chargée.
    original_df_sample = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'retail_customers_COMPLETE_CATEGORICAL.csv'), nrows=1)
    original_cols = original_df_sample.columns.tolist()

    # Créer un DataFrame vide avec les colonnes originales
    df_raw_input = pd.DataFrame(columns=original_cols)
    # Ajouter la ligne d'entrée du formulaire
    df_raw_input = pd.concat([df_raw_input, pd.DataFrame([clean_form])], ignore_index=True)
    df_raw_input = df_raw_input.replace(r'^\s*$', np.nan, regex=True)

    # Convertir les types de données si nécessaire (ex: numérique)
    for col in original_cols:
        if col in df_raw_input.columns and col not in ['RegistrationDate', 'LastLoginIP', 'Country']:
            try:
                df_raw_input[col] = pd.to_numeric(df_raw_input[col], errors='coerce')
            except:
                pass # Laisser comme string si non numérique

    # Appliquer le prétraitement initial (sans imputation/scaling qui sont dans les pipelines)
    # target_col est None car on ne fait pas de target encoding ici pour l'input unique
    df_processed_input = preprocess_data(df_raw_input, target_col=None)

    # S'assurer que toutes les colonnes du processed_columns sont présentes
    # Utiliser reindex() pour garder NaN pour les colonnes manquantes
    # Cela permet aux pipelines (KNNImputer) de gérer correctement les valeurs manquantes
    # plutôt que de les forcer à 0
    final_input_df = df_processed_input.reindex(columns=processed_columns)

    return final_input_df

def predict_churn(df_input: pd.DataFrame):
    """Prédit le Churn et retourne label + probabilité en utilisant le pipeline."""
    # Exclure les features fuyantes du DataFrame d'entrée avant de le passer au pipeline
    # Le pipeline gérera l'imputation et le scaling
    cols_to_drop = [col for col in LEAKY_CHURN_FEATURES if col in df_input.columns] + ['Churn', 'CustomerID']
    X_input_for_churn = df_input.drop(columns=cols_to_drop, errors='ignore')

    prob  = churn_pipeline.predict_proba(X_input_for_churn)[0][1]
    label = int(churn_pipeline.predict(X_input_for_churn)[0])
    return label, round(float(prob), 4)

def predict_cluster(df_input: pd.DataFrame):
    """
    Prédit le cluster KMeans avec la même chaîne de prétraitement que l'entraînement :
      imputation KNN -> winsorisation p1-p99 -> log-transform -> RobustScaler.
    """
    art          = clustering_artifacts
    feature_names = art['feature_names']
    imputer      = art['imputer']
    scaler       = art['scaler']
    clip_bounds  = art['clip_bounds']
    log_cols     = art['log_cols']

    # Sélectionner uniquement les 9 features RFM du clustering
    existing = [c for c in feature_names if c in df_input.columns]
    X = df_input[existing].copy().reindex(columns=feature_names)

    # Imputation
    X_imp = pd.DataFrame(
        imputer.transform(X), columns=feature_names)

    # Winsorisation (mêmes bornes qu'à l'entraînement)
    for col in feature_names:
        if col in clip_bounds:
            lo, hi = clip_bounds[col]
            X_imp[col] = X_imp[col].clip(lo, hi)

    # Log-transform
    for col in log_cols:
        if col in X_imp.columns:
            X_imp[col] = np.log1p(X_imp[col].clip(lower=0))

    # Scaling
    X_sc = scaler.transform(X_imp)
    X_pca = art['pca'].transform(X_sc)

    cluster_id    = int(kmeans_model.predict(X_pca)[0])
    cluster_label = cluster_label_mapping.get(cluster_id, 'Inconnu')
    return cluster_id, cluster_label

def predict_monetary(df_input: pd.DataFrame):
    """Prédit le MonetaryTotal en utilisant le pipeline de régression."""
    # Le modèle de régression a été entraîné sur toutes les colonnes sauf MonetaryTotal,
    # CustomerID et les features dérivées de MonetaryTotal.
    X_input_for_regression = df_input.drop(columns=[
        'MonetaryTotal', 'CustomerID',
        'MonetaryAvg', 'MonetaryStd', 'MonetaryMin', 'MonetaryMax',
        'MonetaryPerDay', 'AvgBasketValue'
    ], errors='ignore')

    monetary_prediction = regression_pipeline.predict(X_input_for_regression)[0]
    return round(float(monetary_prediction), 2)

# --- ROUTES ---

@app.route('/')
def index():
    """Page d'accueil avec métriques du projet."""
    if dashboard_metrics and isinstance(dashboard_metrics, dict):
        stats = dashboard_metrics.get('stats', {
            'n_clients':  4372,
            'n_features': len(processed_columns),
            'churn_rate': '33.3%',
            'rf_auc':     '0.96',
            'rf_f1':      '0.85',
            'reg_r2':     '0.67',
            'n_clusters': len(cluster_label_mapping),
            'n_models':  3,
        })
    else:
        stats = {
            'n_clients':  4372,
            'n_features': len(processed_columns),
            'churn_rate': '33.3%',
            'rf_auc':     '0.96',
            'rf_f1':      '0.85',
            'reg_r2':     '0.67',
            'n_clusters': len(cluster_label_mapping),
            'n_models':  3,
        }

    dashboard = dashboard_metrics if dashboard_metrics else {}
    return render_template('index.html', stats=stats, dashboard=dashboard)


@app.route('/run-models', methods=['POST'])
def run_models():
    """Lance le script d'entraînement depuis le dashboard."""
    run_message = None
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, 'src', 'train_model.py')],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            run_message = f"Erreur d'entraînement : {stderr.splitlines()[-1]}"
        else:
            run_message = "Modèles entraînés avec succès."
        load_artifacts()
    except Exception as exc:
        run_message = f"Erreur interne : {exc}"

    if dashboard_metrics and isinstance(dashboard_metrics, dict):
        stats = dashboard_metrics.get('stats', {
            'n_clients':  4372,
            'n_features': len(processed_columns),
            'churn_rate': '33.3%',
            'rf_auc':     '0.96',
            'rf_f1':      '0.85',
            'reg_r2':     '0.67',
            'n_clusters': len(cluster_label_mapping),
            'n_models':  3,
        })
    else:
        stats = {
            'n_clients':  4372,
            'n_features': len(processed_columns),
            'churn_rate': '33.3%',
            'rf_auc':     '0.96',
            'rf_f1':      '0.85',
            'reg_r2':     '0.67',
            'n_clusters': len(cluster_label_mapping),
            'n_models':  3,
        }
    dashboard = dashboard_metrics if dashboard_metrics else {}
    return render_template('index.html', stats=stats, dashboard=dashboard, run_message=run_message)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Formulaire + résultat de prédiction."""
    result = None

    if request.method == 'POST':
        form = request.form.to_dict()

        # Construire le DataFrame d'entrée traité
        df_input_processed = get_processed_df_from_form(form)

        # Prédictions
        churn_label, churn_prob = predict_churn(df_input_processed)
        cluster_id, cluster_label = predict_cluster(df_input_processed)
        monetary_pred = predict_monetary(df_input_processed)

        display_info = DISPLAY_CLUSTER_INFO.get(cluster_label, ('Inconnu', 'secondary', 'fa-question', '-'))
        cluster_name, cluster_color, cluster_icon, cluster_churn_info = display_info

        # Niveau de risque basé sur la probabilité de churn
        if churn_prob >= 0.75:
            risk_level, risk_color = 'Critique', 'danger'
        elif churn_prob >= 0.5:
            risk_level, risk_color = 'Élevé', 'warning'
        elif churn_prob >= 0.25:
            risk_level, risk_color = 'Modéré', 'info'
        else:
            risk_level, risk_color = 'Faible', 'success'

        result = {
            'churn_label':    churn_label,
            'churn_prob':     round(churn_prob * 100, 1),
            'churn_text':     'Client Parti (Churned)' if churn_label == 1 else 'Client Fidèle',
            'churn_color':    'danger' if churn_label == 1 else 'success',
            'risk_level':     risk_level,
            'risk_color':     risk_color,
            'cluster_id':     cluster_id,
            'cluster_name':   cluster_name,
            'cluster_color':  cluster_color,
            'cluster_icon':   cluster_icon,
            'cluster_churn':  cluster_churn_info,
            'monetary_pred':  monetary_pred,
            'form_data':      dict(form),
        }

    # Pour le formulaire, nous avons besoin des catégories pour les listes déroulantes
    # Ces listes devraient être générées dynamiquement ou chargées depuis un fichier de config
    # Pour l'instant, je vais les hardcoder comme dans votre version originale pour l'exemple
    # mais il faudrait les rendre dynamiques si elles changent.
    form_options = {
        'AgeCategory': list(ORDINAL_MAPPINGS['AgeCategory'].keys()),
        'SpendingCategory': list(ORDINAL_MAPPINGS['SpendingCategory'].keys()),
        'LoyaltyLevel': list(ORDINAL_MAPPINGS['LoyaltyLevel'].keys()),
        'ChurnRiskCategory': list(ORDINAL_MAPPINGS['ChurnRiskCategory'].keys()),
        'BasketSizeCategory': list(ORDINAL_MAPPINGS['BasketSizeCategory'].keys()),
        'PreferredTimeOfDay': list(ORDINAL_MAPPINGS['PreferredTimeOfDay'].keys()),
        'FavoriteSeason': ['Automne', 'Hiver', 'Printemps', 'Été'],
        'Region': ['Afrique','Amérique_du_Nord','Amérique_du_Sud','Asie','Autre','Europe_centrale','Europe_continentale','Europe_de_lEst','Europe_du_Nord','Europe_du_Sud','Moyen-Orient','Océanie','UK'],
        'WeekendPreference': ['Inconnu', 'Semaine', 'Weekend'],
        'ProductDiversity': ['Explorateur', 'Modéré', 'Spécialisé'],
        'Gender': ['F', 'M', 'Unknown'],
        'AccountStatus': ['Active', 'Closed', 'Pending', 'Suspended'],
        'Country': ['France', 'Germany', 'UK', 'USA', 'Canada', 'Australia', 'Other'] # Exemple, à adapter
    }

    return render_template('predict.html', result=result, form_options=form_options)


@app.route('/segments')
def segments():
    """Page de visualisation des segments clients."""
    # Compute cluster profiles dynamically
    import pandas as pd
    import numpy as np
    
    df_processed = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'retail_customers_processed.csv'))
    # Le clustering a été entraîné uniquement sur les clients avec MonetaryTotal > 0
    df_processed = df_processed[df_processed['MonetaryTotal'] > 0].copy()
    
    # Load clustering artifacts
    art = clustering_artifacts
    feature_names = art['feature_names']
    imputer = art['imputer']
    scaler = art['scaler']
    clip_bounds = art['clip_bounds']
    log_cols = art['log_cols']
    
    # Select features
    existing = [c for c in feature_names if c in df_processed.columns]
    X = df_processed[existing].copy().reindex(columns=feature_names)
    
    # Preprocess
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_names)
    for col in feature_names:
        if col in clip_bounds:
            lo, hi = clip_bounds[col]
            X_imp[col] = X_imp[col].clip(lo, hi)
    for col in log_cols:
        if col in X_imp.columns:
            X_imp[col] = np.log1p(X_imp[col].clip(lower=0))
    X_sc = scaler.transform(X_imp)
    
    # Predict clusters
    X_pca = art['pca'].transform(X_sc)
    clusters = kmeans_model.predict(X_pca)
    df_processed['Cluster'] = clusters

    # Compute summary
    summary = df_processed.groupby('Cluster').agg({
        'Frequency': 'mean',
        'MonetaryTotal': 'mean',
        'CustomerTenureDays': 'mean',
        'SatisfactionScore': 'mean',
        'Churn': 'mean'
    }).round(2)
    
    sizes = df_processed.groupby('Cluster').size()
    
    segments_data = []
    for cluster_id, label in cluster_label_mapping.items():
        display_info = DISPLAY_CLUSTER_INFO.get(label, ('Inconnu', 'secondary', 'fa-question', '-'))
        name, color, icon, churn_info = display_info
        
        size = sizes.get(cluster_id, 0)
        freq = summary.loc[cluster_id, 'Frequency'] if cluster_id in summary.index else 0
        mon = summary.loc[cluster_id, 'MonetaryTotal'] if cluster_id in summary.index else 0
        ten = summary.loc[cluster_id, 'CustomerTenureDays'] if cluster_id in summary.index else 0
        sat = summary.loc[cluster_id, 'SatisfactionScore'] if cluster_id in summary.index else 0
        churn_pct = (summary.loc[cluster_id, 'Churn'] * 100) if cluster_id in summary.index else 0
        
        # Define actions based on segment
        if label == 'VIP':
            action = 'Fidélisation et programmes de récompenses premium.'
        elif label == 'Fidele':
            action = 'Campagnes de rétention et cross-selling.'
        elif label == 'Occasionnel':
            action = 'Stimuler la fréquence d\'achat avec promotions.'
        elif label == 'A Risque':
            action = 'Actions de réactivation et analyse des causes de churn.'
        else:
            action = 'Actions marketing spécifiques à définir.'
        
        segments_data.append({
            'id':          cluster_id,
            'name':        name,
            'color':       color,
            'icon':        icon,
            'size':        f'{size} clients ({size/len(df_processed)*100:.1f}%)',
            'churn':       f'{churn_pct:.1f}%',
            'frequency':   f'{freq:.1f} achats',
            'monetary':    f'{mon:.0f} €',
            'tenure':      f'{ten:.0f} jours',
            'satisfaction':f'{sat:.2f}/5',
            'action':      action,
            'badge':       f'Priorité : {label}',
        })
    stats = dashboard_metrics.get('stats', {}) if dashboard_metrics else {}
    return render_template('segments.html', segments=segments_data, dashboard=dashboard_metrics, stats=stats)


@app.route('/about')
def about():
    """Page de description du projet."""
    pipeline = [
        ('1', 'Exploration',     'Analyse qualité, distributions, corrélations',        'fa-search',    'primary'),
        ('2', 'Préparation',     'Nettoyage, encodage, feature engineering',             'fa-tools',     'info'),
        ('3', 'Transformation',  'Réduction de dimension par ACP (20 composantes)',     'fa-compress',  'warning'), # Mise à jour
        ('4', 'Modélisation',    'KMeans, Random Forest, Régression Linéaire',           'fa-brain',     'success'),
        ('5', 'Évaluation',      'Métriques, courbes ROC, recommandations métier',       'fa-chart-bar', 'danger'),
        ('6', 'Déploiement',     'Interface Flask avec prédiction en temps réel',        'fa-rocket',    'dark'),
    ]
    models_info = [
        ('KMeans (k=4)',         'Clustering',           'Silhouette = 0.21 / Davies-Bouldin = 1.30',  'Segmentation clients'), # Mise à jour
        ('Random Forest',        'Classification Churn', 'AUC = 0.96 / F1 = 0.85', 'Prédiction churn'), # Mise à jour
        ('Régression Linéaire',  'Régression',           'R² = 0.67 / MAE = À calculer', 'Prédiction CA'), # Mise à jour
    ]
    return render_template('about.html', pipeline=pipeline, models_info=models_info)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API JSON pour la prédiction.
    Entrée  : JSON avec les features du client
    Sortie  : { churn, probability, cluster, cluster_name, monetary_prediction }
    """
    data = request.get_json(force=True)
    df_input_processed = get_processed_df_from_form(data)
    churn_label, churn_prob = predict_churn(df_input_processed)
    cluster_id, cluster_label = predict_cluster(df_input_processed)
    monetary_pred = predict_monetary(df_input_processed)

    return jsonify({
        'churn':        churn_label,
        'probability':  churn_prob,
        'risk':         'Élevé' if churn_prob >= 0.5 else 'Faible',
        'cluster':      cluster_id,
        'cluster_name': cluster_label,
        'monetary_prediction': monetary_pred,
    })


# -- MAIN ---
if __name__ == '__main__':
    print("=" * 50)
    print("  Démarrage de l'application Flask")
    print("  URL : http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)