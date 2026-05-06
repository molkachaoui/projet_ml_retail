"""
predict.py — Inférence sur de nouvelles données
Projet : Analyse Comportementale Clientèle Retail

MISE À JOUR: Ce script utilise maintenant les mêmes pipelines et artefacts
que train_model.py et app.py pour assurer la cohérence:
- churn_pipeline.pkl pour la prédiction de churn
- regression_pipeline.pkl pour la prédiction monétaire
- clustering_preprocessor_pipeline.pkl + kmeans_model.pkl pour le clustering
- preprocess_data() pour le prétraitement des données brutes
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Importer les fonctions de prétraitement et constantes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data, LEAKY_CHURN_FEATURES, LEAKY_CLUSTERING_FEATURES

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, 'models')

# Charger les artefacts de modèles (comme dans app.py)
churn_pipeline = joblib.load(os.path.join(MODELS_DIR, 'churn_pipeline.pkl'))
clustering_artifacts = joblib.load(os.path.join(MODELS_DIR, 'clustering_preprocessor_pipeline.pkl'))
kmeans_model = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
regression_pipeline = joblib.load(os.path.join(MODELS_DIR, 'regression_pipeline.pkl'))
cluster_label_mapping = joblib.load(os.path.join(MODELS_DIR, 'cluster_label_mapping.pkl'))
processed_columns = joblib.load(os.path.join(MODELS_DIR, 'processed_columns.pkl'))


def predict_churn(input_data: dict) -> dict:
    """
    Prédit si un client va churner.
    Args:
        input_data : dict des features brutes du client (sera prétraité)
    Returns:
        {'churn': 0/1, 'probability': float, 'label': str}
    """
    # Créer un DataFrame avec les données brutes
    df_raw = pd.DataFrame([input_data])

    # Appliquer le prétraitement initial (sans imputation/scaling qui sont dans le pipeline)
    df_processed = preprocess_data(df_raw, target_col=None, drop_constant=False)

    # S'assurer que toutes les colonnes du processed_columns sont présentes
    df_final = pd.DataFrame(columns=processed_columns)
    df_final = pd.concat([df_final, df_processed], ignore_index=True)
    df_final = df_final[processed_columns].fillna(0)

    # Exclure les features fuyantes avant de passer au pipeline
    cols_to_drop = [col for col in LEAKY_CHURN_FEATURES if col in df_final.columns] + ['Churn', 'CustomerID']
    X_input = df_final.drop(columns=cols_to_drop, errors='ignore')

    # Prédiction avec le pipeline
    prediction = int(churn_pipeline.predict(X_input)[0])
    probability = float(churn_pipeline.predict_proba(X_input)[0][1])

    return {
        'churn':       prediction,
        'probability': round(probability, 4),
        'label':       'Parti' if prediction == 1 else 'Fidèle',
    }


def predict_monetary(input_data: dict) -> dict:
    """
    Prédit le montant total dépensé (MonetaryTotal) par un client.
    Args:
        input_data : dict des features brutes du client (sera prétraité)
    Returns:
        {'predicted_monetary': float}
    """
    # Créer un DataFrame avec les données brutes
    df_raw = pd.DataFrame([input_data])

    # Appliquer le prétraitement initial
    df_processed = preprocess_data(df_raw, target_col=None, drop_constant=False)

    # S'assurer que toutes les colonnes du processed_columns sont présentes
    df_final = pd.DataFrame(columns=processed_columns)
    df_final = pd.concat([df_final, df_processed], ignore_index=True)
    df_final = df_final[processed_columns].fillna(0)

    # Exclure les colonnes non utilisées par le modèle de régression
    X_input = df_final.drop(columns=['MonetaryTotal', 'CustomerID'], errors='ignore')

    # Prédiction avec le pipeline
    prediction = float(regression_pipeline.predict(X_input)[0])

    return {'predicted_monetary': round(prediction, 2)}


def predict_cluster(input_data: dict) -> dict:
    """
    Assigne un nouveau client à un cluster KMeans.
    Args:
        input_data : dict des features brutes du client (sera prétraité)
    Returns:
        {'cluster': int, 'cluster_label': str}
    """
    # Créer un DataFrame avec les données brutes
    df_raw = pd.DataFrame([input_data])

    # Appliquer le prétraitement initial
    df_processed = preprocess_data(df_raw, target_col=None, drop_constant=False)

    # S'assurer que toutes les colonnes du processed_columns sont présentes
    df_final = pd.DataFrame(columns=processed_columns)
    df_final = pd.concat([df_final, df_processed], ignore_index=True)
    df_final = df_final[processed_columns].fillna(0)

    # Utiliser la même logique de prétraitement que dans app.py
    art = clustering_artifacts
    feature_names = art['feature_names']
    imputer = art['imputer']
    scaler = art['scaler']
    clip_bounds = art['clip_bounds']
    log_cols = art['log_cols']

    # Sélectionner uniquement les features RFM du clustering
    existing = [c for c in feature_names if c in df_final.columns]
    X = df_final[existing].copy().reindex(columns=feature_names)

    # Imputation
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_names)

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

    # Prédiction du cluster
    cluster_id = int(kmeans_model.predict(X_sc)[0])
    cluster_label = cluster_label_mapping.get(cluster_id, 'Inconnu')

    return {'cluster': cluster_id, 'cluster_label': cluster_label}


if __name__ == '__main__':
    # Exemple d'utilisation avec des données brutes similaires au formulaire
    # Inclut les colonnes requises pour le prétraitement
    sample = {
        'CustomerID': 99999,  # ID fictif pour l'exemple
        'Frequency': 5,
        'MonetaryTotal': 1200.0,
        'MonetaryAvg': 240.0,
        'MonetaryMax': 800.0,
        'TotalTransactions': 30,
        'AvgDaysBetweenPurchases': 45.0,
        'UniqueProducts': 20,
        'ReturnRatio': 0.05,
        'CustomerTenureDays': 365,
        'FirstPurchaseDaysAgo': 400,
        'PreferredMonth': 6,
        'PreferredHour': 18,
        'WeekendPurchaseRatio': 0.4,
        'FavoriteSeason': 'Été',
        'WeekendPreference': 'Weekend',
        'Age': 35,
        'RegistrationDate': '2023-01-15',  # Date fictive
        'LastLoginIP': '192.168.1.1',  # IP fictive
        'SupportTicketsCount': 1,
        'SatisfactionScore': 4.5,
        'Gender': 'M',
        'AccountStatus': 'Active',
        'ProductDiversity': 'Modéré',
        'Region': 'Europe_centrale',
        'Country': 'France',  # Sera encodé
        'Churn': 0  # Non utilisé pour la prédiction
    }

    print("=== Prédictions pour un client exemple ===")
    print("Churn prediction:", predict_churn(sample))
    print("Monetary prediction:", predict_monetary(sample))
    print("Cluster prediction:", predict_cluster(sample))
