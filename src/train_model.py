"""
train_model.py — Pipeline d'entrainement complet
================================================
Corrections appliquees:
  1. CHURN     : fuite de donnees exclue et evaluation plus robuste.
  2. CLUSTERING: features RFM ciblees, log-transform, PCA, et labels metier.
  3. REGRESSION: pipelines multiples, selection du meilleur modele.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    preprocess_data,
    LEAKY_CHURN_FEATURES,
    LEAKY_CLUSTERING_FEATURES,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(
    BASE,
    'data',
    'raw',
    'retail_customers_COMPLETE_CATEGORICAL.csv',
)
PROCESSED_PATH = os.path.join(
    BASE,
    'data',
    'processed',
    'retail_customers_processed.csv',
)
MODELS_DIR = os.path.join(BASE, 'models')
DASHBOARD_METRICS = os.path.join(MODELS_DIR, 'dashboard_metrics.pkl')
PROCESSED_COLUMNS = os.path.join(MODELS_DIR, 'processed_columns.pkl')

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Features metier pour le clustering  (subset RFM + enrichissements)
# ---------------------------------------------------------------------------
CLUSTERING_FEATURES = [
    'Recency',
    'Frequency',
    'MonetaryTotal',
    'SatisfactionScore',
    'ReturnRatio',
    'UniqueProducts',
    'AvgDaysBetweenPurchases',
    'Age',
    'SupportTicketsCount',
    'AvgBasketValue',
    'CancelRate',
]

LOG_COLS_CLUSTERING = [
    'Recency',
    'Frequency',
    'MonetaryTotal',
    'UniqueProducts',
    'AvgDaysBetweenPurchases',
    'AvgBasketValue',
]


def _safe_drop(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors='ignore')


def _log_transform(df: pd.DataFrame, cols: list):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def _compute_elbow(X: np.ndarray, max_k: int = 7):
    ks = []
    inertias = []
    silhouettes = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        ks.append(int(k))
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X, labels)))
    return ks, inertias, silhouettes


# ---------------------------------------------------------------------------
# Preprocessing specifique au clustering
# ---------------------------------------------------------------------------
def _preprocess_for_clustering(df_input: pd.DataFrame):
    """
    Retourne (X_pca, imputer, scaler, pca, clip_bounds, feature_names, index).
    """
    df_c = df_input[df_input['MonetaryTotal'] > 0].copy()
    print(f"[i] Clients apres filtrage MonetaryTotal>0 : {len(df_c)} "
          f"(retires: {len(df_input) - len(df_c)})")

    feature_names = [c for c in CLUSTERING_FEATURES if c in df_c.columns]
    X = df_c[feature_names].copy()

    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_names, index=df_c.index)

    clip_bounds = {}
    for col in feature_names:
        p01 = X_imp[col].quantile(0.01)
        p99 = X_imp[col].quantile(0.99)
        clip_bounds[col] = (p01, p99)
        X_imp[col] = X_imp[col].clip(p01, p99)

    X_imp = _log_transform(X_imp, [c for c in LOG_COLS_CLUSTERING if c in X_imp.columns])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, imputer, scaler, pca, clip_bounds, feature_names, df_c.index


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION CHURN
# ---------------------------------------------------------------------------

def _classification_metrics(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline.named_steps.get('classifier'), 'predict_proba'):
        y_proba = pipeline.named_steps['classifier'].predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }
    return metrics


def train_churn_model(df):
    print("\n" + "=" * 62)
    print("  CLASSIFICATION CHURN  (Random Forest + XGBoost)")
    print("=" * 62)

    cols_to_drop = LEAKY_CHURN_FEATURES + ['Churn', 'CustomerID']
    X = _safe_drop(df, cols_to_drop)
    y = df['Churn']

    print(f"[i] Features : {X.shape[1]} colonnes | {len(y)} exemples")
    print(f"[i] Churn=1  : {y.sum()}   Churn=0 : {(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    churn_pipeline = ImbPipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('selector', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='median',
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
        )),
    ])

    print("[->] Entrainement RandomForest...")
    churn_pipeline.fit(X_train, y_train)
    rf_metrics = _classification_metrics(churn_pipeline, X_test, y_test)
    print(f"[OK] RandomForest ROC-AUC = {rf_metrics['roc_auc']:.3f}")

    churn_scores = {'RandomForest': rf_metrics}

    if HAS_XGBOOST:
        pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
        xgb_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )),
        ])
        print("[->] Entrainement XGBoost...")
        xgb_pipeline.fit(X_train, y_train)
        xgb_metrics = _classification_metrics(xgb_pipeline, X_test, y_test)
        churn_scores['XGBoost'] = xgb_metrics
        print(f"[OK] XGBoost ROC-AUC = {xgb_metrics['roc_auc']:.3f}")

    rf = churn_pipeline.named_steps['classifier']
    importances = sorted(
        zip(X_train.columns, rf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n--- Top 15 features ---")
    for name, imp in importances[:15]:
        print(f"  {name:<44s} {imp:.4f}")

    joblib.dump(churn_pipeline, os.path.join(MODELS_DIR, 'churn_pipeline.pkl'))
    print(f"\n[OK] Sauvegarde -> churn_pipeline.pkl")
    return churn_scores


# ---------------------------------------------------------------------------
# 2. CLUSTERING KMEANS
# ---------------------------------------------------------------------------
def train_clustering_model(df):
    print("\n" + "=" * 62)
    print("  CLUSTERING  (KMeans)")
    print("=" * 62)

    X_pca, imputer, scaler, pca, clip_bounds, feature_names, valid_idx = _preprocess_for_clustering(df)

    K = 4
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_pca)
    sil = silhouette_score(X_pca, clusters)
    print(f"[OK] KMeans k={K} | Silhouette = {sil:.3f}")

    df_c = df.loc[valid_idx].copy()
    df_c['Cluster'] = clusters

    profile_cols = [c for c in ['MonetaryTotal', 'Recency', 'Frequency', 'Churn', 'SatisfactionScore'] if c in df_c.columns]
    summary = df_c.groupby('Cluster')[profile_cols].mean()
    summary['n_clients'] = df_c.groupby('Cluster').size()
    print("\n--- Profil brut des clusters ---")
    print(summary.round(2))

    sorted_ids = summary['MonetaryTotal'].sort_values(ascending=False).index.tolist()
    base_labels = ['VIP', 'Fidele', 'Occasionnel', 'A Risque']
    cluster_mapping = {cid: base_labels[i] for i, cid in enumerate(sorted_ids)}
    df_c['Label'] = df_c['Cluster'].map(cluster_mapping)

    labeled_summary = df_c.groupby('Label')[profile_cols].mean()
    labeled_summary['n_clients'] = df_c.groupby('Label').size()
    print("\n--- Profil par label ---")
    print(labeled_summary.round(2))

    if 'Churn' in df_c.columns:
        for label in base_labels:
            grp = df_c[df_c['Label'] == label]
            if len(grp):
                churn_rate = grp['Churn'].mean()
                print(f"  {label:<12s}: churn={churn_rate:.1%}  n={len(grp)}")

    ks, inertias, silhouettes = _compute_elbow(X_pca, max_k=7)
    cluster_sizes = []
    palette = ['#5B8FF9', '#5AD8A6', '#5D7092', '#F6BD16']
    for cid, count in df_c['Cluster'].value_counts().sort_index().items():
        cluster_sizes.append({
            'id': int(cid),
            'count': int(count),
            'pct': float(100 * count / len(df_c)),
            'label': cluster_mapping.get(int(cid), f'Cluster {cid}'),
            'color': palette[int(cid) % len(palette)],
        })

    clustering_metrics = {
        'k': K,
        'silhouette': float(sil),
        'inertia': float(kmeans.inertia_),
        'elbow': {
            'ks': ks,
            'inertia': inertias,
            'silhouette': silhouettes,
        },
        'cluster_sizes': cluster_sizes,
    }

    clustering_artifacts = {
        'imputer': imputer,
        'scaler': scaler,
        'pca': pca,
        'clip_bounds': clip_bounds,
        'feature_names': feature_names,
        'log_cols': LOG_COLS_CLUSTERING,
    }
    reverse_cluster_mapping = {label: cid for cid, label in cluster_mapping.items()}

    joblib.dump(clustering_artifacts, os.path.join(MODELS_DIR, 'clustering_preprocessor_pipeline.pkl'))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    joblib.dump(cluster_mapping, os.path.join(MODELS_DIR, 'cluster_label_mapping.pkl'))
    joblib.dump(reverse_cluster_mapping, os.path.join(MODELS_DIR, 'reverse_cluster_label_mapping.pkl'))
    print(f"\n[OK] Sauvegardes clustering -> {MODELS_DIR}")
    return clustering_metrics


# ---------------------------------------------------------------------------
# 3. REGRESSION (MonetaryTotal)
# ---------------------------------------------------------------------------
def train_regression_model(df):
    print("\n" + "=" * 62)
    print("  REGRESSION  (MonetaryTotal)")
    print("=" * 62)

    target_leaks = [
        'MonetaryAvg',
        'MonetaryStd',
        'MonetaryMin',
        'MonetaryMax',
        'MonetaryPerDay',
        'AvgBasketValue',
        'CustomerID',
    ]
    X = _safe_drop(df, target_leaks + ['MonetaryTotal'])
    y = df['MonetaryTotal']
    print(f"[i] Features : {X.shape[1]} colonnes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    candidates = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'RandomForest': RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        candidates['XGBoost'] = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )

    best_name = None
    best_r2 = -np.inf
    regression_metrics = {}
    best_pipeline = None

    for model_name, estimator in candidates.items():
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('regressor', estimator),
        ])
        print(f"[->] Entrainement {model_name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        regression_metrics[model_name] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
        }
        print(f"  {model_name}: R2={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
        if r2 > best_r2:
            best_r2 = r2
            best_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError('Aucun modele de regression valide.')

    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, 'regression_pipeline.pkl'))
    print(f"[OK] Meilleur modele -> {best_name}")
    return {
        'best_model': best_name,
        'models': regression_metrics,
    }


def save_dashboard_metrics(metrics: dict):
    joblib.dump(metrics, DASHBOARD_METRICS)
    print(f"[OK] Sauvegarde dashboard -> {DASHBOARD_METRICS}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    cols_path = os.path.join(MODELS_DIR, 'processed_columns.pkl')

    if not os.path.exists(PROCESSED_PATH):
        print("[!] Pretraitement initial necessaire...")
        df = preprocess_data(pd.read_csv(RAW_PATH), target_col='Churn')
        df.to_csv(PROCESSED_PATH, index=False)
        joblib.dump(df.columns.tolist(), cols_path)
    else:
        print(f"[OK] Chargement -> {PROCESSED_PATH}")
        df = pd.read_csv(PROCESSED_PATH)
        if os.path.exists(cols_path):
            expected = joblib.load(cols_path)
            missing  = set(expected) - set(df.columns)
            if missing:
                print(f"[!] Colonnes manquantes : {missing}. Re-pretraitement.")
                df = preprocess_data(pd.read_csv(RAW_PATH), target_col='Churn')
                df.to_csv(PROCESSED_PATH, index=False)
                joblib.dump(df.columns.tolist(), cols_path)
            else:
                df = df[[c for c in expected if c in df.columns]]
        else:
            print(f"[!] {cols_path} manquant. Sauvegarde des colonnes du dataset traité.")
            joblib.dump(df.columns.tolist(), cols_path)

    churn_scores = train_churn_model(df)
    clustering_metrics = train_clustering_model(df)
    regression_metrics = train_regression_model(df)

    dashboard = {
        'stats': {
            'n_clients': int(df.shape[0]),
            'n_features': int(df.shape[1]),
            'churn_rate': f"{df['Churn'].mean() * 100:.1f}%",
            'n_clusters': clustering_metrics['k'],
            'n_models': len(regression_metrics['models']) + 2,
            'best_regression_model': regression_metrics['best_model'],
        },
        'clustering': clustering_metrics,
        'churn': {
            'models': churn_scores,
            'labels': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
        },
        'regression': regression_metrics,
    }
    save_dashboard_metrics(dashboard)

    print("\n" + "=" * 62)
    print("  TOUS LES MODELES ENTRAINES ET SAUVEGARDES")
    print("=" * 62)


if __name__ == '__main__':
    main()