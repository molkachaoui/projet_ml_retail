# 🛒 Retail ML — Analyse Comportementale Clientèle

> Système complet de Machine Learning pour la rétention client, la segmentation et la prévision des dépenses — déployé via Flask.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3+-green?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?logo=bootstrap)

---

## 📋 Table des Matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Démonstration](#-démonstration)
3. [Architecture](#-architecture-du-projet)
4. [Modèles ML](#-modèles-ml)
5. [Data Leakage — Problème & Solution](#️-data-leakage--problème--solution)
6. [Installation](#-installation)
7. [Utilisation](#-utilisation)
8. [API REST](#-api-rest)
9. [Résultats & Performances](#-résultats--performances)
10. [Recommandations Métier](#-recommandations-métier)
11. [Améliorations Futures](#-améliorations-futures)

---

## 🎯 Vue d'ensemble

Ce projet implémente un pipeline bout-en-bout d'analyse comportementale pour une entreprise de retail e-commerce (UK). Il répond à trois questions métier critiques :

| Question | Modèle | Performance |
|---|---|---|
| Ce client va-t-il partir ? | Random Forest (classification) | AUC = **0.958**, F1 = **0.85** |
| À quel segment appartient-il ? | KMeans (clustering, k=4) | Silhouette = **0.228** |
| Combien va-t-il dépenser ? | Régression Linéaire | R² = **0.668**, MAE = 864 £ |

**Dataset :** 4 372 clients, 52 features initiales → 87 après encodage, taux de churn global de 33.3%.

---

## 🖥️ Démonstration

L'application Flask expose 4 pages :

| Route | Description |
|---|---|
| `/` | Dashboard principal — métriques des 3 modèles, aperçu des segments |
| `/predict` | Formulaire RFM → prédiction churn + segment + valeur monétaire |
| `/segments` | Profils détaillés des 4 segments clients |
| `/about` | Pipeline technique, stack, recommandations métier |

**Exemple de prédiction :**
- Client haute fréquence (80 commandes, 2999 £ dépensés) → **Churn 15.9%**, segment *Clients Fidèles*, valeur estimée 88 789 £
- Client basse fréquence → **Churn 96.4%**, segment *Clients Occasionnels*, valeur estimée 3 552 £

---

## 📁 Architecture du Projet

```
ML_Project/
│
├── data/
│   ├── raw/
│   │   └── retail_customers_COMPLETE_CATEGORICAL.csv
│   ├── processed/
│   │   └── retail_customers_processed.csv
│   └── train_test/
│
├── src/
│   ├── preprocessing.py      # Pipeline nettoyage (6 étapes)
│   ├── train_model.py        # Entraînement des 3 modèles
│   ├── predict.py            # Inférence sur nouvelles données
│   └── utils.py              # Fonctions utilitaires
│
├── models/
│   ├── churn_pipeline.pkl                    # Random Forest + SMOTE
│   ├── clustering_preprocessor_pipeline.pkl  # KNN Imputer + Scaler + PCA
│   ├── kmeans_model.pkl                      # KMeans k=4
│   ├── regression_pipeline.pkl               # Linear Regression
│   ├── cluster_label_mapping.pkl             # id → label métier
│   ├── processed_columns.pkl                 # Colonnes après preprocessing
│   └── dashboard_metrics.pkl                 # Métriques pré-calculées
│
├── app/
│   ├── app.py
│   └── templates/
│       ├── base.html
│       ├── index.html        # Dashboard dark-mode
│       ├── predict.html      # Formulaire de prédiction
│       ├── segments.html     # Profils segments
│       └── about.html        # Description projet
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── reports/
│   └── rapport.pdf
│
├── requirements.txt
└── README.md
```

---

## 🤖 Modèles ML

### 1 — Classification du Churn (Random Forest)

Détecte les clients à risque de départ avant qu'ils ne partent.

**Pipeline scikit-learn :**
```python
ImbPipeline([
    ('imputer',    KNNImputer(n_neighbors=5)),
    ('scaler',     RobustScaler()),
    ('smote',      SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        class_weight='balanced'
    ))
])
```

**Performances :**

| Métrique | Valeur |
|---|---|
| AUC-ROC | **0.958** |
| Accuracy | 88% |
| Precision (Churn=1) | 0.81 |
| Recall (Churn=1) | **0.85** |
| F1-Score | 0.87 |

**Top 5 features importantes :**
1. `FavoriteSeason_Automne` — 23.8%
2. `PreferredMonth` — 12.6%
3. `Frequency` — 6.3%
4. `UniqueInvoices` — 6.2%
5. `FavoriteSeason_Printemps` — 5.3%

> **Note :** Le seuil de décision peut être abaissé de 0.5 → 0.35 pour réduire les faux négatifs (clients partis non détectés).

---

### 2 — Segmentation Clients (KMeans, k=4)

Regroupe les clients en 4 segments homogènes sur la base de leurs comportements RFM.

**Prétraitement spécifique :**
- Sélection de 9 features RFM uniquement
- Filtrage : clients avec `MonetaryTotal > 0`
- Winsorisation aux percentiles 1–99
- Log-transform (`np.log1p`) sur les variables asymétriques
- Normalisation avec `RobustScaler`
- Réduction de dimension par ACP (20 composantes)

**Segments identifiés :**

| Segment | Taille | Monetary moy. | Recency moy. | Frequency moy. | Churn |
|---|---|---|---|---|---|
| 🌟 **VIP** | 1 623 clients | 3 660 £ | 34 j | 8.8 | **8.3%** |
| 💙 **Fidèle** | 461 clients | 2 708 £ | 79 j | 8.1 | 27.8% |
| ⏱️ **Occasionnel** | 136 clients | 1 795 £ | 109 j | 5.9 | 42.6% |
| ⚠️ **À Risque** | 2 100 clients | 420 £ | 135 j | 1.6 | **52.1%** |

**Métriques clustering :**
- Silhouette Score : **0.228** (acceptable sur données comportementales continues)
- Davies-Bouldin : 1.30
- Calinski-Harabasz : 692
- Inertie : 20 586 522

---

### 3 — Prévision de la Valeur Client (Régression Linéaire)

Estime le montant total dépensé par un client.

**Pipeline :**
```python
Pipeline([
    ('imputer',    KNNImputer(n_neighbors=5)),
    ('scaler',     RobustScaler()),
    ('regressor',  LinearRegression())
])
```

**Performances :**

| Métrique | Valeur |
|---|---|
| R² | **0.668** |
| MAE | **863.83 £** |
| RMSE | 1 124 £ |

---

## ⚠️ Data Leakage — Problème & Solution

### Symptôme
AUC initial = **1.00** → perfection irréaliste, signe d'une fuite de données.

### Cause
14 features contenaient des informations du futur ou dérivées directement de la cible :

| Feature | Corrélation avec Churn | Raison |
|---|---|---|
| `ChurnRiskCategory` | 0.88 | Construite pour prédire le churn |
| `Recency` | 0.86 | Client parti = Recency élevée |
| `CustomerType_Perdu` | 0.70 | "Perdu" = churné par définition |
| `RFMSegment_Dormants` | 0.58 | "Dormant" = inactif |
| `LoyaltyLevel` | -0.43 | Calculé sur historique récent |

### Solution
Exclusion stricte de 14 features fuyantes via `LEAKY_CHURN_FEATURES` dans `preprocessing.py` :

```python
LEAKY_CHURN_FEATURES = [
    'Recency', 'CustomerTenureDays', 'FirstPurchaseDaysAgo',
    'TenureRatio', 'MonetaryPerDay',
    'ChurnRiskCategory', 'LoyaltyLevel', 'SpendingCategory',
    'CustomerType_Perdu', 'CustomerType_Hyperactif',
    'CustomerType_Nouveau', 'CustomerType_Occasionnel',
    'CustomerType_Regulier',
    'RFMSegment_Champions', 'RFMSegment_Dormants',
    'RFMSegment_Fideles', 'RFMSegment_Potentiels',
    'AccountStatus_Closed', 'AccountStatus_Suspended',
    'AccountStatus_Active', 'AccountStatus_Pending'
]
```

**Résultat :** AUC 1.00 → **0.958** (performances réalistes et généralisables).

---

## 📦 Installation

### Prérequis
- Python 3.8+
- pip
- virtualenv (recommandé)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/ml-retail-analysis.git
cd ml-retail-analysis

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Vérifier l'installation
python -c "import sklearn; import flask; print('✓ Installation réussie')"
```

---

## 🎮 Utilisation

### Étape 1 — Prétraitement

```bash
python src/preprocessing.py \
  --input  data/raw/retail_customers_COMPLETE_CATEGORICAL.csv \
  --output data/processed/retail_customers_processed.csv
```

Le pipeline applique 6 étapes :
1. Suppression des features à variance nulle
2. Correction des valeurs aberrantes (-1, 99, 999)
3. Parsing de la date d'inscription → 4 features temporelles
4. Feature engineering sur IP
5. Création de features comportementales (`AvgBasketValue`, etc.)
6. Encodage catégorielles (Ordinal, One-Hot, Target Encoding)

### Étape 2 — Entraînement

```bash
python src/train_model.py \
  --data       data/processed/retail_customers_processed.csv \
  --output-dir models/
```

Génère les 7 artefacts dans `models/`.

### Étape 3 — Lancer l'application

```bash
# Développement
python app/app.py

# Production
gunicorn -w 4 -b 0.0.0.0:5000 app.app:app
```

Accédez à : **http://localhost:5000**

---

## 🌐 API REST

### `POST /api/predict`

Prédit churn, segment et valeur monétaire pour un client donné.

**Requête :**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Frequency": 5,
    "MonetaryTotal": 1200,
    "Age": 35,
    "Gender": "M",
    "AgeCategory": "35-44",
    "PreferredMonth": 3,
    "FavoriteSeason": "Printemps",
    "SatisfactionScore": 4,
    "SupportTicketsCount": 1
  }'
```

**Réponse :**
```json
{
  "churn": 0,
  "probability": 0.15,
  "risk": "Faible",
  "cluster": 1,
  "cluster_name": "Fidele",
  "monetary_prediction": 1850.25
}
```

### Autres endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Dashboard principal |
| `GET/POST` | `/predict` | Interface de prédiction |
| `GET` | `/segments` | Profils des 4 segments |
| `GET` | `/about` | Documentation projet |
| `POST` | `/run-models` | Relance l'entraînement |

---

## 📊 Résultats & Performances

```
┌─────────────────────┬──────────────┬──────────┬──────────────────────┐
│ Modèle              │ Métrique     │ Valeur   │ Interprétation       │
├─────────────────────┼──────────────┼──────────┼──────────────────────┤
│ Random Forest       │ AUC-ROC      │ 0.958    │ Excellent            │
│ (Churn)             │ Recall       │ 0.85     │ Détecte 85% des      │
│                     │              │          │ churners             │
├─────────────────────┼──────────────┼──────────┼──────────────────────┤
│ KMeans              │ Silhouette   │ 0.228    │ Acceptable (données  │
│ (Clustering)        │ Segments     │ 4        │ comportementales)    │
├─────────────────────┼──────────────┼──────────┼──────────────────────┤
│ Linear Regression   │ R²           │ 0.668    │ Acceptable           │
│ (Prévision)         │ MAE          │ 864 £    │ Erreur moyenne       │
└─────────────────────┴──────────────┴──────────┴──────────────────────┘
```

### Défis résolus

| Problème | Avant | Après | Solution |
|---|---|---|---|
| Data leakage churn | AUC = 1.00 | AUC = 0.958 | Exclusion 14 features fuyantes |
| Clustering trivial | Silhouette = 0.99 | Silhouette = 0.228 | Winsorisation + log-transform |
| Déséquilibre classes | 67/33% | Équilibré | SMOTE + class_weight='balanced' |
| Fuite dans imputation | Avant split | Dans pipeline | KNNImputer intégré au Pipeline |

---

## 💼 Recommandations Métier

### Réduction du Churn
- Campagnes de rétention ciblant les clients avec ancienneté < 6 mois
- Offres exclusives avant la saison Automne (feature #1 en importance)
- Abaisser le seuil de décision de 0.5 → 0.35 pour minimiser les faux négatifs

### Personnalisation Marketing
- **Segment VIP (8.3% churn)** : programme premium, ventes privées, early access
- **Segment Fidèle (27.8%)** : campagnes de rétention, cross-selling ciblé
- **Segment Occasionnel (42.6%)** : promotions saisonnières, réactivation
- **Segment À Risque (52.1%)** : enquête retours, geste commercial, offre de réengagement

### Optimisation du CA
- Scorer les clients par potentiel de dépense avant chaque campagne
- Envisager un modèle XGBoost dédié aux clients avec dépenses élevées (outliers)
- Collecter des données de navigation et de wishlist pour enrichir les features

---

## 🛠️ Stack Technique

| Catégorie | Technologies |
|---|---|
| **ML / Data** | scikit-learn 1.3+, imbalanced-learn, pandas 2.0+, numpy 1.24+ |
| **Visualisation** | matplotlib, seaborn, plotly, Chart.js |
| **Déploiement** | Flask 2.3+, joblib, gunicorn |
| **Frontend** | Bootstrap 5.3, Font Awesome 6.4 |
| **Dev** | Jupyter, pytest, black |

---

## 📈 Améliorations Futures

### Court terme
- [ ] Hyperparameter tuning (GridSearchCV / Optuna)
- [ ] Feature selection automatique (RFE, SHAP)
- [ ] Tests unitaires avec pytest
- [ ] Documentation API avec Swagger / OpenAPI

### Moyen terme
- [ ] Déploiement Docker + Kubernetes
- [ ] Monitoring du data drift (Evidently AI)
- [ ] Dashboard interactif avec Streamlit ou Dash
- [ ] A/B testing des modèles en production

### Long terme
- [ ] Deep Learning pour séquences temporelles d'achats
- [ ] Pipeline MLOps complet (MLflow + DVC)
- [ ] Prédictions en temps réel (Kafka + streaming)
- [ ] AutoML (H2O.ai) pour exploration automatique

---

## 👤 Auteur

Projet réalisé dans le cadre du **Module Machine Learning — GI2S1 MolkaChaoui — 2025-2026**

*Analyse Comportementale Clientèle Retail — Atelier Machine Learning*
