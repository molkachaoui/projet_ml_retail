# Projet Machine Learning - Analyse Comportementale Clientèle Retail

## 📋 Table des Matières

- [À Propos](#à-propos)
- [Objectifs](#objectifs)
- [Architecture du Projet](#architecture-du-projet)
- [Technologies Utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Modèles ML](#modèles-ml)
- [Résultats](#résultats)
- [API Flask](#api-flask)
- [Data Leakage](#data-leakage)
- [Auteur](#auteur)

## 🎯 À Propos

Ce projet implémente un système complet d'analyse comportementale clientèle pour une entreprise de retail. Il combine trois modèles de Machine Learning pour répondre aux besoins métier critiques : prédiction du churn, segmentation client, et prévision des dépenses.

Le système est déployable en production via une application web Flask avec API REST.

## 🚀 Objectifs

1. **Segmentation Clientèle** : Identifier 4 groupes homogènes (VIP, Fidèle, Occasionnel, À Risque)
2. **Prédiction du Churn** : Détecter les clients à risque de départ avec 85% de recall
3. **Prévision de Valeur** : Estimer les dépenses futures (MAE = 864 GBP)
4. **Déploiement** : Application web accessible aux équipes métier

## 📁 Architecture du Projet

```
ML_Project/
│
├── data/
│   ├── raw/
│   │   └── retail_customers_COMPLETE_CATEGORICAL.csv
│   └── processed/
│       └── processed_data.csv
│
├── src/
│   ├── preprocessing.py      # Pipeline de nettoyage (6 étapes)
│   ├── train_model.py        # Entraînement des 3 modèles
│   ├── predict.py            # Inférence sur nouvelles données
│   └── utils.py              # Fonctions utilitaires
│
├── models/
│   ├── churn_pipeline.pkl              # Random Forest + SMOTE
│   ├── clustering_preprocessor_pipeline.pkl
│   ├── kmeans_model.pkl                # KMeans k=4
│   └── regression_pipeline.pkl         # Linear Regression
│
├── app/
│   ├── app.py                # Application Flask
│   ├── templates/
│   │   ├── index.html        # Dashboard
│   │   ├── predict.html      # Formulaire de prédiction
│   │   └── segments.html     # Profils des segments
│   └── static/
│       ├── css/
│       └── js/
│
├── reports/
│   ├── rapport.pdf           # Rapport technique complet
│
├── notebooks/
│   ├── 01_EDA.ipynb         # Analyse exploratoire
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠️ Technologies Utilisées

### Machine Learning
- **scikit-learn 1.3+** : Algorithmes ML (Random Forest, KMeans, Linear Regression)
- **imbalanced-learn** : SMOTE pour gérer le déséquilibre des classes
- **pandas 2.0+** : Manipulation de données
- **numpy 1.24+** : Calculs numériques

### Visualisation
- **matplotlib 3.7+** : Graphiques statiques
- **seaborn 0.12+** : Visualisations statistiques
- **plotly 5.14+** : Graphiques interactifs

### Déploiement
- **Flask 2.3+** : Application web
- **joblib** : Sérialisation des modèles
- **gunicorn** : Serveur WSGI pour production

### Développement
- **jupyter** : Notebooks pour prototypage
- **pytest** : Tests unitaires
- **black** : Formatage de code

## 📦 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip
- virtualenv (recommandé)

### Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/ml-retail-analysis.git
cd ml-retail-analysis

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Vérifier l'installation
python -c "import sklearn; import flask; print('✓ Installation réussie')"
```

## 🎮 Utilisation

### 1️⃣ Prétraitement des Données

```bash
python src/preprocessing.py --input data/raw/retail_customers_COMPLETE_CATEGORICAL.csv \
                            --output data/processed/processed_data.csv
```

**Étapes du pipeline :**
1. Suppression features à variance nulle
2. Correction valeurs aberrantes (-1, 99, 999)
3. Parsing date d'inscription → 4 features temporelles
4. Feature engineering sur IP
5. Création features comportementales (AvgBasketValue)
6. Encodage catégorielles (Ordinal, One-Hot, Target)

### 2️⃣ Entraînement des Modèles

```bash
python src/train_model.py --data data/processed/processed_data.csv \
                          --output-dir models/
```

**Modèles entraînés :**
- `churn_pipeline.pkl` : Random Forest (200 arbres, max_depth=10)
- `kmeans_model.pkl` : KMeans (k=4, features RFM)
- `regression_pipeline.pkl` : Linear Regression

### 3️⃣ Prédictions

```bash
# Prédiction sur un seul client
python src/predict.py --input data/new_customer.json

# Prédiction batch
python src/predict.py --input data/customers_batch.csv \
                      --output predictions.csv
```

**Exemple de fichier JSON :**
```json
{
  "Frequency": 5,
  "MonetaryTotal": 1200,
  "Age": 35,
  "Gender": "M",
  "Country": "UK",
  ...
}
```

### 4️⃣ Lancer l'Application Flask

```bash
# Mode développement
python app/app.py

# Mode production
gunicorn -w 4 -b 0.0.0.0:5000 app.app:app
```

Accédez à : `http://localhost:5000`

## 🤖 Modèles ML

### Modèle 1 : Classification du Churn (Random Forest)

**Pipeline :**
```python
ImbPipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
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
|----------|--------|
| AUC | **0.958** |
| Accuracy | 88% |
| Precision (Churn=1) | 0.81 |
| Recall (Churn=1) | **0.85** |
| F1-Score | 0.87 |

**Top 5 Features Importantes :**
1. `FavoriteSeason_Automne` (0.238)
2. `PreferredMonth` (0.126)
3. `Frequency` (0.063)
4. `UniqueInvoices` (0.062)
5. `FavoriteSeason_Printemps` (0.053)

### Modèle 2 : Clustering (KMeans)

**Prétraitement spécifique :**
- Sélection : 9 features RFM uniquement
- Filtrage : Exclusion clients avec MonetaryTotal ≤ 0
- Winsorisation : Clip aux percentiles 1 et 99
- Log-transform : `np.log1p()` sur variables asymétriques
- Normalisation : RobustScaler

**k=4 segments identifiés :**

| Segment | Taille | Monetary | Recency | Frequency | Churn |
|---------|--------|----------|---------|-----------|-------|
| **VIP** | 1 623 | 3 660€ | 34j | 8.8 | **8.3%** |
| **Fidèle** | 461 | 2 708€ | 79j | 8.1 | 27.8% |
| **Occasionnel** | 136 | 1 795€ | 109j | 5.9 | 42.6% |
| **À Risque** | 2 100 | 420€ | 135j | 1.6 | **52.1%** |

**Métriques :**
- Silhouette Score : 0.228 (acceptable)
- Inertie : 20 586 522

### Modèle 3 : Régression (Linear Regression)

**Pipeline :**
```python
Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()),
    ('regressor', LinearRegression())
])
```

**Performances :**
- **R² = 0.668** (explique 66.8% de la variance)
- **MAE = 863.83 GBP** (erreur moyenne)
- RMSE = 1 124 GBP

## 📊 Résultats

### Synthèse des Performances

```
┌─────────────────────┬──────────────┬─────────┬─────────────────┐
│ Modèle              │ Métrique     │ Valeur  │ Interprétation  │
├─────────────────────┼──────────────┼─────────┼─────────────────┤
│ Random Forest       │ AUC          │ 0.958   │ Excellent       │
│ (Churn)             │ Recall       │ 0.85    │ Détecte 85%     │
├─────────────────────┼──────────────┼─────────┼─────────────────┤
│ KMeans              │ Silhouette   │ 0.228   │ Acceptable      │
│ (Clustering)        │ Segments     │ 4       │ Actionnables    │
├─────────────────────┼──────────────┼─────────┼─────────────────┤
│ Linear Regression   │ R²           │ 0.668   │ Acceptable      │
│ (Prévision)         │ MAE          │ 864 GBP │ Erreur moyenne  │
└─────────────────────┴──────────────┴─────────┴─────────────────┘
```

### Applications Métier

✅ **Segmentation** : Identifier les 1 623 VIP pour offres premium  
✅ **Rétention** : Cibler les 2 100 clients À Risque (52% churn)  
✅ **Prévision** : Estimer revenus futurs ±864£ par client  
✅ **Automatisation** : API JSON pour intégration CRM/ERP

## 🌐 API Flask

### Endpoints

#### 1. Dashboard Principal
```
GET /
```
Affiche les métriques de performance des 3 modèles.

#### 2. Formulaire de Prédiction
```
GET /predict
```
Interface web pour saisir les features d'un client.

#### 3. API JSON - Prédiction
```
POST /api/predict
Content-Type: application/json

{
  "Frequency": 5,
  "MonetaryTotal": 1200,
  "Age": 35,
  "Gender": "M",
  "Country": "UK",
  ...
}
```

**Réponse :**
```json
{
  "churn": 0,
  "churn_probability": 0.15,
  "risk_level": "Faible",
  "cluster": 1,
  "cluster_name": "Fidèle",
  "monetary_prediction": 1850.25,
  "recommendations": [
    "Client à faible risque de churn",
    "Segment Fidèle - Maintenir engagement"
  ]
}
```

#### 4. Visualisation des Segments
```
GET /segments
```
Affiche les profils détaillés des 4 segments clients.

### Exemple cURL

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Frequency": 5,
    "MonetaryTotal": 1200,
    "Age": 35,
    "Gender": "M",
    "Country": "UK",
    "AgeCategory": "35-44",
    "PreferredMonth": 3,
    "FavoriteSeason": "Printemps",
    "SatisfactionScore": 4,
    "SupportTicketsCount": 1
  }'
```

## ⚠️ Data Leakage

### Problème Identifié

**Symptôme initial** : AUC = 1.00 (perfection irréaliste)

### Cause Racine

14 features contenaient des informations du futur ou de la cible :

| Feature | Corrélation | Raison du Leakage |
|---------|-------------|-------------------|
| `ChurnRiskCategory` | 0.88 | Construite pour prédire le churn |
| `Recency` | 0.86 | Client parti = Recency élevée |
| `CustomerType_Perdu` | 0.70 | "Perdu" = churné par définition |
| `RFMSegment_Dormants` | 0.58 | "Dormant" = inactif |
| `LoyaltyLevel` | -0.43 | Calculé sur historique récent |
| `CustomerTenureDays` | -0.45 | Durée courte = parti |

### Solution Appliquée

**Exclusion de 14 features fuyantes :**

```python
LEAKY_CHURN_FEATURES = [
    # Proxies temporels directs
    'Recency', 'CustomerTenureDays', 'FirstPurchaseDaysAgo',
    'TenureRatio', 'MonetaryPerDay',
    
    # Catégories synthétiques
    'ChurnRiskCategory', 'LoyaltyLevel', 'SpendingCategory',
    
    # One-Hot de CustomerType
    'CustomerType_Perdu', 'CustomerType_Hyperactif', 
    'CustomerType_Nouveau', 'CustomerType_Occasionnel', 
    'CustomerType_Regulier',
    
    # One-Hot de RFMSegment
    'RFMSegment_Champions', 'RFMSegment_Dormants', 
    'RFMSegment_Fideles', 'RFMSegment_Potentiels',
    
    # One-Hot de AccountStatus
    'AccountStatus_Closed', 'AccountStatus_Suspended',
    'AccountStatus_Active', 'AccountStatus_Pending'
]
```

### Résultat Après Correction

**AUC : 1.00 → 0.958** (performances réalistes et généralisables)

## 🔍 Défis Résolus

### 1. Data Leakage (Churn)
❌ **Avant** : AUC = 1.00 (14 features fuyantes)  
✅ **Après** : AUC = 0.958 (exclusion rigoureuse)

### 2. Clustering Trivial
❌ **Avant** : Silhouette = 0.99 (clusters dominés par outliers)  
✅ **Après** : Silhouette = 0.228 (winsorisation + log-transform)

### 3. Déséquilibre Classes (Churn)
❌ **Avant** : 67% Fidèle / 33% Churn  
✅ **Après** : SMOTE + class_weight='balanced'

### 4. Imputation Sans Fuite
❌ **Avant** : SimpleImputer avant split train/test  
✅ **Après** : KNNImputer dans les pipelines

## 📈 Améliorations Futures

### Court Terme
- [ ] Hyperparameter tuning avec GridSearchCV
- [ ] Feature selection automatique (RFE)
- [ ] Tests unitaires avec pytest
- [ ] Documentation API avec Swagger

### Moyen Terme
- [ ] Déploiement Docker + Kubernetes
- [ ] Monitoring drift avec Evidently AI
- [ ] A/B testing des modèles
- [ ] Dashboard interactif avec Streamlit

### Long Terme
- [ ] Deep Learning pour séquences temporelles
- [ ] AutoML avec H2O.ai
- [ ] MLOps pipeline complet (MLflow)
- [ ] Prédictions en temps réel (Kafka)

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/

# Tests avec couverture
pytest --cov=src tests/

# Tests spécifiques
pytest tests/test_preprocessing.py -v
```

## 📝 Documentation

- **Rapport technique complet** : `reports/rapport.pdf`
- **Notebooks exploratoires** : `notebooks/`
- **Documentation API** : `http://localhost:5000/api/docs` (après lancement Flask)
