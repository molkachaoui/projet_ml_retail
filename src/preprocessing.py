import pandas as pd
import numpy as np
import ipaddress
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Mappings pour l'encodage ordinal
ORDINAL_MAPPINGS = {
    'AgeCategory': {
        '18-24': 0, '25-34': 1, '35-44': 2,
        '45-54': 3, '55-64': 4, '65+': 5, 'Inconnu': -1
    },
    'SpendingCategory': {
        'Bas': 0, 'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3, 'VIP': 3, 'Inconnu': -1
    },
    'LoyaltyLevel': {
        'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3, 'Jeune': 0, 'Inconnu': -1
    },
    'ChurnRiskCategory': {
        'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3, 'Critique': 3, 'Inconnu': -1
    },
    'BasketSizeCategory': {
        'Small': 0, 'Medium': 1, 'Large': 2, 'Very Large': 3, 'Moyen': 1, 'Inconnu': -1
    },
    'PreferredTimeOfDay': {
        'Morning': 0, 'Matin': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3, 'Inconnu': -1
    },
}

BINARY_MAPPINGS = {
    'NewsletterSubscribed': {
        'Yes': 1, 'No': 0, 'Oui': 1, 'Non': 0, 'Inconnu': -1
    }
}

ONE_HOT_COLS = [
    'RFMSegment', 'CustomerType', 'FavoriteSeason', 'Region',
    'WeekendPreference', 'ProductDiversity', 'Gender', 'AccountStatus',
]

# --- Preprocessing helpers ---

def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[AUDIT] shape={df.shape}")
    print(f"[AUDIT] types:\n{df.dtypes.value_counts().to_dict()}")
    missing = df.isna().sum()
    if missing.sum() > 0:
        print("[AUDIT] valeurs manquantes par colonne :")
        print(missing[missing > 0].sort_values(ascending=False).to_dict())
    dup_count = df.duplicated().sum()
    if dup_count:
        print(f"[AUDIT] duplications detectees : {dup_count}")
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    skip = {'RegistrationDate', 'LastLoginIP', 'Country'}
    object_cols = [c for c in df.select_dtypes(include=['object']).columns if c not in skip]
    for col in object_cols:
        cleaned = df[col].astype(str).str.replace(',', '.', regex=False)
        numeric = pd.to_numeric(cleaned, errors='coerce')
        if numeric.notna().sum() >= len(df) * 0.5:
            df[col] = numeric
            print(f"[OK] Conversion numerique : {col}")
    return df


def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    invalid_codes = {
        'SupportTicketsCount': [-1, 999],
        'SatisfactionScore': [-1, 99],
        'Age': [-1, 999],
    }
    for col, codes in invalid_codes.items():
        if col in df.columns:
            before = df[col].isna().sum()
            df.loc[df[col].isin(codes), col] = np.nan
            after = df[col].isna().sum()
            if after > before:
                print(f"[OK] Nettoyage {col} : {after-before} valeurs invalides remplacees")

    if 'Age' in df.columns:
        mask = (df['Age'] < 10) | (df['Age'] > 100)
        if mask.any():
            df.loc[mask, 'Age'] = np.nan
            print(f"[OK] Age reeliste garde : {mask.sum()} ages extrêmes retirés")

    return df


def parse_registration_date(df: pd.DataFrame, col: str = 'RegistrationDate') -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    df['RegYear'] = df[col].dt.year
    df['RegMonth'] = df[col].dt.month
    df['RegDay'] = df[col].dt.day
    df['RegWeekday'] = df[col].dt.weekday
    n_failed = int(df[col].isna().sum())
    if n_failed > 0:
        print(f"[!] {n_failed} dates non parsees -> NaT")
    df.drop(columns=[col], inplace=True)
    print(f"[OK] '{col}' -> RegYear, RegMonth, RegDay, RegWeekday")
    return df


def engineer_ip_features(df: pd.DataFrame, col: str = 'LastLoginIP') -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df

    def _is_private(ip_str):
        try:
            return int(ipaddress.ip_address(str(ip_str)).is_private)
        except Exception:
            return np.nan

    def _first_octet(ip_str):
        try:
            return int(str(ip_str).split('.')[0])
        except Exception:
            return np.nan

    df['IP_IsPrivate'] = df[col].apply(_is_private)
    df['IP_FirstOctet'] = df[col].apply(_first_octet)
    df.drop(columns=[col], inplace=True)
    print(f"[OK] '{col}' -> IP_IsPrivate, IP_FirstOctet")
    return df


def business_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'MonetaryTotal' in df.columns and 'CustomerTenureDays' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / df['CustomerTenureDays'].replace(0, np.nan)
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency'].replace(0, np.nan)
    if 'CancelledTransactions' in df.columns and 'TotalTransactions' in df.columns:
        df['CancelRate'] = df['CancelledTransactions'] / df['TotalTransactions'].replace(0, np.nan)
    if 'TotalQuantity' in df.columns and 'TotalTransactions' in df.columns:
        df['ProductsPerTrans'] = df['TotalQuantity'] / df['TotalTransactions'].replace(0, np.nan)
    print("[OK] Features comportementales creees : MonetaryPerDay, AvgBasketValue, CancelRate, ProductsPerTrans")
    return df


def drop_irrelevant_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    to_drop = [col for col in ['CustomerID', 'RegistrationDate', 'LastLoginIP'] if col in df.columns]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"[OK] Colonnes irrelvantes supprimees : {to_drop}")
    return df


def drop_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    to_drop = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"[OK] Colonnes constantes supprimees : {to_drop}")
    return df


def encode_categorical_features(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)
            print(f"[OK] '{col}' encodee ordinalement.")

    existing_ohe = [c for c in ONE_HOT_COLS if c in df.columns]
    if existing_ohe:
        df = pd.get_dummies(df, columns=existing_ohe, drop_first=False, dtype=int)
        print(f"[OK] {existing_ohe} encodees One-Hot.")

    if 'Country' in df.columns and target_col and target_col in df.columns:
        country_target_mean = df.groupby('Country')[target_col].mean()
        df['Country_encoded'] = df['Country'].map(country_target_mean)
        df.drop(columns=['Country'], inplace=True)
        print(f"[OK] Target encoded : Country -> Country_encoded")
    elif 'Country' in df.columns:
        df.drop(columns=['Country'], inplace=True)
        print("[!] 'Country' supprimee (pas de target_col).")

    return df

def impute_missing_values(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Age' in df.columns:
        support_cols = [c for c in ['Frequency', 'MonetaryTotal', 'Recency', 'TotalTransactions', 'AvgDaysBetweenPurchases', 'AvgBasketValue'] if c in df.columns]
        age_cols = ['Age'] + support_cols
        if len(age_cols) > 1:
            if df.shape[0] > 1 and df.shape[0] >= n_neighbors:
                imputer = KNNImputer(n_neighbors=n_neighbors)
                age_imputed = imputer.fit_transform(df[age_cols])
                if age_imputed.ndim == 2 and age_imputed.shape[1] == len(age_cols):
                    df[age_cols] = age_imputed
                    print(f"[OK] Age imputee par KNN avec {len(support_cols)} variables de support.")
                else:
                    df[age_cols] = pd.DataFrame(age_imputed, columns=age_cols, index=df.index)
                    print("[OK] Age imputee par KNN (fallback conversion).")
            else:
                # Pas assez d'exemples pour KNN ; on utilise la mediane locale
                medians = df[age_cols].median()
                df[age_cols] = df[age_cols].fillna(medians)
                print("[OK] Age imputee par mediane (pas assez d'exemples pour KNN).")

    remaining = [c for c in numeric_cols if c != 'Age']
    if remaining:
        medians = df[remaining].median()
        df[remaining] = df[remaining].fillna(medians)
        print(f"[OK] Imputation mediane appliquee sur {len(remaining)} colonnes numeriques.")
    return df


def preprocess_data(df: pd.DataFrame, target_col: str = None, drop_constant: bool = True) -> pd.DataFrame:
    df_processed = df.copy()
    if len(df_processed) > 1:
        df_processed = audit_data(df_processed)
    df_processed = convert_numeric_columns(df_processed)
    df_processed = clean_invalid_values(df_processed)
    df_processed = parse_registration_date(df_processed)
    df_processed = engineer_ip_features(df_processed)
    df_processed = business_feature_engineering(df_processed)
    df_processed = drop_irrelevant_features(df_processed)
    df_processed = encode_categorical_features(df_processed, target_col)
    df_processed = impute_missing_values(df_processed)
    if drop_constant and len(df_processed) > 1:
        df_processed = drop_constant_features(df_processed)
    print("[OK] Pretraitement initial termine.")
    return df_processed


# =============================================================================
# LISTES DES FEATURES FUYANTES (DATA LEAKAGE)
# =============================================================================

# --- Churn Classification ---
# Ces colonnes sont des proxies DIRECTS ou quasi-directs du Churn.
#
# * Recency (corr=0.86) : Un client parti n'a pas achete depuis longtemps
#   -> fuite parfaite. C'est la principale cause de l'AUC=1.00.
#
# * ChurnRiskCategory (corr=0.88) : Cette colonne est CONSTRUITE pour
#   predire le churn. La mettre en feature revient a donner la reponse.
#
# * CustomerTenureDays (corr=-0.45) : Duree de vie courte = client parti.
#
# * CustomerType_Perdu (corr=0.70) : "Perdu" signifie churne.
#
# * RFMSegment_Dormants (corr=0.58) : "Dormant" = proxy du churn.
#
# * LoyaltyLevel (corr=-0.43), SpendingCategory (corr=-0.38) : Ces
#   categories sont calculees sur l'historique recent -> fuite indirecte.
#
# * AccountStatus_Closed/Suspended : Un compte ferme = parti.
#
# * TenureRatio, MonetaryPerDay : Derives de Recency/Tenure -> fuite.
LEAKY_CHURN_FEATURES = [
    # Proxies temporels directs
    'Recency',
    'CustomerTenureDays',
    'FirstPurchaseDaysAgo',
    'TenureRatio',
    'MonetaryPerDay',
    # Categories construites sur le comportement churn
    'ChurnRiskCategory',
    'LoyaltyLevel',
    'SpendingCategory',
    # One-Hot issues de CustomerType
    'CustomerType_Perdu',
    'CustomerType_Hyperactif',
    'CustomerType_Nouveau',
    'CustomerType_Occasionnel',
    'CustomerType_Regulier',
    # One-Hot issues de RFMSegment
    'RFMSegment_Champions',
    'RFMSegment_Dormants',
    'RFMSegment_Fideles',
    'RFMSegment_Potentiels',
    # One-Hot issues de AccountStatus
    'AccountStatus_Closed',
    'AccountStatus_Suspended',
    'AccountStatus_Active',
    'AccountStatus_Pending',
]

# Variantes avec accents (au cas ou le CSV les contient)
LEAKY_CHURN_FEATURES += [
    'CustomerType_R\u00e9gulier',
    'RFMSegment_Fid\u00e8les',
]

# --- Clustering KMeans ---
# Pour le clustering, on retire Churn (la cible) et toutes les colonnes
# categorielles synthetiques qui resumaient deja le comportement client.
# Raison du Silhouette=0.99 : RFMSegment, CustomerType, SpendingCategory,
# LoyaltyLevel etaient inclus -> le modele clusterisait sur des labels
# pre-existants, pas sur le comportement brut.
# On garde : Frequency, MonetaryTotal, MonetaryAvg, TotalQuantity,
# SatisfactionScore, SupportTicketsCount, Age, etc.
LEAKY_CLUSTERING_FEATURES = [
    'Churn',
    # Categories synthetiques resumant deja le comportement
    'ChurnRiskCategory',
    'SpendingCategory',
    'LoyaltyLevel',
    'BasketSizeCategory',
    # Segments pre-definis = les futurs labels du clustering
    'RFMSegment_Champions',
    'RFMSegment_Dormants',
    'RFMSegment_Fideles',
    'RFMSegment_Potentiels',
    'RFMSegment_Fid\u00e8les',
    'CustomerType_Hyperactif',
    'CustomerType_Nouveau',
    'CustomerType_Occasionnel',
    'CustomerType_Perdu',
    'CustomerType_Regulier',
    'CustomerType_R\u00e9gulier',
    # Statuts de compte biaisent fortement les clusters
    'AccountStatus_Closed',
    'AccountStatus_Suspended',
]