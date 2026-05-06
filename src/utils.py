"""
utils.py — Fonctions utilitaires partagées
Projet : Analyse Comportementale Clientèle Retail
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Charge un CSV et affiche un résumé rapide."""
    df = pd.read_csv(filepath)
    print(f"[✓] Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────────────────────
# 2. RÉSUMÉ RAPIDE
# ─────────────────────────────────────────────────────────────

def quick_summary(df: pd.DataFrame) -> None:
    """Affiche un résumé du DataFrame : shape, types, valeurs manquantes."""
    print("=" * 60)
    print(f"  SHAPE         : {df.shape}")
    print(f"  COLONNES      : {df.shape[1]}")
    print("-" * 60)
    print("  TYPES DE DONNÉES :")
    print(df.dtypes.value_counts().to_string())
    print("-" * 60)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  VALEURS MANQUANTES : aucune ✓")
    else:
        pct = (missing / len(df) * 100).round(2)
        print("  VALEURS MANQUANTES :")
        print(pd.DataFrame({'Count': missing, 'Percent (%)': pct}).to_string())
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 3. HEATMAP DE CORRÉLATION
# ─────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame,
                              threshold: float = 0.8,
                              save_path: str = None) -> None:
    """
    Affiche la heatmap de corrélation des features numériques.
    Signale les paires avec |corr| > threshold.
    """
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    plt.figure(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                annot=False, linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Matrice de Corrélation — Features Numériques", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[✓] Heatmap sauvegardée : {save_path}")
    plt.show()

    # Paires fortement corrélées
    high_corr = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr.iloc[i, j]) > threshold:
                high_corr.append((cols[i], cols[j], round(corr.iloc[i, j], 3)))

    if high_corr:
        print(f"\n[!] Paires fortement corrélées (|corr| > {threshold}) :")
        for a, b, v in high_corr:
            print(f"    {a}  ↔  {b}  :  {v}")
    else:
        print(f"\n[✓] Aucune paire avec |corr| > {threshold}")


# ─────────────────────────────────────────────────────────────
# 4. DÉTECTION OUTLIERS (IQR)
# ─────────────────────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Détecte les outliers via la méthode IQR pour chaque colonne fournie.
    Retourne un DataFrame résumé.
    """
    results = []
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        results.append({
            'Feature':        col,
            'Q1':             round(Q1, 3),
            'Q3':             round(Q3, 3),
            'IQR':            round(IQR, 3),
            'Lower bound':    round(lower, 3),
            'Upper bound':    round(upper, 3),
            'Outliers count': n_out,
            'Outliers (%)':   round(n_out / len(df) * 100, 2),
        })
    return pd.DataFrame(results).sort_values('Outliers count', ascending=False)


# ─────────────────────────────────────────────────────────────
# 5. STANDARDSCALER (sans data leakage)
# ─────────────────────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   columns: list) -> tuple:
    """
    Fit StandardScaler sur X_train uniquement, transforme X_test.
    ⚠ Jamais fit sur X_test → évite le data leakage.
    Retourne (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns]  = scaler.transform(X_test[columns])
    print(f"[✓] StandardScaler appliqué sur {len(columns)} features.")
    return X_train, X_test, scaler


# ─────────────────────────────────────────────────────────────
# 6. SAUVEGARDE
# ─────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Sauvegarde un DataFrame en CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[✓] Sauvegardé : {filepath}")


def ensure_dirs(*paths) -> None:
    """Crée les répertoires s'ils n'existent pas."""
    for p in paths:
        os.makedirs(p, exist_ok=True)
