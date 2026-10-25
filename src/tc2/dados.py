"""Carga dos dados e split treino/teste — idêntico ao da Fase 1 (80/20, estratificado)."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import CSV, SEED


def carregar_dados(caminho=CSV):
    """Retorna (df, X, y). Maligno = 1 (classe positiva), benigno = 0."""
    df = pd.read_csv(caminho)
    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"].map({"M": 1, "B": 0})
    return df, X, y


def dividir(X, y):
    """Mesmo split da Fase 1: 80/20, estratificado, random_state=SEED."""
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def criar_cv(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
