"""Modelos da Fase 1, seus espaços de busca (genes) e a função fitness."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PESO_F1, PESO_RECALL, SEED
from .ga import Gene

NOMES = ["Regressao Logistica", "Random Forest", "KNN"]

# Hiperparâmetros usados na Fase 1 (baseline), expressos no mesmo formato dos genes.
BASELINE = {
    "Regressao Logistica": {"C": 1.0, "penalty": "l2", "class_weight": None},
    "Random Forest": {"n_estimators": 300, "max_depth": 0, "min_samples_split": 2,
                      "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None},
    "KNN": {"n_neighbors": 5, "weights": "uniform", "p": 2},
}

# Codificação do cromossomo: um gene por hiperparâmetro.
# max_depth = 0 significa "sem limite" (None) — assim o gene continua inteiro.
ESPACOS = {
    "Regressao Logistica": [
        Gene("C", "log", 1e-3, 1e2),
        Gene("penalty", "cat", opcoes=["l1", "l2"]),
        Gene("class_weight", "cat", opcoes=[None, "balanced"]),
    ],
    "Random Forest": [
        Gene("n_estimators", "int", 50, 200),
        Gene("max_depth", "int", 0, 20),
        Gene("min_samples_split", "int", 2, 20),
        Gene("min_samples_leaf", "int", 1, 10),
        Gene("max_features", "cat", opcoes=["sqrt", "log2", 0.5, 1.0]),
        Gene("class_weight", "cat", opcoes=[None, "balanced"]),
    ],
    "KNN": [
        Gene("n_neighbors", "int", 1, 30),
        Gene("weights", "cat", opcoes=["uniform", "distance"]),
        Gene("p", "int", 1, 2),
    ],
}


def construir_pipeline(nome: str, params: dict) -> Pipeline:
    """Monta o Pipeline (StandardScaler + classificador) a partir de um dict de hiperparâmetros."""
    if nome == "Regressao Logistica":
        # l1_ratio substitui o argumento `penalty` (depreciado no sklearn >= 1.8): 1 = L1, 0 = L2.
        # L2 usa o lbfgs padrão da Fase 1 (baseline idêntico); L1 exige liblinear.
        l1 = params["penalty"] == "l1"
        clf = LogisticRegression(C=params["C"], l1_ratio=1.0 if l1 else 0.0,
                                 class_weight=params["class_weight"],
                                 solver="liblinear" if l1 else "lbfgs", max_iter=5000, random_state=SEED)
    elif nome == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if int(params["max_depth"]) == 0 else int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"], class_weight=params["class_weight"],
            random_state=SEED, n_jobs=-1)
    elif nome == "KNN":
        clf = KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]),
                                   weights=params["weights"], p=int(params["p"]))
    else:
        raise ValueError(f"Modelo desconhecido: {nome}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def criar_fitness(nome: str, X_treino, y_treino, cv):
    """Fitness = 0,7 * recall + 0,3 * F1 (médias da validação cruzada, só no TREINO).

    O conjunto de teste nunca entra aqui: o GA otimiza no treino e o teste só avalia o resultado.
    """
    def fitness(params: dict) -> float:
        pipe = construir_pipeline(nome, params)
        res = cross_validate(pipe, X_treino, y_treino, cv=cv,
                             scoring={"recall": "recall", "f1": "f1"})
        return float(PESO_RECALL * np.mean(res["test_recall"]) + PESO_F1 * np.mean(res["test_f1"]))
    return fitness
