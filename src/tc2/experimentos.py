"""Orquestra os experimentos de GA e a comparação baseline x otimizado (no teste)."""
import logging
import time

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .ga import AlgoritmoGenetico, ConfigGA, busca_aleatoria
from .modelos import BASELINE, ESPACOS, NOMES, construir_pipeline, criar_fitness

log = logging.getLogger("tc2.experimentos")

# Três configurações que variam população, taxa de mutação, elitismo e tipo de cruzamento.
CONFIGS = [
    ConfigGA(nome="A-pequena", tamanho_populacao=10, geracoes=6, taxa_mutacao=0.10,
             elitismo=1, cruzamento="uniforme"),
    ConfigGA(nome="B-media", tamanho_populacao=16, geracoes=6, taxa_mutacao=0.20,
             elitismo=1, cruzamento="uniforme"),
    ConfigGA(nome="C-exploratoria", tamanho_populacao=16, geracoes=6, taxa_mutacao=0.40,
             elitismo=2, cruzamento="aritmetico"),
]


def rodar_ga(nome_modelo, X_treino, y_treino, cv, configs=CONFIGS):
    """Roda cada configuração de GA para um modelo. Retorna {config.nome: ResultadoGA}."""
    fitness = criar_fitness(nome_modelo, X_treino, y_treino, cv)
    resultados = {}
    for cfg in configs:
        log.info("Iniciando GA | modelo=%s | config=%s", nome_modelo, cfg)
        resultados[cfg.nome] = AlgoritmoGenetico(ESPACOS[nome_modelo], fitness, cfg).run() \
            if hasattr(AlgoritmoGenetico, "run") else AlgoritmoGenetico(
                ESPACOS[nome_modelo], fitness, cfg).executar()
    return resultados


def rodar_busca_aleatoria(nome_modelo, X_treino, y_treino, cv, n_avaliacoes, seed=42):
    fitness = criar_fitness(nome_modelo, X_treino, y_treino, cv)
    t0 = time.perf_counter()
    params, nota = busca_aleatoria(ESPACOS[nome_modelo], fitness, n_avaliacoes, seed)
    return params, nota, round(time.perf_counter() - t0, 2)


def avaliar_no_teste(nome_modelo, params, X_treino, y_treino, X_teste, y_teste,
                     limiar=0.5):
    """Treina no treino inteiro e mede no teste. Retorna (dict de métricas, pipeline treinado)."""
    pipe = construir_pipeline(nome_modelo, params)
    t0 = time.perf_counter()
    pipe.fit(X_treino, y_treino)
    t_treino = time.perf_counter() - t0
    t0 = time.perf_counter()
    prob = pipe.predict_proba(X_teste)[:, 1]
    t_pred = time.perf_counter() - t0
    pred = (prob >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_teste, pred).ravel()
    return {
        "Accuracy": accuracy_score(y_teste, pred), "Precisão": precision_score(y_teste, pred),
        "Recall": recall_score(y_teste, pred), "F1": f1_score(y_teste, pred),
        "ROC-AUC": roc_auc_score(y_teste, prob), "FN": int(fn), "FP": int(fp),
        "tempo_treino_s": round(t_treino, 3), "tempo_pred_ms": round(t_pred * 1000, 1),
    }, pipe


def comparar_baseline_otimizado(otimos, X_treino, y_treino, X_teste, y_teste, cv):
    """Tabela baseline (Fase 1) x otimizado, no TESTE e com o fitness (CV no treino)."""
    linhas = []
    for nome in NOMES:
        fit = criar_fitness(nome, X_treino, y_treino, cv)
        for versao, params in (("Original (Fase 1)", BASELINE[nome]), ("Otimizado (GA)", otimos[nome])):
            m, _ = avaliar_no_teste(nome, params, X_treino, y_treino, X_teste, y_teste)
            linhas.append({"Modelo": nome, "Versão": versao, "Fitness (CV treino)": fit(params), **m})
    return pd.DataFrame(linhas)


def tabela_experimentos(resultados_por_modelo, busca_por_modelo=None):
    """Resume todos os experimentos de GA (e a busca aleatória) numa tabela."""
    linhas = []
    for modelo, res in resultados_por_modelo.items():
        for nome_cfg, r in res.items():
            c = r.config
            linhas.append({"Modelo": modelo, "Experimento": nome_cfg, "População": c.tamanho_populacao,
                           "Gerações": c.geracoes, "Mutação": c.taxa_mutacao, "Elitismo": c.elitismo,
                           "Cruzamento": c.cruzamento, "Melhor fitness": r.melhor_fitness,
                           "Avaliações": r.avaliacoes, "Tempo (s)": r.tempo_s})
        if busca_por_modelo and modelo in busca_por_modelo:
            _, nota, tempo, n = busca_por_modelo[modelo]
            linhas.append({"Modelo": modelo, "Experimento": "Busca aleatória", "População": "-",
                           "Gerações": "-", "Mutação": "-", "Elitismo": "-", "Cruzamento": "-",
                           "Melhor fitness": nota, "Avaliações": n, "Tempo (s)": tempo})
    return pd.DataFrame(linhas)
