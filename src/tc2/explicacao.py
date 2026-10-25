"""Explicação por paciente: transforma o modelo treinado num contexto estruturado para a LLM.

Contribuições em PONTOS PERCENTUAIS de probabilidade de malignidade (valores de Shapley):
- modelos de árvore -> SHAP TreeExplainer (exato), como na Fase 1;
- demais (regressão logística, KNN) -> SHAP por permutação sobre predict_proba.

Por que não "oclusão" (trocar uma medida pela mediana)? Raio, perímetro e área são quase a mesma
grandeza (correlação > 0,95): tirar uma de cada vez não muda a previsão, e a explicação sai com
efeito ~0 para tudo. Shapley distribui o efeito entre as medidas correlacionadas.
"""
import numpy as np
import pandas as pd
import shap

from .config import LIMIAR_CLINICO, SEED

_BASE = {
    "radius": "raio do núcleo", "texture": "textura", "perimeter": "perímetro do núcleo",
    "area": "área do núcleo", "smoothness": "suavidade do contorno",
    "compactness": "compacidade", "concavity": "concavidade", "concave points": "pontos côncavos",
    "symmetry": "simetria", "fractal_dimension": "dimensão fractal",
}
_SUFIXO = {"mean": "média", "se": "erro padrão", "worst": "pior valor"}


def nome_legivel(feature: str) -> str:
    """'area_worst' -> 'área do núcleo (pior valor)'."""
    base, _, suf = feature.rpartition("_")
    return f"{_BASE.get(base, base)} ({_SUFIXO.get(suf, suf)})"


def faixa_risco(prob: float, limiar: float = LIMIAR_CLINICO) -> tuple[str, str]:
    """Mesmas faixas e condutas da Fase 1."""
    if prob >= 0.70:
        return "ALTO", "Encaminhar com PRIORIDADE ao mastologista."
    if prob >= limiar:
        return "MODERADO", "Caso duvidoso: exige revisão humana."
    return "BAIXO", "Seguir acompanhamento de rotina."


def _contribuicoes(pipe, X_treino: pd.DataFrame, linha: pd.DataFrame) -> pd.Series:
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    if hasattr(clf, "estimators_"):  # ensemble de árvores
        esc = pd.DataFrame(scaler.transform(linha), columns=linha.columns)
        sv = shap.TreeExplainer(clf).shap_values(esc)
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
        vals = sv[0, :, 1] if sv.ndim == 3 else sv[0]
        return pd.Series(vals * 100, index=linha.columns)
    cols = list(linha.columns)
    f = lambda a: pipe.predict_proba(pd.DataFrame(a, columns=cols))[:, 1]
    fundo = X_treino.sample(min(40, len(X_treino)), random_state=SEED)[cols].values
    expl = shap.PermutationExplainer(f, shap.maskers.Independent(fundo, max_samples=40), seed=SEED)
    sv = expl(linha.values, max_evals=4 * len(cols) + 1).values[0]
    return pd.Series(sv * 100, index=cols)


def montar_contexto(pipe, X_treino, y_treino, linha: pd.Series, paciente_id=None,
                    top_k: int = 5, limiar: float = LIMIAR_CLINICO, notas_clinicas: str | None = None) -> dict:
    """Tudo que a LLM pode citar sobre UMA paciente — e nada além disso."""
    X_linha = linha.to_frame().T[list(X_treino.columns)].astype(float)
    prob = float(pipe.predict_proba(X_linha)[0, 1])
    faixa, conduta = faixa_risco(prob, limiar)
    contrib = _contribuicoes(pipe, X_treino, X_linha)
    med_b, med_m = X_treino[y_treino == 0].median(), X_treino[y_treino == 1].median()

    fatores = []
    for col in contrib.abs().sort_values(ascending=False).head(top_k).index:
        fatores.append({
            "medida": nome_legivel(col), "coluna": col, "valor": round(float(linha[col]), 3),
            "mediana_benignos": round(float(med_b[col]), 3),
            "mediana_malignos": round(float(med_m[col]), 3),
            "efeito_pp": round(float(contrib[col]), 1),
            "direcao": "aumenta o risco" if contrib[col] > 0 else "reduz o risco",
        })
    ctx = {"paciente_id": paciente_id, "probabilidade_malignidade_pct": round(prob * 100, 1),
           "limiar_clinico_pct": round(limiar * 100), "faixa_risco": faixa,
           "conduta_padrao": conduta, "fatores_principais": fatores}
    if notas_clinicas:
        ctx["notas_clinicas"] = notas_clinicas
    return ctx


def resumo_lote(pipe, X, y_real, ids, limiar: float = LIMIAR_CLINICO) -> dict:
    """Estatísticas agregadas de um lote de pacientes (insumo do resumo para o gestor)."""
    prob = pipe.predict_proba(X)[:, 1]
    faixas = pd.Series([faixa_risco(p, limiar)[0] for p in prob])
    incerteza = pd.DataFrame({"id": list(ids), "prob_pct": (prob * 100).round(1)})
    incerteza["dist"] = (incerteza.prob_pct - 50).abs()
    top_incerteza = incerteza.sort_values("dist").head(3)[["id", "prob_pct"]].to_dict("records")
    y_real = np.asarray(y_real)
    pred = prob >= limiar
    return {
        "total_pacientes": int(len(prob)), "limiar_clinico_pct": round(limiar * 100),
        "pacientes_por_faixa": {k: int(v) for k, v in faixas.value_counts().items()},
        "malignos_reais": int(y_real.sum()),
        "falsos_negativos_no_limiar": int(((~pred) & (y_real == 1)).sum()),
        "falsos_positivos_no_limiar": int((pred & (y_real == 0)).sum()),
        "casos_de_maior_incerteza": top_incerteza,
    }
