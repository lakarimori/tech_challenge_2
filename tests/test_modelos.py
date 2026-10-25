import numpy as np
import pytest

from tc2.experimentos import avaliar_no_teste
from tc2.modelos import BASELINE, ESPACOS, NOMES, construir_pipeline, criar_fitness


@pytest.mark.parametrize("nome", NOMES)
def test_baseline_e_cromossomos_aleatorios_geram_pipelines_validos(nome):
    construir_pipeline(nome, BASELINE[nome])
    rng = np.random.default_rng(0)
    for _ in range(15):
        params = {g.nome: g.sortear(rng) for g in ESPACOS[nome]}
        pipe = construir_pipeline(nome, params)
        assert pipe.steps[0][0] == "scaler"  # pré-processamento dentro do Pipeline (sem vazamento)


@pytest.mark.parametrize("nome", NOMES)
def test_baseline_cobre_exatamente_os_genes_do_espaco(nome):
    assert set(BASELINE[nome]) == {g.nome for g in ESPACOS[nome]}


def test_fitness_fica_entre_0_e_1(dados):
    f = criar_fitness("Regressao Logistica", dados["X_tr"], dados["y_tr"], dados["cv"])
    v = f(BASELINE["Regressao Logistica"])
    assert 0.9 < v <= 1.0


def test_modelo_desconhecido_falha_claramente():
    with pytest.raises(ValueError):
        construir_pipeline("SVM", {})


def test_avaliacao_no_teste_devolve_metricas_coerentes(dados):
    m, pipe = avaliar_no_teste("KNN", BASELINE["KNN"], dados["X_tr"], dados["y_tr"],
                               dados["X_te"], dados["y_te"])
    n_malignos = int(dados["y_te"].sum())
    assert 0 <= m["FN"] <= n_malignos
    assert m["Recall"] == pytest.approx((n_malignos - m["FN"]) / n_malignos)
    assert m["Accuracy"] > 0.9


FASE1 = {  # accuracy / recall / F1 no teste, conforme o README da Fase 1
    "Regressao Logistica": (0.9649, 0.9286, 0.9512),
    "Random Forest": (0.9737, 0.9286, 0.9630),
    "KNN": (0.9561, 0.9048, 0.9383),
}


@pytest.mark.parametrize("nome", NOMES)
def test_baseline_reproduz_os_numeros_da_fase_1(dados, nome):
    m, _ = avaliar_no_teste(nome, BASELINE[nome], dados["X_tr"], dados["y_tr"], dados["X_te"], dados["y_te"])
    acc, rec, f1 = FASE1[nome]
    assert (round(m["Accuracy"], 4), round(m["Recall"], 4), round(m["F1"], 4)) == (acc, rec, f1)
