import numpy as np

from tc2.ga import AlgoritmoGenetico, ConfigGA, Gene, busca_aleatoria

GENES = [Gene("x", "int", 0, 20), Gene("y", "log", 1e-3, 1e2), Gene("c", "cat", opcoes=["a", "b", "c"])]


def fitness_toy(ind):
    """Máximo em x=20, y próximo de 1, c='b'."""
    return ind["x"] / 20 - abs(np.log10(ind["y"])) / 3 + (1.0 if ind["c"] == "b" else 0.0)


def test_sorteio_respeita_limites():
    rng = np.random.default_rng(0)
    for _ in range(200):
        for g in GENES:
            v = g.sortear(rng)
            if g.tipo == "cat":
                assert v in g.opcoes
            else:
                assert g.baixo <= v <= g.alto


def test_mutacao_muda_valor_e_respeita_limites():
    rng = np.random.default_rng(1)
    gx, gc = GENES[0], GENES[2]
    for _ in range(200):
        v = gx.sortear(rng)
        m = gx.mutar(v, rng)
        assert m != v and gx.baixo <= m <= gx.alto and isinstance(m, int)
        c = gc.sortear(rng)
        assert gc.mutar(c, rng) != c


def test_cruzamento_uniforme_herda_genes_dos_pais():
    ga = AlgoritmoGenetico(GENES, fitness_toy, ConfigGA(taxa_cruzamento=1.0, seed=3))
    a, b = {"x": 1, "y": 0.01, "c": "a"}, {"x": 19, "y": 50.0, "c": "c"}
    for _ in range(50):
        f1, f2 = ga.cruzar(a, b)
        for g in GENES:
            assert {f1[g.nome], f2[g.nome]} == {a[g.nome], b[g.nome]}  # troca, não inventa


def test_cruzamento_aritmetico_fica_entre_os_pais():
    ga = AlgoritmoGenetico(GENES, fitness_toy, ConfigGA(taxa_cruzamento=1.0, cruzamento="aritmetico"))
    a, b = {"x": 2, "y": 0.01, "c": "a"}, {"x": 18, "y": 10.0, "c": "c"}
    for _ in range(50):
        f1, _ = ga.cruzar(a, b)
        assert 2 <= f1["x"] <= 18 and 0.01 <= f1["y"] <= 10.0 and f1["c"] in ("a", "c")


def test_mesma_semente_mesmo_resultado():
    def rodar():
        return AlgoritmoGenetico(GENES, fitness_toy, ConfigGA(tamanho_populacao=8, geracoes=4, seed=7)).executar()
    r1, r2 = rodar(), rodar()
    assert r1.melhor_params == r2.melhor_params
    assert [h["melhor"] for h in r1.historico] == [h["melhor"] for h in r2.historico]


def test_elitismo_nunca_piora_o_melhor():
    r = AlgoritmoGenetico(GENES, fitness_toy, ConfigGA(tamanho_populacao=10, geracoes=10, elitismo=1)).executar()
    melhores = [h["melhor"] for h in r.historico]
    assert all(b >= a - 1e-12 for a, b in zip(melhores, melhores[1:]))
    assert len(r.historico) == 11  # geração 0 + 10


def test_ga_supera_busca_aleatoria_no_mesmo_orcamento():
    r = AlgoritmoGenetico(GENES, fitness_toy, ConfigGA(tamanho_populacao=12, geracoes=10, seed=0)).executar()
    _, nota_rs = busca_aleatoria(GENES, fitness_toy, r.avaliacoes, seed=0)
    assert r.melhor_fitness >= nota_rs - 0.05


def test_cache_nao_reavalia_individuos_iguais():
    chamadas = []

    def f(ind):
        chamadas.append(1)
        return 0.5

    ga = AlgoritmoGenetico(GENES, f, ConfigGA())
    ind = {"x": 3, "y": 1.0, "c": "a"}
    ga.avaliar(ind), ga.avaliar(dict(ind))
    assert len(chamadas) == 1 and ga.avaliacoes == 1


def test_busca_aleatoria_respeita_orcamento():
    n = []
    busca_aleatoria(GENES, lambda i: n.append(1) or 0.0, 25)
    assert len(n) == 25
