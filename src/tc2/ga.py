"""Algoritmo genético para otimização de hiperparâmetros (implementação própria, sem libs de GA).

Representação: o indivíduo é um dict {nome_do_gene: valor}. Cada Gene sabe sortear, mutar e
limitar o próprio valor, o que mantém os operadores genéricos para qualquer modelo.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger("tc2.ga")


@dataclass(frozen=True)
class Gene:
    """Um hiperparâmetro. tipo: 'int' | 'float' | 'log' (float em escala logarítmica) | 'cat'."""
    nome: str
    tipo: str
    baixo: float | None = None
    alto: float | None = None
    opcoes: list | None = None

    def sortear(self, rng: np.random.Generator):
        if self.tipo == "cat":
            return self.opcoes[int(rng.integers(len(self.opcoes)))]
        if self.tipo == "int":
            return int(rng.integers(self.baixo, self.alto + 1))
        if self.tipo == "log":
            return float(10 ** rng.uniform(math.log10(self.baixo), math.log10(self.alto)))
        return float(rng.uniform(self.baixo, self.alto))

    def limitar(self, valor):
        if self.tipo == "cat":
            return valor
        valor = min(max(valor, self.baixo), self.alto)
        return int(round(valor)) if self.tipo == "int" else float(valor)

    def mutar(self, valor, rng: np.random.Generator, sigma: float = 0.2):
        """Numéricos: perturbação gaussiana (em log10 para 'log'), sigma = fração da amplitude.
        Categóricos: sorteia outra opção (garante que a mutação muda de fato o valor)."""
        if self.tipo == "cat":
            outras = [o for o in self.opcoes if o != valor]
            return outras[int(rng.integers(len(outras)))] if outras else valor
        if self.tipo == "log":
            lo, hi = math.log10(self.baixo), math.log10(self.alto)
            novo = 10 ** (math.log10(valor) + rng.normal(0, sigma * (hi - lo)))
        else:
            novo = valor + rng.normal(0, sigma * (self.alto - self.baixo))
        novo = self.limitar(novo)
        if self.tipo == "int" and novo == valor:  # passo mínimo, senão a mutação seria nula
            passos = [d for d in (-1, 1) if self.baixo <= valor + d <= self.alto]  # sem sair do limite
            novo = valor + passos[int(rng.integers(len(passos)))] if passos else valor
        return novo

    def combinar(self, a, b, rng: np.random.Generator):
        """Cruzamento aritmético: ponto aleatório entre os pais (categóricos: escolhe um)."""
        if self.tipo == "cat":
            return a if rng.random() < 0.5 else b
        w = rng.random()
        if self.tipo == "log":
            return self.limitar(10 ** (w * math.log10(a) + (1 - w) * math.log10(b)))
        return self.limitar(w * a + (1 - w) * b)


@dataclass
class ConfigGA:
    nome: str = "padrao"
    tamanho_populacao: int = 20
    geracoes: int = 8
    taxa_cruzamento: float = 0.8
    taxa_mutacao: float = 0.2          # probabilidade de mutar CADA gene
    tamanho_torneio: int = 3
    elitismo: int = 1                  # nº de melhores copiados direto para a próxima geração
    cruzamento: str = "uniforme"       # 'uniforme' | 'aritmetico'
    seed: int = 42


@dataclass
class ResultadoGA:
    config: ConfigGA
    melhor_params: dict
    melhor_fitness: float
    historico: list = field(default_factory=list)   # 1 dict por geração
    avaliacoes: int = 0                              # fitness realmente calculados (sem cache)
    tempo_s: float = 0.0

    def salvar_json(self, caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, default=str)


def _chave(ind: dict):
    return tuple(sorted((k, str(v)) for k, v in ind.items()))


class AlgoritmoGenetico:
    """GA generacional com seleção por torneio, cruzamento (uniforme/aritmético),
    mutação gene a gene e elitismo. Maximiza a função fitness recebida."""

    def __init__(self, genes: list[Gene], fitness: Callable[[dict], float], config: ConfigGA):
        self.genes, self.fitness, self.cfg = genes, fitness, config
        self.rng = np.random.default_rng(config.seed)
        self._cache: dict = {}
        self.avaliacoes = 0

    # --- operadores -------------------------------------------------------------------
    def individuo_aleatorio(self) -> dict:
        return {g.nome: g.sortear(self.rng) for g in self.genes}

    def selecao_torneio(self, populacao, notas) -> dict:
        idx = self.rng.choice(len(populacao), size=min(self.cfg.tamanho_torneio, len(populacao)),
                              replace=False)
        return populacao[int(max(idx, key=lambda i: notas[i]))]

    def cruzar(self, pai_a: dict, pai_b: dict) -> tuple[dict, dict]:
        if self.rng.random() > self.cfg.taxa_cruzamento:
            return dict(pai_a), dict(pai_b)
        f1, f2 = {}, {}
        for g in self.genes:
            a, b = pai_a[g.nome], pai_b[g.nome]
            if self.cfg.cruzamento == "aritmetico":
                f1[g.nome], f2[g.nome] = g.combinar(a, b, self.rng), g.combinar(a, b, self.rng)
            elif self.rng.random() < 0.5:   # uniforme: cada gene vem de um dos pais
                f1[g.nome], f2[g.nome] = a, b
            else:
                f1[g.nome], f2[g.nome] = b, a
        return f1, f2

    def mutar(self, ind: dict) -> dict:
        return {g.nome: (g.mutar(ind[g.nome], self.rng) if self.rng.random() < self.cfg.taxa_mutacao
                         else ind[g.nome]) for g in self.genes}

    # --- avaliação com cache ------------------------------------------------------------
    def avaliar(self, ind: dict) -> float:
        k = _chave(ind)
        if k not in self._cache:
            self._cache[k] = self.fitness(ind)
            self.avaliacoes += 1
        return self._cache[k]

    # --- laço principal -----------------------------------------------------------------
    def executar(self) -> ResultadoGA:
        c, t0 = self.cfg, time.perf_counter()
        pop = [self.individuo_aleatorio() for _ in range(c.tamanho_populacao)]
        historico, melhor, melhor_nota = [], None, -math.inf

        for ger in range(c.geracoes + 1):  # geração 0 = população inicial
            notas = [self.avaliar(ind) for ind in pop]
            i_best = int(np.argmax(notas))
            if notas[i_best] > melhor_nota:
                melhor, melhor_nota = dict(pop[i_best]), notas[i_best]
            historico.append({"geracao": ger, "melhor": float(max(notas)),
                              "media": float(np.mean(notas)), "pior": float(min(notas)),
                              "melhor_global": float(melhor_nota), "avaliacoes": self.avaliacoes,
                              "tempo_s": round(time.perf_counter() - t0, 2)})
            log.info("[%s] ger %02d | melhor %.4f | media %.4f | avaliacoes %d",
                     c.nome, ger, max(notas), np.mean(notas), self.avaliacoes)
            if ger == c.geracoes:
                break

            ordem = np.argsort(notas)[::-1]
            nova = [dict(pop[int(i)]) for i in ordem[:c.elitismo]]
            while len(nova) < c.tamanho_populacao:
                f1, f2 = self.cruzar(self.selecao_torneio(pop, notas), self.selecao_torneio(pop, notas))
                nova.extend([self.mutar(f1), self.mutar(f2)])
            pop = nova[:c.tamanho_populacao]

        return ResultadoGA(c, melhor, float(melhor_nota), historico, self.avaliacoes,
                           round(time.perf_counter() - t0, 2))


def busca_aleatoria(genes, fitness, n_avaliacoes: int, seed: int = 42):
    """Baseline justo: mesma quantidade de avaliações do GA, sem nenhuma evolução."""
    rng = np.random.default_rng(seed)
    melhor, nota = None, -math.inf
    for _ in range(n_avaliacoes):
        ind = {g.nome: g.sortear(rng) for g in genes}
        n = fitness(ind)
        if n > nota:
            melhor, nota = ind, n
    return melhor, float(nota)
