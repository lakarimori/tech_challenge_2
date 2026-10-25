"""Constantes globais e configuração de logging."""
import logging
from pathlib import Path

SEED = 42
RAIZ = Path(__file__).resolve().parents[2]
CSV = RAIZ / "breast-cancer.csv"
OUT = RAIZ / "outputs"
LOGS = RAIZ / "logs"

# Limiar clínico adotado na Fase 1 (mais conservador que 0,50 por causa do falso negativo)
LIMIAR_CLINICO = 0.30

# Pesos da função fitness: recall da classe maligna é o objetivo principal (custo
# assimétrico do falso negativo); F1 entra para impedir a solução "alarmista".
PESO_RECALL = 0.7
PESO_F1 = 0.3


def configurar_logging(nome_arquivo: str = "execucao.log", nivel: int = logging.INFO) -> logging.Logger:
    """Logger do projeto: arquivo em logs/. Idempotente (não duplica handlers)."""
    LOGS.mkdir(exist_ok=True)
    logger = logging.getLogger("tc2")
    logger.setLevel(nivel)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        arquivo = logging.FileHandler(LOGS / nome_arquivo, encoding="utf-8")
        arquivo.setFormatter(fmt)
        logger.addHandler(arquivo)
    return logger
