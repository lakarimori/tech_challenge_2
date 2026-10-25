import pytest

from tc2.dados import carregar_dados, criar_cv, dividir


@pytest.fixture(scope="session")
def dados():
    df, X, y = carregar_dados()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    return {"df": df, "X": X, "y": y, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
            "cv": criar_cv(3)}


@pytest.fixture
def logs_tmp(tmp_path, monkeypatch):
    """Redireciona logs/llm_chamadas.jsonl para uma pasta temporária."""
    import tc2.llm as llm
    monkeypatch.setattr(llm, "LOGS", tmp_path)
    return tmp_path
