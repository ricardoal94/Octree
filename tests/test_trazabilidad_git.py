"""Pruebas de la trazabilidad previa a los experimentos."""

import subprocess
import sys
from pathlib import Path

import pytest


RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "fase3_net5"))

from trazabilidad_git import (
    _sanitizar_url_remota,
    capturar_estado_git,
    exigir_commit_git_publicado,
    exigir_estado_git_limpio,
)


def _git(raiz: Path, *argumentos: str) -> None:
    subprocess.run(["git", *argumentos], cwd=raiz, check=True,
                   capture_output=True, text=True)


def test_estado_git_distingue_repositorio_limpio_y_sucio(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "pruebas@example.com")
    _git(tmp_path, "config", "user.name", "Pruebas")
    archivo = tmp_path / "control.txt"
    archivo.write_text("versionado\n", encoding="utf-8")
    _git(tmp_path, "add", "control.txt")
    _git(tmp_path, "commit", "-m", "estado inicial")

    limpio = capturar_estado_git(tmp_path)
    assert limpio["git_commit"]
    assert limpio["git_branch"]
    assert limpio["git_dirty"] is False
    assert limpio["git_status_inicio"] == []
    exigir_estado_git_limpio(limpio)

    (tmp_path / "local.txt").write_text("no versionado\n", encoding="utf-8")
    sucio = capturar_estado_git(tmp_path)
    assert sucio["git_dirty"] is True
    assert sucio["git_status_inicio"] == ["?? local.txt"]
    with pytest.raises(RuntimeError, match="cambios locales"):
        exigir_estado_git_limpio(sucio)


def test_estado_git_distingue_commit_publicado_de_commit_local(tmp_path):
    remoto = tmp_path / "remoto.git"
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(tmp_path, "init", "--bare", str(remoto))
    _git(repo, "init")
    _git(repo, "config", "user.email", "pruebas@example.com")
    _git(repo, "config", "user.name", "Pruebas")
    archivo = repo / "control.txt"
    archivo.write_text("publicado\n", encoding="utf-8")
    _git(repo, "add", "control.txt")
    _git(repo, "commit", "-m", "commit publicado")
    _git(repo, "remote", "add", "origin", str(remoto))
    rama = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    _git(repo, "push", "-u", "origin", rama)

    publicado = capturar_estado_git(repo)
    assert publicado["git_commit_publicado"] is True
    assert publicado["git_upstream"] == f"origin/{rama}"
    assert publicado["git_ahead"] == 0
    assert publicado["git_behind"] == 0
    exigir_commit_git_publicado(publicado)

    archivo.write_text("solo local\n", encoding="utf-8")
    _git(repo, "add", "control.txt")
    _git(repo, "commit", "-m", "commit sin publicar")
    solo_local = capturar_estado_git(repo)
    assert solo_local["git_dirty"] is False
    assert solo_local["git_commit_publicado"] is False
    assert solo_local["git_ahead"] == 1
    with pytest.raises(RuntimeError, match="no aparece.*referencia remota"):
        exigir_commit_git_publicado(solo_local)

    _git(repo, "push")
    republicado = capturar_estado_git(repo)
    assert republicado["git_commit_publicado"] is True
    exigir_commit_git_publicado(republicado)


def test_url_remota_no_expone_credenciales():
    assert _sanitizar_url_remota(
        "https://usuario:secreto@example.com/grupo/repo.git"
    ) == "https://example.com/grupo/repo.git"
    assert _sanitizar_url_remota(
        "git@example.com:grupo/repo.git"
    ) == "git@example.com:grupo/repo.git"
