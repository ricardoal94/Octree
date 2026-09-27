"""Trazabilidad reproducible del codigo usado en los experimentos."""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def _git(raiz: Path, *argumentos: str) -> str:
    return subprocess.run(
        ["git", *argumentos],
        cwd=raiz,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_opcional(raiz: Path, *argumentos: str) -> str | None:
    """Ejecuta Git y devuelve ``None`` cuando la consulta no aplica."""

    try:
        return _git(raiz, *argumentos)
    except (OSError, subprocess.CalledProcessError):
        return None


def _sanitizar_url_remota(url: str | None) -> str | None:
    """Elimina credenciales incrustadas sin alterar rutas locales o SSH."""

    if not url or "://" not in url:
        return url
    partes = urlsplit(url)
    netloc = partes.netloc.rsplit("@", 1)[-1]
    return urlunsplit(
        (partes.scheme, netloc, partes.path, partes.query, partes.fragment)
    )


def capturar_estado_git(raiz: Path) -> dict:
    """Captura el estado local y su relacion con las referencias remotas.

    Un arbol limpio no garantiza por si solo que el commit pueda recuperarse
    desde el repositorio compartido. Por eso tambien se registra si ``HEAD``
    esta contenido en alguna referencia remota conocida y la divergencia con
    la rama upstream configurada.
    """

    try:
        commit = _git(raiz, "rev-parse", "HEAD")
        rama = _git(raiz, "branch", "--show-current") or None
        estado = _git(raiz, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
            "git_status_inicio": None,
            "git_remote_url": None,
            "git_upstream": None,
            "git_upstream_commit": None,
            "git_ahead": None,
            "git_behind": None,
            "git_remote_refs_con_commit": None,
            "git_commit_publicado": None,
        }

    entradas = estado.splitlines() if estado else []
    upstream = _git_opcional(
        raiz, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}",
    )
    upstream_commit = (
        _git_opcional(raiz, "rev-parse", "@{upstream}") if upstream else None
    )
    remoto = upstream.split("/", 1)[0] if upstream and "/" in upstream else "origin"
    remote_url = _sanitizar_url_remota(
        _git_opcional(raiz, "remote", "get-url", remoto)
    )
    refs_salida = _git_opcional(
        raiz,
        "for-each-ref",
        "--contains",
        commit,
        "--format=%(refname:short)",
        "refs/remotes",
    )
    refs_remotas = sorted(refs_salida.splitlines()) if refs_salida else []

    ahead = behind = None
    if upstream:
        divergencia = _git_opcional(
            raiz, "rev-list", "--left-right", "--count", f"{upstream}...HEAD",
        )
        if divergencia:
            valores = divergencia.split()
            if len(valores) == 2:
                behind, ahead = (int(valor) for valor in valores)

    return {
        "git_commit": commit,
        "git_branch": rama,
        "git_dirty": bool(entradas),
        "git_status_inicio": entradas,
        "git_remote_url": remote_url,
        "git_upstream": upstream,
        "git_upstream_commit": upstream_commit,
        "git_ahead": ahead,
        "git_behind": behind,
        "git_remote_refs_con_commit": refs_remotas,
        "git_commit_publicado": bool(refs_remotas),
    }


def exigir_estado_git_limpio(estado: dict) -> None:
    """Interrumpe una corrida si no puede vincularse a un commit limpio."""

    if estado.get("git_commit") is None or estado.get("git_dirty") is None:
        raise RuntimeError(
            "No fue posible identificar el commit y el estado del repositorio."
        )
    if estado["git_dirty"]:
        detalles = "\n".join(estado.get("git_status_inicio") or [])
        raise RuntimeError(
            "El repositorio tiene cambios locales antes de la corrida. "
            "Confirme, descarte o retire esos archivos y repita la ejecucion."
            + (f"\n{detalles}" if detalles else "")
        )


def exigir_commit_git_publicado(estado: dict) -> None:
    """Interrumpe una corrida si ``HEAD`` no esta en una referencia remota."""

    if estado.get("git_commit") is None:
        raise RuntimeError(
            "No fue posible identificar el commit usado por la corrida."
        )
    if not estado.get("git_commit_publicado"):
        commit = str(estado["git_commit"])[:12]
        upstream = estado.get("git_upstream") or "sin upstream configurado"
        raise RuntimeError(
            f"El commit {commit} no aparece en ninguna referencia remota "
            f"conocida ({upstream}). Publique la rama y actualice las "
            "referencias remotas antes de repetir la corrida."
        )
