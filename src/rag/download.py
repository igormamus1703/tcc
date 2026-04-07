"""
Script para baixar as fontes de conhecimento do RAG.

Fontes:
1. MITRE ATT&CK Enterprise (STIX 2.1 JSON) — taxonomia de táticas e técnicas
2. Sigma Rules (repositório YAML) — regras de detecção

Uso:
    python -m src.rag.download
    python -m src.rag.download --source mitre   # só MITRE
    python -m src.rag.download --source sigma    # só Sigma
"""

import argparse
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)

# URLs das fontes
MITRE_ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)

SIGMA_REPO_URL = "https://github.com/SigmaHQ/sigma.git"

# Diretórios de destino
RAG_SOURCES_DIR = config.DATA_DIR / "rag" / "sources"
MITRE_FILE = RAG_SOURCES_DIR / "enterprise-attack.json"
SIGMA_DIR = RAG_SOURCES_DIR / "sigma"


def download_mitre():
    """
    Baixa o MITRE ATT&CK Enterprise em formato STIX 2.1 JSON.
    
    O arquivo contém todas as táticas, técnicas, sub-técnicas,
    mitigações e relações do framework Enterprise ATT&CK.
    Tamanho: ~30MB.
    """
    RAG_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    if MITRE_FILE.exists():
        size_mb = MITRE_FILE.stat().st_size / (1024 * 1024)
        logger.info(f"MITRE ATT&CK já existe ({size_mb:.1f}MB): {MITRE_FILE}")
        logger.info("  Para re-baixar, delete o arquivo e rode novamente.")
        return

    logger.info("Baixando MITRE ATT&CK Enterprise (STIX 2.1)...")
    logger.info(f"  URL: {MITRE_ATTACK_URL}")

    try:
        urllib.request.urlretrieve(MITRE_ATTACK_URL, MITRE_FILE)
        size_mb = MITRE_FILE.stat().st_size / (1024 * 1024)
        logger.info(f"  Download concluído: {size_mb:.1f}MB → {MITRE_FILE}")

        # Validar que é JSON válido
        with open(MITRE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_objects = len(data.get("objects", []))
        logger.info(f"  Validado: {n_objects} objetos STIX")
    except Exception as e:
        if MITRE_FILE.exists():
            MITRE_FILE.unlink()
        raise RuntimeError(f"Falha ao baixar MITRE ATT&CK: {e}")


def download_sigma():
    """
    Clona o repositório Sigma Rules (shallow clone, só branch principal).
    
    O repositório contém milhares de regras de detecção em YAML.
    Usamos shallow clone para economizar espaço (~50MB vs ~500MB full).
    """
    RAG_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    if SIGMA_DIR.exists() and any(SIGMA_DIR.rglob("*.yml")):
        n_rules = len(list(SIGMA_DIR.rglob("*.yml")))
        logger.info(f"Sigma Rules já existe ({n_rules} arquivos .yml): {SIGMA_DIR}")
        logger.info("  Para re-baixar, delete a pasta e rode novamente.")
        return

    logger.info("Clonando Sigma Rules (shallow clone)...")
    logger.info(f"  URL: {SIGMA_REPO_URL}")

    # Limpar diretório se existir mas estiver vazio/corrompido
    if SIGMA_DIR.exists():
        shutil.rmtree(SIGMA_DIR)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", SIGMA_REPO_URL, str(SIGMA_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
        n_rules = len(list(SIGMA_DIR.rglob("*.yml")))
        logger.info(f"  Clone concluído: {n_rules} arquivos .yml")
    except FileNotFoundError:
        raise RuntimeError(
            "Git não encontrado. Instale o Git e tente novamente.\n"
            "  Windows: https://git-scm.com/download/win\n"
            "  Ou baixe manualmente: https://github.com/SigmaHQ/sigma/archive/refs/heads/main.zip\n"
            f"  e extraia em {SIGMA_DIR}"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Falha ao clonar Sigma Rules: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Baixar fontes de conhecimento para RAG")
    parser.add_argument(
        "--source",
        choices=["mitre", "sigma", "all"],
        default="all",
        help="Qual fonte baixar (default: all)",
    )
    args = parser.parse_args()

    if args.source in ("mitre", "all"):
        download_mitre()
    if args.source in ("sigma", "all"):
        download_sigma()

    logger.info("Downloads concluídos.")


if __name__ == "__main__":
    main()
