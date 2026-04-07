"""
Módulo de carregamento dos datasets brutos.

Lida com as particularidades de cada dataset:
- CIC-IDS2017: múltiplos CSVs (um por dia), encoding latin-1, espaços nos nomes de colunas
- UNSW-NB15: múltiplos CSVs numerados, sem header no arquivo principal, header separado
"""

import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)


def load_cic_ids2017(data_dir: Path = None) -> pd.DataFrame:
    """
    Carrega todos os CSVs do CIC-IDS2017 e concatena em um único DataFrame.

    O CIC-IDS2017 é distribuído em múltiplos arquivos CSV, um para cada dia
    de captura (Monday, Tuesday, Wednesday, Thursday-morning, Thursday-afternoon,
    Friday-morning, Friday-afternoon). Todos compartilham o mesmo schema.

    Problemas conhecidos tratados aqui:
    - Nomes de colunas com espaços no início/fim
    - Encoding latin-1 em alguns arquivos
    - Valores infinitos em colunas numéricas
    - Linhas completamente vazias
    """
    data_dir = data_dir or config.CIC_RAW_DIR
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {data_dir}. "
            f"Baixe o CIC-IDS2017 de https://www.unb.ca/cic/datasets/ids-2017.html "
            f"e coloque os CSVs em {data_dir}"
        )

    logger.info(f"CIC-IDS2017: encontrados {len(csv_files)} arquivos CSV")

    frames = []
    for csv_file in csv_files:
        logger.info(f"  Carregando {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding="latin-1", low_memory=False)

        # Limpar espaços nos nomes de colunas (problema conhecido do CIC-IDS2017)
        df.columns = df.columns.str.strip()

        frames.append(df)
        logger.info(f"    → {len(df)} registros, {len(df.columns)} colunas")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"CIC-IDS2017 carregado: {len(combined)} registros totais, "
        f"{len(combined.columns)} colunas"
    )

    return combined


def load_unsw_nb15(data_dir: Path = None) -> pd.DataFrame:
    """
    Carrega o UNSW-NB15.

    O UNSW-NB15 pode vir em diferentes formatos:
    1. Arquivos UNSW-NB15_1.csv a UNSW-NB15_4.csv (sem header) + arquivo de features
    2. Arquivos UNSW_NB15_training-set.csv e UNSW_NB15_testing-set.csv (com header)

    Esta função tenta primeiro os arquivos com header (training/testing sets)
    e, caso não encontre, busca os arquivos numerados.
    """
    data_dir = data_dir or config.UNSW_RAW_DIR

    # Estratégia 1: training-set e testing-set (mais comuns e mais fáceis)
    train_file = _find_file(data_dir, ["UNSW_NB15_training-set.csv", "training-set.csv", "train.csv"])
    test_file = _find_file(data_dir, ["UNSW_NB15_testing-set.csv", "testing-set.csv", "test.csv"])

    if train_file and test_file:
        logger.info("UNSW-NB15: usando training-set + testing-set")
        df_train = pd.read_csv(train_file, low_memory=False)
        df_test = pd.read_csv(test_file, low_memory=False)

        # Remover coluna 'id' se existir (é apenas índice do arquivo)
        for df in [df_train, df_test]:
            if "id" in df.columns:
                df.drop(columns=["id"], inplace=True)

        combined = pd.concat([df_train, df_test], ignore_index=True)
        logger.info(
            f"UNSW-NB15 carregado: {len(combined)} registros "
            f"({len(df_train)} treino + {len(df_test)} teste), "
            f"{len(combined.columns)} colunas"
        )
        return combined

    # Estratégia 2: arquivos numerados (UNSW-NB15_1.csv a UNSW-NB15_4.csv)
    numbered_files = sorted(data_dir.glob("UNSW-NB15_[1-4].csv"))
    features_file = _find_file(data_dir, ["NUSW-NB15_features.csv", "features.csv"])

    if numbered_files:
        logger.info(f"UNSW-NB15: encontrados {len(numbered_files)} arquivos numerados")

        if features_file:
            try:
                features_df = pd.read_csv(features_file, encoding="utf-8")
            except UnicodeDecodeError:
                features_df = pd.read_csv(features_file, encoding="latin-1")
            col_names = features_df["Name"].tolist()
        else:
            logger.warning(
                "Arquivo de features não encontrado. "
                "Usando nomes de coluna genéricos."
            )
            col_names = None

        frames = []
        for f in numbered_files:
            logger.info(f"  Carregando {f.name}...")
            try:
                df = pd.read_csv(f, header=None, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(f, header=None, low_memory=False, encoding="latin-1")
            if col_names and len(df.columns) == len(col_names):
                df.columns = col_names
            frames.append(df)
            logger.info(f"    → {len(df)} registros")

        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"UNSW-NB15 carregado: {len(combined)} registros totais")
        return combined

    raise FileNotFoundError(
        f"Nenhum arquivo UNSW-NB15 encontrado em {data_dir}. "
        f"Baixe de https://research.unsw.edu.au/projects/unsw-nb15-dataset "
        f"e coloque os CSVs em {data_dir}"
    )


def _find_file(directory: Path, candidates: list[str]) -> Path | None:
    """Busca um arquivo por nome (case-insensitive) em um diretório."""
    existing = {f.name.lower(): f for f in directory.iterdir() if f.is_file()}
    for name in candidates:
        if name.lower() in existing:
            return existing[name.lower()]
    return None