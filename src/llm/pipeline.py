"""
CLI da Etapa 3: Pipeline de triagem.

Carrega registros do dataset processado, executa o pipeline completo
(descrição → RAG → LLM → triagem) e salva os resultados.

Uso:
    # Triar 10 registros aleatórios do dataset unificado
    python -m src.llm.pipeline --n 10

    # Triar registros estratificados (igual nº de cada classe)
    python -m src.llm.pipeline --n 5 --stratified

    # Triar só registros do CIC-IDS2017
    python -m src.llm.pipeline --n 10 --dataset cic

    # Triar sem RAG (baseline)
    python -m src.llm.pipeline --n 5 --no-rag

    # Triar um registro específico (índice no dataset)
    python -m src.llm.pipeline --index 12345
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.llm.triage import TriageEngine, TriageResult
from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)


def load_dataset(dataset_choice: str, sample_size: int | None) -> pd.DataFrame:
    """
    Carrega o dataset solicitado.

    Args:
        dataset_choice: "unified", "cic" ou "unsw"
        sample_size: se não-None, carrega apenas as primeiras N linhas
            (para evitar carregar 5M registros se vamos usar só 10).
    """
    paths = {
        "unified": config.UNIFIED_PROCESSED_FILE,
        "cic": config.CIC_PROCESSED_FILE,
        "unsw": config.UNSW_PROCESSED_FILE,
    }
    path = paths.get(dataset_choice)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_choice}' não encontrado em {path}. "
            f"Rode primeiro: python -m src.data.pipeline"
        )

    logger.info(f"Carregando {path.name}...")
    df = pd.read_parquet(path)
    logger.info(f"  {len(df):,} registros, {len(df.columns)} colunas")
    return df


def select_records(
    df: pd.DataFrame,
    n: int,
    stratified: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Seleciona registros para triagem.

    Args:
        df: DataFrame completo
        n: número de registros desejado
        stratified: se True, tenta pegar n/k registros de cada classe
        seed: para reprodutibilidade
    """
    if stratified and "label" in df.columns:
        per_class = max(1, n // df["label"].nunique())
        logger.info(f"Amostragem estratificada: ~{per_class} registros por classe")
        sampled = (
            df.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(min(len(x), per_class), random_state=seed))
        )
        return sampled.reset_index(drop=True)

    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def save_results(results: list[TriageResult], output_path: Path):
    """Salva resultados em arquivo JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_records": len(results),
        "n_valid": sum(1 for r in results if r.is_valid),
        "avg_elapsed_seconds": (
            sum(r.elapsed_seconds for r in results) / len(results)
            if results else 0
        ),
        "results": [r.to_dict() for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Resultados salvos em {output_path}")


def print_summary(results: list[TriageResult]):
    """Imprime resumo dos resultados no console."""
    print()
    print("=" * 70)
    print(f"RESUMO DA TRIAGEM ({len(results)} registros)")
    print("=" * 70)

    # Acertos vs ground truth
    if all(r.ground_truth is not None for r in results):
        # Comparação simples (estrita ou compatível)
        correct = sum(
            1 for r in results
            if r.attack_type and _matches_label(r.attack_type, r.ground_truth)
        )
        print(f"Acertos vs ground truth: {correct}/{len(results)} "
              f"({100 * correct / len(results):.1f}%)")

    # Distribuição de severidade
    from collections import Counter
    sev_counts = Counter(r.severity for r in results)
    print(f"Distribuição de severidade: {dict(sev_counts)}")

    # Tempo médio
    avg = sum(r.elapsed_seconds for r in results) / len(results) if results else 0
    print(f"Tempo médio por triagem: {avg:.2f}s")

    # Mostrar uma amostra
    print()
    print("--- AMOSTRAS ---")
    for i, r in enumerate(results[:3], 1):
        print(f"\n[{i}] Ground truth: {r.ground_truth}")
        print(f"    Triagem: {r.attack_type} ({r.severity}, conf={r.confidence:.2f})")
        print(f"    MITRE: {', '.join(r.mitre_techniques) or '(nenhum)'}")
        exp = r.explanation[:150] + "..." if len(r.explanation) > 150 else r.explanation
        print(f"    Explicação: {exp}")
        if r.validation_errors:
            print(f"    ⚠ Erros: {r.validation_errors}")
    print("=" * 70)


def _matches_label(predicted: str, actual: str) -> bool:
    """Comparação flexível entre rótulos (case insensitive, considera Benign≈Normal)."""
    if not predicted or not actual:
        return False
    return predicted.strip().lower() == actual.strip().lower()


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de triagem (Etapa 3): RAG + LLM"
    )
    parser.add_argument(
        "--dataset",
        choices=["unified", "cic", "unsw"],
        default="unified",
        help="Dataset a usar (default: unified)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Número de registros a triar (default: 10)",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Amostragem estratificada (tentar pegar de cada classe)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Triar um registro específico pelo índice",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Rodar SEM RAG (baseline para avaliação)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Número de docs a buscar no RAG (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Arquivo JSON de saída (default: outputs/triage_<timestamp>.json)",
    )

    args = parser.parse_args()

    # Carregar dataset
    df = load_dataset(args.dataset, sample_size=None)

    # Selecionar registros
    if args.index is not None:
        if args.index >= len(df):
            raise ValueError(f"Índice {args.index} fora do range [0, {len(df)})")
        sample = df.iloc[[args.index]].reset_index(drop=True)
    else:
        sample = select_records(df, args.n, stratified=args.stratified)

    logger.info(f"{len(sample)} registros selecionados para triagem")

    # Inicializar engine
    engine = TriageEngine(
        use_rag=not args.no_rag,
        rag_top_k=args.top_k,
    )

    # Rodar triagem em lote
    results = engine.triage_batch(sample, log_every=5)

    # Salvar resultados
    output_path = args.output
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rag_tag = "norag" if args.no_rag else "rag"
        output_path = config.OUTPUTS_DIR / f"triage_{rag_tag}_{timestamp}.json"
    else:
        output_path = Path(output_path)

    save_results(results, output_path)

    # Resumo no console
    print_summary(results)


if __name__ == "__main__":
    main()