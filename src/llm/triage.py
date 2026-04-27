"""
Orquestrador de triagem.

Este módulo conecta TODAS as etapas anteriores em um pipeline funcional:

1. Recebe um registro pré-processado (Etapa 1)
2. Converte em descrição textual (text_converter)
3. Busca contexto na base RAG (Etapa 2 - retriever)
4. Monta o prompt completo (prompts)
5. Envia ao LLM via Ollama (llm_client)
6. Parseia, valida e estrutura a resposta

A classe principal é `TriageEngine`. Use-a assim:

    engine = TriageEngine()
    triage = engine.triage_record(record)
    print(triage.attack_type, triage.severity)
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd

from src.llm.text_converter import record_to_text
from src.llm.llm_client import OllamaClient, parse_json_response
from src.llm.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    validate_triage_output,
)
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════════
# Estruturas de dados
# ════════════════════════════════════════════════════════════════

@dataclass
class TriageResult:
    """Resultado completo de uma triagem."""

    # Saída estruturada do LLM
    attack_type: str
    severity: str
    confidence: float
    mitre_techniques: list[str]
    explanation: str
    recommendations: list[str]

    # Metadados do processo
    record_description: str = ""
    retrieved_context_titles: list[str] = field(default_factory=list)
    rag_distances: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    model_name: str = ""

    # Rastreabilidade
    ground_truth: Optional[str] = None  # rótulo real (se disponível)
    raw_llm_response: str = ""
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0


# ════════════════════════════════════════════════════════════════
# Engine de triagem
# ════════════════════════════════════════════════════════════════

class TriageEngine:
    """
    Orquestrador completo do pipeline de triagem explicada.

    Reúne RAG + LLM em uma única interface de alto nível.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[OllamaClient] = None,
        rag_top_k: int = 5,
        rag_max_tokens: int = 2500,
        use_rag: bool = True,
    ):
        """
        Args:
            retriever: instância pronta. Se None, cria uma nova.
            llm_client: cliente Ollama. Se None, cria um novo com defaults.
            rag_top_k: quantos documentos buscar no RAG por query.
            rag_max_tokens: tamanho máximo do contexto RAG no prompt.
            use_rag: se False, faz triagem sem contexto (baseline para avaliação).
        """
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.rag_max_tokens = rag_max_tokens

        # Inicializar retriever (custoso: carrega modelo de embeddings + chromadb)
        if use_rag:
            self.retriever = retriever or Retriever()
        else:
            self.retriever = None
            logger.info("Modo SEM RAG (baseline)")

        # Inicializar cliente LLM
        self.llm_client = llm_client or OllamaClient()

        # Validação rápida de conectividade
        if not self.llm_client.health_check():
            logger.warning(
                f"⚠ Ollama não está respondendo em {self.llm_client.host}. "
                f"Inicie o Ollama antes de rodar triagens."
            )
        elif not self.llm_client.model_available():
            logger.warning(
                f"⚠ Modelo '{self.llm_client.model}' não disponível no Ollama. "
                f"Modelos instalados: {self.llm_client.list_models()}. "
                f"Rode: ollama pull {self.llm_client.model}"
            )

    # ────────────────────────────────────────────────────────────
    # Triagem de um único registro
    # ────────────────────────────────────────────────────────────

    def triage_record(self, record: pd.Series) -> TriageResult:
        """
        Executa o pipeline completo de triagem para um único registro.

        Args:
            record: linha de um DataFrame com features pré-processadas.

        Returns:
            TriageResult com todos os campos preenchidos.
        """
        t_start = time.time()

        # 1. Converter registro em descrição textual
        description = record_to_text(record)
        logger.debug(f"Descrição: {description[:200]}...")

        # 2. Buscar contexto na base RAG (se habilitado)
        rag_results = []
        rag_context = ""
        if self.use_rag and self.retriever is not None:
            rag_results = self.retriever.search(
                query=description, top_k=self.rag_top_k
            )
            rag_context = self.retriever.format_context(
                rag_results, max_tokens=self.rag_max_tokens
            )
            logger.debug(f"  RAG retornou {len(rag_results)} documentos")

        # 3. Montar prompts
        user_prompt = build_user_prompt(description, rag_context)

        # 4. Chamar o LLM
        raw_response = ""
        try:
            raw_response = self.llm_client.generate(
                prompt=user_prompt,
                system=SYSTEM_PROMPT,
                temperature=0.2,  # Baixa para consistência
                max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"Erro na chamada ao LLM: {e}")
            return self._build_error_result(
                description, rag_results,
                error_msg=f"LLM falhou: {e}",
                elapsed=time.time() - t_start,
                ground_truth=record.get("label"),
            )

        # 5. Parsear JSON da resposta
        parsed = parse_json_response(raw_response)
        if parsed is None:
            logger.warning("LLM não retornou JSON parseável")
            return self._build_error_result(
                description, rag_results,
                error_msg="resposta do LLM não é JSON válido",
                elapsed=time.time() - t_start,
                raw_response=raw_response,
                ground_truth=record.get("label"),
            )

        # 6. Validar contra schema
        is_valid, errors = validate_triage_output(parsed)
        if not is_valid:
            logger.warning(f"Saída inválida: {errors}")

        # 7. Montar resultado final
        return TriageResult(
            attack_type=parsed.get("attack_type", "Unknown"),
            severity=parsed.get("severity", "informational").lower(),
            confidence=float(parsed.get("confidence", 0.0)),
            mitre_techniques=list(parsed.get("mitre_techniques", [])),
            explanation=str(parsed.get("explanation", "")),
            recommendations=list(parsed.get("recommendations", [])),
            record_description=description,
            retrieved_context_titles=[r["title"] for r in rag_results],
            rag_distances=[r["distance"] for r in rag_results],
            elapsed_seconds=round(time.time() - t_start, 2),
            model_name=self.llm_client.model,
            ground_truth=record.get("label"),
            raw_llm_response=raw_response,
            validation_errors=errors,
        )

    # ────────────────────────────────────────────────────────────
    # Triagem em lote
    # ────────────────────────────────────────────────────────────

    def triage_batch(
        self, df: pd.DataFrame, log_every: int = 10
    ) -> list[TriageResult]:
        """
        Roda triagem em múltiplos registros, sequencialmente.

        Não paralelizamos porque o LLM local já satura a GPU
        com uma única inferência.
        """
        results = []
        n = len(df)
        logger.info(f"Iniciando triagem em lote: {n} registros")

        for i, (_, row) in enumerate(df.iterrows(), 1):
            result = self.triage_record(row)
            results.append(result)

            if i % log_every == 0 or i == n:
                avg_time = sum(r.elapsed_seconds for r in results) / len(results)
                logger.info(
                    f"  [{i}/{n}] Última triagem: {result.attack_type} "
                    f"(severidade={result.severity}, "
                    f"conf={result.confidence:.2f}). "
                    f"Tempo médio: {avg_time:.1f}s/registro"
                )

        return results

    # ────────────────────────────────────────────────────────────
    # Helpers internos
    # ────────────────────────────────────────────────────────────

    def _build_error_result(
        self,
        description: str,
        rag_results: list[dict],
        error_msg: str,
        elapsed: float,
        raw_response: str = "",
        ground_truth: Optional[str] = None,
    ) -> TriageResult:
        """Cria um TriageResult representando uma falha do pipeline."""
        return TriageResult(
            attack_type="Unknown",
            severity="informational",
            confidence=0.0,
            mitre_techniques=[],
            explanation=f"[FALHA] {error_msg}",
            recommendations=[],
            record_description=description,
            retrieved_context_titles=[r["title"] for r in rag_results],
            rag_distances=[r["distance"] for r in rag_results],
            elapsed_seconds=round(elapsed, 2),
            model_name=self.llm_client.model,
            ground_truth=ground_truth,
            raw_llm_response=raw_response,
            validation_errors=[error_msg],
        )