"""
Retriever — interface de alto nível para busca na base RAG.

Este módulo é o que o restante do pipeline vai usar.
Recebe uma query em texto natural, busca na base vetorial
e retorna os documentos mais relevantes formatados.

Uso:
    retriever = Retriever()
    results = retriever.search("SSH brute force login attempts")
    for doc in results:
        print(doc["title"], doc["distance"])
        print(doc["text"][:200])
"""

from src.rag.embeddings import EmbeddingModel
from src.rag.vectorstore import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Interface de busca semântica na base de conhecimento.

    Combina o modelo de embeddings com o ChromaDB para
    buscar documentos relevantes dado uma query de texto.
    """

    def __init__(self, embedding_model: EmbeddingModel = None, vector_store: VectorStore = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()

        doc_count = self.vector_store.count()
        if doc_count == 0:
            logger.warning(
                "Base vetorial vazia. Rode primeiro: python -m src.rag.pipeline"
            )
        else:
            logger.info(f"Retriever pronto ({doc_count} documentos na base)")

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str = None,
    ) -> list[dict]:
        """
        Busca documentos relevantes para uma query.

        Args:
            query: texto de busca (ex: "SSH brute force detection")
            top_k: número de resultados
            source_filter: filtrar por fonte ("mitre_attack" ou "sigma_rules")

        Returns:
            Lista de dicts com: id, title, text, metadata, distance, source
        """
        query_embedding = self.embedding_model.encode_query(query)

        where = None
        if source_filter:
            where = {"source": source_filter}

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        # Enriquecer resultados com title e source
        for doc in results:
            doc["title"] = doc["metadata"].get("title", "")
            doc["source"] = doc["metadata"].get("source", "")

        return results

    def format_context(self, results: list[dict], max_tokens: int = 3000) -> str:
        """
        Formata os resultados de busca em um bloco de contexto
        para incluir no prompt do LLM.

        Estima ~4 chars por token para controlar o tamanho.
        """
        max_chars = max_tokens * 4
        context_parts = []
        total_chars = 0

        for i, doc in enumerate(results, 1):
            source_label = {
                "mitre_attack": "MITRE ATT&CK",
                "sigma_rules": "Sigma Rule",
            }.get(doc["source"], doc["source"])

            header = f"[{source_label}] {doc['title']}"
            text = doc["text"]

            # Truncar texto individual se muito longo
            available = max_chars - total_chars - len(header) - 20
            if available <= 0:
                break
            if len(text) > available:
                text = text[:available] + "..."

            block = f"--- Referência {i} ({source_label}) ---\n{text}"
            context_parts.append(block)
            total_chars += len(block)

        return "\n\n".join(context_parts)
