"""
Parser do MITRE ATT&CK (formato STIX 2.1).

Extrai técnicas e sub-técnicas do JSON STIX e converte em documentos
textuais estruturados, prontos para embedding.

Cada técnica vira um chunk contendo:
- ID e nome (ex: T1059 - Command and Scripting Interpreter)
- Táticas associadas (ex: Execution)
- Descrição completa
- Plataformas alvo
- Mitigações relacionadas
- Detecções sugeridas
"""

import json
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_mitre_attack(stix_path: Path) -> list[dict]:
    """
    Lê o JSON STIX do MITRE ATT&CK e retorna uma lista de documentos.

    Cada documento tem:
        - id: ID da técnica (ex: "T1059")
        - title: nome da técnica
        - text: texto completo para embedding
        - metadata: dict com campos extras (táticas, plataformas, url)
        - source: "mitre_attack"

    Returns:
        Lista de dicts, um por técnica/sub-técnica.
    """
    logger.info(f"Parsing MITRE ATT&CK de {stix_path}...")

    with open(stix_path, "r", encoding="utf-8") as f:
        stix_data = json.load(f)

    objects = stix_data.get("objects", [])

    # Indexar objetos por ID para resolver relações
    obj_by_id = {obj["id"]: obj for obj in objects}

    # Extrair relações (mitigações, detecções)
    relations = _extract_relations(objects)

    # Extrair técnicas (attack-pattern)
    techniques = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        doc = _build_technique_doc(obj, relations, obj_by_id)
        if doc:
            techniques.append(doc)

    logger.info(f"  Extraídas {len(techniques)} técnicas/sub-técnicas")
    return techniques


def _build_technique_doc(obj: dict, relations: dict, obj_by_id: dict) -> dict | None:
    """Converte um objeto STIX attack-pattern em documento textual."""

    # Extrair ID externo (T1059, T1059.001, etc.)
    technique_id = None
    url = None
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            technique_id = ref.get("external_id")
            url = ref.get("url")
            break

    if not technique_id:
        return None

    name = obj.get("name", "")
    description = obj.get("description", "")

    # Táticas (kill chain phases)
    tactics = []
    for phase in obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") == "mitre-attack":
            tactics.append(phase["phase_name"].replace("-", " ").title())

    # Plataformas
    platforms = obj.get("x_mitre_platforms", [])

    # Mitigações relacionadas
    mitigations = _get_related_texts(
        obj["id"], relations, obj_by_id, "mitigates"
    )

    # Detecções
    # No STIX, detecções podem estar em x_mitre_detection ou em relações
    detection = obj.get("x_mitre_detection", "")

    # ── Montar texto completo para embedding ──
    parts = [
        f"Técnica MITRE ATT&CK: {technique_id} - {name}",
    ]

    if tactics:
        parts.append(f"Táticas: {', '.join(tactics)}")

    if platforms:
        parts.append(f"Plataformas: {', '.join(platforms)}")

    if description:
        # Limpar markdown e referências do STIX
        clean_desc = _clean_stix_text(description)
        parts.append(f"Descrição: {clean_desc}")

    if mitigations:
        parts.append("Mitigações:")
        for mit_name, mit_text in mitigations:
            parts.append(f"  - {mit_name}: {_clean_stix_text(mit_text)}")

    if detection:
        parts.append(f"Detecção: {_clean_stix_text(detection)}")

    text = "\n".join(parts)

    return {
        "id": technique_id,
        "title": f"{technique_id} - {name}",
        "text": text,
        "metadata": {
            "technique_id": technique_id,
            "name": name,
            "tactics": tactics,
            "platforms": platforms,
            "url": url or "",
        },
        "source": "mitre_attack",
    }


def _extract_relations(objects: list[dict]) -> dict:
    """
    Extrai relações entre objetos STIX.

    Retorna dict: {source_id: [(relationship_type, target_id), ...]}
    """
    relations = {}
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("revoked", False):
            continue

        src = obj.get("source_ref", "")
        tgt = obj.get("target_ref", "")
        rel_type = obj.get("relationship_type", "")

        if src not in relations:
            relations[src] = []
        relations[src].append((rel_type, tgt))

        # Relação inversa para buscar "quem mitiga esta técnica"
        if tgt not in relations:
            relations[tgt] = []
        relations[tgt].append((rel_type, src))

    return relations


def _get_related_texts(
    technique_id: str,
    relations: dict,
    obj_by_id: dict,
    relation_type: str,
) -> list[tuple[str, str]]:
    """Busca textos de objetos relacionados a uma técnica."""
    results = []
    for rel_type, related_id in relations.get(technique_id, []):
        if rel_type != relation_type:
            continue
        related_obj = obj_by_id.get(related_id, {})
        name = related_obj.get("name", "")
        desc = related_obj.get("description", "")
        if name and desc:
            results.append((name, desc))
    return results[:5]  # Limitar para não explodir o chunk


def _clean_stix_text(text: str) -> str:
    """Remove formatação markdown e referências STIX do texto."""
    import re
    # Remover citações STIX: (Citation: ...)
    text = re.sub(r"\(Citation:[^)]+\)", "", text)
    # Remover links markdown
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remover tags HTML
    text = re.sub(r"<[^>]+>", "", text)
    # Normalizar whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
