"""
Parser de Sigma Rules (formato YAML).

Extrai regras de detecção do repositório SigmaHQ e converte em documentos
textuais estruturados, prontos para embedding.

Cada regra vira um chunk contendo:
- Título e descrição
- Nível de severidade (critical, high, medium, low, informational)
- Tags MITRE ATT&CK associadas
- Fonte de log (logsource)
- Lógica de detecção
- Falsos positivos conhecidos
"""

from pathlib import Path

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Diretórios dentro do repo Sigma que contêm regras relevantes
SIGMA_RULES_DIRS = [
    "rules",
    "rules-emerging-threats",
    "rules-threat-hunting",
]

# Níveis de severidade aceitos (filtrar regras muito genéricas)
ACCEPTED_LEVELS = {"critical", "high", "medium", "low", "informational"}


def parse_sigma_rules(sigma_dir: Path) -> list[dict]:
    """
    Lê todos os YAML do repositório Sigma e retorna documentos.

    Cada documento tem:
        - id: ID da regra (UUID) ou hash do título
        - title: título da regra
        - text: texto completo para embedding
        - metadata: dict com campos extras (level, tags, logsource)
        - source: "sigma_rules"

    Returns:
        Lista de dicts, um por regra.
    """
    logger.info(f"Parsing Sigma Rules de {sigma_dir}...")

    yaml_files = []
    for subdir in SIGMA_RULES_DIRS:
        rules_path = sigma_dir / subdir
        if rules_path.exists():
            yaml_files.extend(rules_path.rglob("*.yml"))

    if not yaml_files:
        # Fallback: buscar em qualquer subdiretório
        yaml_files = list(sigma_dir.rglob("*.yml"))
        # Excluir configs/tests
        yaml_files = [
            f for f in yaml_files
            if "test" not in str(f).lower()
            and "config" not in str(f).lower()
            and ".github" not in str(f)
        ]

    logger.info(f"  Encontrados {len(yaml_files)} arquivos YAML")

    documents = []
    errors = 0
    for yml_file in yaml_files:
        try:
            doc = _parse_single_rule(yml_file)
            if doc:
                documents.append(doc)
        except Exception:
            errors += 1

    if errors > 0:
        logger.warning(f"  {errors} arquivos não puderam ser parseados")

    logger.info(f"  Extraídas {len(documents)} regras válidas")
    return documents


def _parse_single_rule(yml_file: Path) -> dict | None:
    """Parseia um único arquivo YAML de regra Sigma."""

    with open(yml_file, "r", encoding="utf-8", errors="ignore") as f:
        try:
            rule = yaml.safe_load(f)
        except yaml.YAMLError:
            return None

    if not isinstance(rule, dict):
        return None

    title = rule.get("title", "")
    if not title:
        return None

    rule_id = rule.get("id", str(hash(title)))
    description = rule.get("description", "")
    level = rule.get("level", "unknown")
    status = rule.get("status", "unknown")
    tags = rule.get("tags", [])
    author = rule.get("author", "")
    false_positives = rule.get("falsepositives", [])

    # Logsource
    logsource = rule.get("logsource", {})
    logsource_parts = []
    for key in ["category", "product", "service"]:
        if key in logsource:
            logsource_parts.append(f"{key}={logsource[key]}")
    logsource_str = ", ".join(logsource_parts) if logsource_parts else "não especificado"

    # Detecção (converter dict para texto legível)
    detection = rule.get("detection", {})
    detection_str = _detection_to_text(detection)

    # Tags MITRE
    mitre_tags = [t for t in tags if isinstance(t, str) and t.startswith("attack.")]
    mitre_str = ", ".join(mitre_tags) if mitre_tags else "nenhuma"

    # ── Montar texto completo ──
    parts = [
        f"Regra Sigma: {title}",
        f"Severidade: {level}",
        f"Status: {status}",
    ]

    if description:
        parts.append(f"Descrição: {description}")

    parts.append(f"Tags MITRE ATT&CK: {mitre_str}")
    parts.append(f"Fonte de log: {logsource_str}")

    if detection_str:
        parts.append(f"Lógica de detecção: {detection_str}")

    if false_positives:
        fp_str = "; ".join(str(fp) for fp in false_positives if fp)
        if fp_str:
            parts.append(f"Falsos positivos conhecidos: {fp_str}")

    text = "\n".join(parts)

    return {
        "id": str(rule_id),
        "title": title,
        "text": text,
        "metadata": {
            "rule_id": str(rule_id),
            "level": level,
            "status": status,
            "tags": tags,
            "mitre_tags": mitre_tags,
            "logsource": logsource_str,
            "file": str(yml_file.name),
        },
        "source": "sigma_rules",
    }


def _detection_to_text(detection: dict) -> str:
    """
    Converte a lógica de detecção Sigma (dict) em texto legível.

    A detecção Sigma é um dict com:
    - Seleções nomeadas (selection, filter, etc.)
    - Uma condição que combina as seleções
    """
    if not detection:
        return ""

    parts = []

    # Condição
    condition = detection.get("condition", "")
    if condition:
        parts.append(f"Condição: {condition}")

    # Seleções
    for key, value in detection.items():
        if key == "condition":
            continue
        if isinstance(value, dict):
            fields = ", ".join(f"{k}={v}" for k, v in value.items())
            parts.append(f"  {key}: {fields}")
        elif isinstance(value, list):
            items = ", ".join(str(v) for v in value[:10])  # Limitar
            parts.append(f"  {key}: [{items}]")
        else:
            parts.append(f"  {key}: {value}")

    return "\n".join(parts)
