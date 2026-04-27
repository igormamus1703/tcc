"""
Templates de prompts para o LLM de triagem.

Centraliza todos os prompts usados na geração da triagem explicada,
facilitando manutenção e iteração sobre a engenharia de prompts.

A estrutura adotada segue boas práticas para LLMs de cibersegurança:
- System prompt define papel, escopo e formato esperado
- User prompt contém o registro descrito + contexto recuperado
- Saída em JSON estruturado para parsing confiável
"""

# ════════════════════════════════════════════════════════════════
# SYSTEM PROMPT - Define o papel e comportamento do modelo
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Você é um analista sênior de segurança de redes especializado em triagem de incidentes. Sua tarefa é analisar um registro de tráfego de rede e produzir uma triagem técnica e fundamentada.

REGRAS OBRIGATÓRIAS:
1. Baseie sua análise nas evidências do tráfego e no contexto fornecido (MITRE ATT&CK e Sigma Rules).
2. Se o contexto fornecido não for suficiente, indique baixa confiança em vez de inventar.
3. Cite explicitamente as técnicas MITRE (ex: T1110, T1498) que se aplicam ao caso.
4. Seja conciso e técnico. Sem floreios, sem opiniões pessoais.
5. Sua resposta DEVE ser exclusivamente um objeto JSON válido, sem texto antes ou depois.

CATEGORIAS DE ATAQUE VÁLIDAS:
Benign, DoS, DDoS, Brute Force, Botnet, Reconnaissance, Web Attack, Exploits,
Fuzzers, Backdoor, Generic, Analysis, Shellcode, Worms, Infiltration

NÍVEIS DE SEVERIDADE VÁLIDOS:
informational, low, medium, high, critical

FORMATO DE SAÍDA (JSON):
{
  "attack_type": "<categoria de ataque ou Benign>",
  "severity": "<nível de severidade>",
  "confidence": <número de 0.0 a 1.0>,
  "mitre_techniques": ["<ID da técnica>", ...],
  "explanation": "<2-4 frases técnicas explicando o porquê>",
  "recommendations": ["<ação 1>", "<ação 2>", ...]
}"""


# ════════════════════════════════════════════════════════════════
# USER PROMPT TEMPLATE - Recebe a descrição + contexto recuperado
# ════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE = """## REGISTRO DE TRÁFEGO

{record_description}

## CONTEXTO RECUPERADO (MITRE ATT&CK e Sigma Rules)

{rag_context}

## TAREFA

Analise o registro acima usando o contexto fornecido e produza a triagem em JSON conforme as regras do sistema."""


# ════════════════════════════════════════════════════════════════
# Builders
# ════════════════════════════════════════════════════════════════

def build_user_prompt(record_description: str, rag_context: str) -> str:
    """Monta o prompt do usuário com a descrição do registro e o contexto RAG."""
    return USER_PROMPT_TEMPLATE.format(
        record_description=record_description.strip(),
        rag_context=rag_context.strip() if rag_context else "(nenhum contexto recuperado)",
    )


# ════════════════════════════════════════════════════════════════
# Validação da saída do LLM
# ════════════════════════════════════════════════════════════════

VALID_ATTACK_TYPES = {
    "Benign", "DoS", "DDoS", "Brute Force", "Botnet", "Reconnaissance",
    "Web Attack", "Exploits", "Fuzzers", "Backdoor", "Generic",
    "Analysis", "Shellcode", "Worms", "Infiltration",
}

VALID_SEVERITY = {"informational", "low", "medium", "high", "critical"}


def validate_triage_output(output: dict) -> tuple[bool, list[str]]:
    """
    Valida a saída do LLM contra o schema esperado.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    if not isinstance(output, dict):
        return False, ["output não é um dicionário"]

    # attack_type
    attack = output.get("attack_type")
    if not attack:
        errors.append("campo 'attack_type' ausente")
    elif attack not in VALID_ATTACK_TYPES:
        errors.append(f"attack_type inválido: '{attack}'")

    # severity
    sev = output.get("severity")
    if not sev:
        errors.append("campo 'severity' ausente")
    elif sev.lower() not in VALID_SEVERITY:
        errors.append(f"severity inválida: '{sev}'")

    # confidence
    conf = output.get("confidence")
    if conf is None:
        errors.append("campo 'confidence' ausente")
    else:
        try:
            cf = float(conf)
            if not (0.0 <= cf <= 1.0):
                errors.append(f"confidence fora do intervalo [0,1]: {cf}")
        except (TypeError, ValueError):
            errors.append(f"confidence não é numérica: {conf}")

    # mitre_techniques
    mitre = output.get("mitre_techniques")
    if mitre is None:
        errors.append("campo 'mitre_techniques' ausente")
    elif not isinstance(mitre, list):
        errors.append("mitre_techniques deve ser uma lista")

    # explanation
    exp = output.get("explanation")
    if not exp or not isinstance(exp, str):
        errors.append("explanation ausente ou inválida")

    # recommendations
    recs = output.get("recommendations")
    if recs is None:
        errors.append("campo 'recommendations' ausente")
    elif not isinstance(recs, list):
        errors.append("recommendations deve ser uma lista")

    return len(errors) == 0, errors