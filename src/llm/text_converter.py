"""
Conversão de registros de tráfego de rede em descrição textual.

Esse módulo é a ponte entre a Etapa 1 (dados pré-processados) e o
restante do pipeline (RAG + LLM). Ele recebe uma linha do dataset
(um fluxo de rede) e gera uma descrição em linguagem natural.

A descrição serve a dois propósitos:
1. É usada como query de busca semântica na base RAG
2. É enviada como entrada ao LLM junto com o contexto recuperado

Cada dataset (CIC-IDS2017 e UNSW-NB15) tem features diferentes,
então a função detecta a origem pelo campo 'dataset_source' e
gera uma descrição apropriada.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════════
# Mapeamentos auxiliares
# ════════════════════════════════════════════════════════════════

# Protocolos comuns por número (usado em ambos os datasets)
PROTOCOL_NAMES = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    47: "GRE",
    50: "ESP",
    51: "AH",
}


# ════════════════════════════════════════════════════════════════
# Função pública principal
# ════════════════════════════════════════════════════════════════

def record_to_text(record: pd.Series) -> str:
    """
    Converte um registro de tráfego em descrição textual.

    Args:
        record: Linha de um DataFrame (Series) com colunas do dataset.

    Returns:
        String descritiva, ex:
        "Fluxo de rede TCP com duração 12.3s, transferindo 4500 bytes
         em 230 pacotes a uma taxa de 365 bytes/s. Flags SYN(1), ACK(1)
         presentes. ..."
    """
    source = record.get("dataset_source", "")

    if source == "CIC-IDS2017":
        return _cic_record_to_text(record)
    elif source == "UNSW-NB15":
        return _unsw_record_to_text(record)
    else:
        # Fallback genérico se não houver dataset_source
        return _generic_record_to_text(record)


def records_to_text_batch(df: pd.DataFrame) -> list[str]:
    """Converte múltiplos registros em descrições, mantendo ordem."""
    return [record_to_text(row) for _, row in df.iterrows()]


# ════════════════════════════════════════════════════════════════
# Conversão CIC-IDS2017
# ════════════════════════════════════════════════════════════════

def _cic_record_to_text(rec: pd.Series) -> str:
    """
    Descrição textual para registros do CIC-IDS2017.

    Usa as features que mais carregam significado interpretável:
    duração, contagens de pacotes/bytes, flags TCP, IATs.
    """
    parts = []

    # Protocolo e duração
    proto = _proto_name(rec.get("protocol"))
    duration = rec.get("duration", 0)
    parts.append(f"Fluxo de rede {proto} com duração {_fmt_time(duration)}")

    # Volume de dados
    packets = rec.get("packets_count", 0)
    fwd_bytes = rec.get("fwd_total_payload_bytes", 0)
    if packets > 0 or fwd_bytes > 0:
        parts.append(
            f"transportando {_fmt_int(packets)} pacotes "
            f"e {_fmt_bytes(fwd_bytes)} de payload no sentido forward"
        )

    # Tamanho dos pacotes
    pkt_min = rec.get("payload_bytes_min", 0)
    pkt_max = rec.get("payload_bytes_max", 0)
    pkt_mean = rec.get("payload_bytes_mean", 0)
    if pkt_max > 0:
        parts.append(
            f"Tamanho de payload entre {_fmt_int(pkt_min)} e "
            f"{_fmt_int(pkt_max)} bytes (média {_fmt_int(pkt_mean)})"
        )

    # Taxa
    bytes_rate = rec.get("bytes_rate", 0)
    pkts_rate = rec.get("packets_rate", 0)
    if bytes_rate > 0 or pkts_rate > 0:
        parts.append(
            f"Taxa observada: {_fmt_rate(bytes_rate)} bytes/s, "
            f"{_fmt_rate(pkts_rate)} pacotes/s"
        )

    # Flags TCP (somente as que têm contagem > 0)
    flags = _format_tcp_flags_cic(rec)
    if flags:
        parts.append(f"Flags TCP observadas: {flags}")

    # Janela inicial (indicador de comportamento de conexão)
    fwd_win = rec.get("fwd_init_win_bytes", 0)
    if fwd_win > 0:
        parts.append(f"Janela TCP inicial forward: {_fmt_int(fwd_win)} bytes")

    return ". ".join(parts) + "."


def _format_tcp_flags_cic(rec: pd.Series) -> str:
    """Formata flags TCP do CIC-IDS2017 listando apenas as observadas."""
    flag_cols = {
        "SYN": "syn_flag_counts",
        "FIN": "fin_flag_counts",
        "RST": "rst_flag_counts",
        "PSH": "psh_flag_counts",
        "ACK": "ece_flag_counts",  # CIC junta ACK em ECE em algumas versões
        "CWR": "cwr_flag_counts",
    }
    observed = []
    for flag, col in flag_cols.items():
        count = rec.get(col, 0)
        if count and count > 0:
            observed.append(f"{flag}({_fmt_int(count)})")
    return ", ".join(observed)


# ════════════════════════════════════════════════════════════════
# Conversão UNSW-NB15
# ════════════════════════════════════════════════════════════════

# Mapeamento reverso de service (LabelEncoder transformou em int, mas
# os valores originais ajudam a interpretar)
UNSW_SERVICE_HINTS = {
    0: "service desconhecido",
    1: "DNS",
    2: "FTP",
    3: "FTP-data",
    4: "HTTP",
    5: "IRC",
    6: "POP3",
    7: "RADIUS",
    8: "SMTP",
    9: "SNMP",
    10: "SSH",
    11: "SSL",
    12: "outro",
}

UNSW_STATE_HINTS = {
    0: "estado desconhecido",
    1: "FIN (encerrado)",
    2: "CON (conectado)",
    3: "INT (interrompido)",
    4: "REQ (requisição)",
    5: "RST (reset)",
    6: "ACC (aceito)",
    7: "CLO (fechado)",
}


def _unsw_record_to_text(rec: pd.Series) -> str:
    """Descrição textual para registros do UNSW-NB15."""
    parts = []

    # Protocolo + serviço + estado
    proto = _proto_name(rec.get("proto"))
    service = UNSW_SERVICE_HINTS.get(int(rec.get("service", 0)), "outro")
    state = UNSW_STATE_HINTS.get(int(rec.get("state", 0)), "desconhecido")

    duration = rec.get("dur", 0)
    parts.append(
        f"Fluxo {proto} (serviço {service}, estado {state}) "
        f"com duração {_fmt_time(duration)}"
    )

    # Volume de dados
    sbytes = rec.get("sbytes", 0)
    dbytes = rec.get("dbytes", 0)
    spkts = rec.get("Spkts", 0)
    if sbytes > 0 or dbytes > 0:
        parts.append(
            f"transferindo {_fmt_bytes(sbytes)} no sentido source→dest "
            f"e {_fmt_bytes(dbytes)} no sentido dest→source "
            f"em {_fmt_int(spkts)} pacotes"
        )

    # Carga (load) e tamanho médio
    sload = rec.get("Sload", 0)
    dload = rec.get("Dload", 0)
    if sload > 0 or dload > 0:
        parts.append(
            f"Carga: {_fmt_rate(sload)} bps source, {_fmt_rate(dload)} bps dest"
        )

    smean = rec.get("smeansz", 0)
    if smean > 0:
        parts.append(f"Tamanho médio do pacote source: {_fmt_int(smean)} bytes")

    # TTL (indicador de SO/distância)
    sttl = rec.get("sttl", 0)
    dttl = rec.get("dttl", 0)
    if sttl > 0:
        parts.append(f"TTL source={int(sttl)}, dest={int(dttl)}")

    # Métricas de conexão
    ct_state = rec.get("ct_state_ttl", 0)
    if ct_state > 0:
        parts.append(f"Conexões anteriores no mesmo estado/TTL: {int(ct_state)}")

    is_sm_ports = rec.get("is_sm_ips_ports", 0)
    if is_sm_ports == 1:
        parts.append("IPs e portas de origem/destino são iguais (suspeito)")

    # Indicadores HTTP/FTP
    http_methods = rec.get("ct_flw_http_mthd", 0)
    if http_methods > 0:
        parts.append(f"Métodos HTTP no fluxo: {int(http_methods)}")

    is_ftp = rec.get("is_ftp_login", 0)
    ftp_cmd = rec.get("ct_ftp_cmd", 0)
    if is_ftp > 0 or ftp_cmd > 0:
        parts.append(f"Comandos FTP detectados: {int(ftp_cmd)} (login={int(is_ftp)})")

    return ". ".join(parts) + "."


# ════════════════════════════════════════════════════════════════
# Fallback genérico
# ════════════════════════════════════════════════════════════════

def _generic_record_to_text(rec: pd.Series) -> str:
    """
    Descrição genérica quando dataset_source não é reconhecido.
    Lista pares (campo: valor) das colunas numéricas com valores não-zero.
    """
    parts = ["Registro de tráfego de rede com características"]
    items = []
    for col, val in rec.items():
        if col in ("label", "label_original", "dataset_source"):
            continue
        if pd.isna(val) or val == 0:
            continue
        items.append(f"{col}={_fmt_value(val)}")
        if len(items) >= 15:  # limitar
            break
    parts.append(", ".join(items))
    return ": ".join(parts) + "."


# ════════════════════════════════════════════════════════════════
# Helpers de formatação
# ════════════════════════════════════════════════════════════════

def _proto_name(proto_val) -> str:
    """Converte número de protocolo para nome."""
    if proto_val is None or pd.isna(proto_val):
        return "desconhecido"
    try:
        return PROTOCOL_NAMES.get(int(proto_val), f"protocolo {int(proto_val)}")
    except (ValueError, TypeError):
        return "desconhecido"


def _fmt_time(seconds) -> str:
    """Formata duração em segundos de forma legível."""
    if pd.isna(seconds) or seconds <= 0:
        return "instantânea"
    s = float(seconds)
    if s < 0.001:
        return f"{s * 1_000_000:.0f}μs"
    if s < 1:
        return f"{s * 1000:.1f}ms"
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.1f}h"


def _fmt_bytes(b) -> str:
    """Formata bytes de forma legível (KB, MB, GB)."""
    if pd.isna(b) or b <= 0:
        return "0 bytes"
    b = float(b)
    if b < 1024:
        return f"{b:.0f} bytes"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    if b < 1024 ** 3:
        return f"{b / 1024 ** 2:.1f} MB"
    return f"{b / 1024 ** 3:.2f} GB"


def _fmt_int(n) -> str:
    """Formata número inteiro com separadores de milhar."""
    if pd.isna(n):
        return "0"
    return f"{int(n):,}".replace(",", ".")


def _fmt_rate(r) -> str:
    """Formata taxa (bytes/s ou pacotes/s)."""
    if pd.isna(r) or r <= 0:
        return "0"
    r = float(r)
    if r < 1000:
        return f"{r:.1f}"
    if r < 1_000_000:
        return f"{r / 1000:.1f}K"
    if r < 1_000_000_000:
        return f"{r / 1_000_000:.1f}M"
    return f"{r / 1_000_000_000:.2f}G"


def _fmt_value(v):
    """Formata um valor genérico para descrição."""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.3f}"
    return str(v)