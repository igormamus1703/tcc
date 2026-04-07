"""
Teste do parsing RAG sem necessidade de GPU.
Valida que as fontes foram baixadas e os parsers funcionam.

Uso: python tests/test_rag_parsing.py
"""

from src.rag.sources.mitre import parse_mitre_attack
from src.rag.sources.sigma import parse_sigma_rules
from src.rag.download import MITRE_FILE, SIGMA_DIR


def main():
    print("=" * 60)
    print("TESTE DE PARSING RAG (sem GPU)")
    print("=" * 60)

    # ── MITRE ATT&CK ──
    print("\n--- MITRE ATT&CK ---")
    if not MITRE_FILE.exists():
        print(f"ERRO: arquivo não encontrado: {MITRE_FILE}")
        print("Rode primeiro: python -m src.rag.download --source mitre")
        return

    mitre = parse_mitre_attack(MITRE_FILE)
    print(f"Técnicas extraídas: {len(mitre)}")

    if mitre:
        print(f"\nExemplo (primeira técnica):")
        print(f"  ID: {mitre[0]['id']}")
        print(f"  Título: {mitre[0]['title']}")
        print(f"  Texto (primeiros 300 chars):")
        print(f"  {mitre[0]['text'][:300]}...")

    # ── Sigma Rules ──
    print("\n\n--- SIGMA RULES ---")
    if not SIGMA_DIR.exists():
        print(f"ERRO: diretório não encontrado: {SIGMA_DIR}")
        print("Rode primeiro: python -m src.rag.download --source sigma")
        return

    sigma = parse_sigma_rules(SIGMA_DIR)
    print(f"Regras extraídas: {len(sigma)}")

    if sigma:
        print(f"\nExemplo (primeira regra):")
        print(f"  ID: {sigma[0]['id']}")
        print(f"  Título: {sigma[0]['title']}")
        print(f"  Texto (primeiros 300 chars):")
        print(f"  {sigma[0]['text'][:300]}...")

    # ── Resumo ──
    print("\n\n" + "=" * 60)
    total = len(mitre) + len(sigma)
    if total > 0:
        print(f"SUCESSO: {len(mitre)} técnicas MITRE + {len(sigma)} regras Sigma = {total} documentos")
        print("O parsing está funcionando. Pode comitar pro seu amigo rodar o pipeline completo.")
    else:
        print("FALHA: nenhum documento extraído. Verifique os downloads.")
    print("=" * 60)


if __name__ == "__main__":
    main()