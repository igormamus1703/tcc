"""
Ponto de entrada principal do projeto.

Uso:
    python -m src.data.pipeline              # pipeline completo
    python -m src.data.pipeline --dataset cic # só CIC-IDS2017
    python -m src.data.pipeline --sample 0.1  # usar 10% dos dados (para testes rápidos)
"""

from src.data.pipeline import main

if __name__ == "__main__":
    main()
