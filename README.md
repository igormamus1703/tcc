# Triagem Explicada de Incidentes de Rede com LLM e RAG Local

**Projeto Transformador I** — Bacharelado em Ciência da Computação, PUCPR  
Turma 7B — Grupo 3

## Visão Geral

Sistema local de triagem explicada de incidentes de rede, integrando um LLM
especializado em cibersegurança (Foundation-Sec-8B) com RAG local para produzir
classificações e explicações fundamentadas em MITRE ATT&CK e Sigma Rules.

## Estrutura do Projeto

```
├── data/
│   ├── raw/              # Datasets originais (não versionados)
│   │   ├── cic-ids2017/  # CSVs do CIC-IDS2017
│   │   └── unsw-nb15/    # CSVs do UNSW-NB15
│   ├── processed/        # Dados pré-processados (Parquet)
│   └── rag/              # Base vetorial (ChromaDB) — futuro
├── src/
│   ├── data/             # Ingestão e pré-processamento
│   │   ├── loader.py     # Carregamento dos CSVs brutos
│   │   ├── preprocessor.py # Limpeza, normalização, encoding
│   │   └── pipeline.py   # Orquestrador do pipeline
│   ├── config.py         # Caminhos e parâmetros centralizados
│   ├── utils/logger.py   # Logger padronizado
│   ├── rag/              # Módulo RAG — futuro
│   ├── llm/              # Integração com LLM — futuro
│   ├── app/              # Interface Streamlit — futuro
│   └── evaluation/       # Métricas e avaliação — futuro
├── tests/                # Testes automatizados
├── notebooks/            # Análise exploratória
├── docs/                 # Documentação
└── outputs/              # Resultados e figuras
```

## Setup

```bash
python -m venv .venv
source .\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Obter os Datasets

1. **CIC-IDS2017**: Baixar de 
   - Colocar os CSVs em `data/raw/cic-ids2017/`

2. **UNSW-NB15**: Baixar de https://research.unsw.edu.au/projects/unsw-nb15-dataset
   - Colocar `UNSW_NB15_training-set.csv` e `UNSW_NB15_testing-set.csv` em `data/raw/unsw-nb15/`

## Executar o Pipeline de Pré-processamento

```bash
# Pipeline completo (ambos os datasets)
python -m src.data.pipeline

# Apenas um dataset
python -m src.data.pipeline --dataset cic
python -m src.data.pipeline --dataset unsw

# Teste rápido com 10% dos dados
python -m src.data.pipeline --sample 0.1

# Sem remoção de features correlacionadas
python -m src.data.pipeline --skip-corr
```

### Saída

O pipeline gera em `data/processed/`:
- `cic_ids2017_clean.parquet` — CIC-IDS2017 limpo
- `unsw_nb15_clean.parquet` — UNSW-NB15 limpo
- `unified_dataset.parquet` — Dataset unificado e normalizado
- `normalization_params.json` — Parâmetros para reproduzir a normalização
- `preprocessing_report.json` — Relatório completo do pipeline

## Testes

```bash
pytest tests/ -v
```

## Equipe

- Igor Mamus dos Santos
- Felipe Ribas Boaretto
- Leonardo dos Santos Marques
- João Vitor Manfrim
