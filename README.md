# Triagem Explicada de Incidentes de Rede com LLM e RAG Local

**Projeto Transformador I** — Bacharelado em Ciência da Computação, PUCPR  
Turma 7B — Grupo 3

## Visão Geral

Sistema local de triagem explicada de incidentes de rede, integrando um LLM
especializado em cibersegurança (Foundation-Sec-8B) com RAG local para produzir
classificações e explicações fundamentadas em MITRE ATT&CK e Sigma Rules.

## Estrutura do Projeto

```
tcc-triagem/
├── data/
│   ├── raw/                        # Datasets originais (não versionados)
│   │   ├── cic-ids2017/            # CSVs do CIC-IDS2017
│   │   └── unsw-nb15/              # CSVs do UNSW-NB15
│   ├── processed/                  # Dados pré-processados (Parquet)
│   │   ├── cic_ids2017_clean.parquet
│   │   ├── unsw_nb15_clean.parquet
│   │   ├── unified_dataset.parquet
│   │   ├── normalization_params.json
│   │   └── preprocessing_report.json
│   └── rag/                        # Base de conhecimento RAG
│       ├── sources/                # MITRE ATT&CK JSON + Sigma Rules YAMLs
│       └── chromadb/               # Base vetorial persistente
├── src/
│   ├── config.py                   # Caminhos e parâmetros centralizados
│   ├── main.py                     # Ponto de entrada
│   ├── utils/logger.py             # Logger padronizado
│   ├── data/                       # [Etapa 1] Ingestão e pré-processamento
│   │   ├── loader.py               # Carregamento dos CSVs brutos
│   │   ├── preprocessor.py         # Limpeza, normalização, encoding
│   │   └── pipeline.py             # Orquestrador do pré-processamento
│   ├── rag/                        # [Etapa 2] RAG - Base de conhecimento
│   │   ├── download.py             # Download do MITRE ATT&CK e Sigma Rules
│   │   ├── sources/mitre.py        # Parser do MITRE ATT&CK (STIX JSON)
│   │   ├── sources/sigma.py        # Parser das Sigma Rules (YAML)
│   │   ├── embeddings.py           # Geração de embeddings (sentence-transformers)
│   │   ├── vectorstore.py          # Interface com ChromaDB
│   │   ├── retriever.py            # Busca semântica na base RAG
│   │   └── pipeline.py             # Orquestrador do pipeline RAG
│   ├── llm/                        # [Futuro] Integração com LLM
│   ├── app/                        # [Futuro] Interface Streamlit
│   └── evaluation/                 # [Futuro] Métricas e avaliação
├── tests/
│   ├── test_preprocessor.py        # Testes do pré-processamento
│   └── test_rag_parsing.py         # Testes do parsing RAG (sem GPU)
├── notebooks/                      # Análise exploratória
├── docs/                           # Documentação
├── outputs/                        # Resultados e figuras
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Inicial

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

---

## Etapa 1 — Pré-processamento dos Datasets

### 1.1 Obter os Datasets

**CIC-IDS2017:**
- Acessar https://www.unb.ca/cic/datasets/ids-2017.html
- Preencher o formulário (o link de download chega por email)
- Colocar todos os CSVs em `data/raw/cic-ids2017/`

**UNSW-NB15:**
- Acessar https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Baixar a pasta "CSV Files" (9 arquivos)
- Colocar todos os CSVs em `data/raw/unsw-nb15/`

### 1.2 Rodar o Pipeline

```bash
# Pipeline completo (ambos os datasets)
python -m src.data.pipeline

# Apenas um dataset (para testar)
python -m src.data.pipeline --dataset cic
python -m src.data.pipeline --dataset unsw

# Teste rápido com 10% dos dados
python -m src.data.pipeline --sample 0.1
```

### 1.3 Saída Esperada

Em `data/processed/`:
- `cic_ids2017_clean.parquet` — CIC-IDS2017 limpo (~2.4M registros, 65 colunas)
- `unsw_nb15_clean.parquet` — UNSW-NB15 limpo (~2.5M registros, 38 colunas)
- `unified_dataset.parquet` — Dataset unificado e normalizado (~5M registros)
- `normalization_params.json` — Parâmetros para reproduzir a normalização
- `preprocessing_report.json` — Relatório completo do pipeline

---

## Etapa 2 — Base de Conhecimento RAG

### Pré-requisitos

- **Git** instalado (para clonar o repositório Sigma Rules)
- **GPU recomendada** para gerar embeddings (funciona em CPU, mas é mais lento)
- Dependências já incluídas no `requirements.txt`:
  - `sentence-transformers` (modelo de embeddings)
  - `chromadb` (banco vetorial)
  - `pyyaml` (parser das Sigma Rules)

### 2.1 Baixar as Fontes de Conhecimento

```bash
python -m src.rag.download
```

Isso baixa:
- **MITRE ATT&CK Enterprise** (~30MB) — taxonomia de táticas e técnicas adversárias
- **Sigma Rules** (~50MB) — regras de detecção da comunidade

Os arquivos ficam em `data/rag/sources/`.

### 2.2 Testar o Parsing (sem GPU)

Para validar que as fontes foram baixadas e os parsers funcionam:

```bash
python -m tests.test_rag_parsing
```

Saída esperada: ~691 técnicas MITRE + ~3698 regras Sigma = ~4389 documentos.

### 2.3 Rodar o Pipeline RAG Completo

```bash
# Pipeline completo com testes de busca
python -m src.rag.pipeline --test

# Só indexar (sem testes)
python -m src.rag.pipeline

# Re-indexar do zero (limpa a base antes)
python -m src.rag.pipeline --reset --test

# Pular download (se as fontes já estão baixadas)
python -m src.rag.pipeline --skip-download --test
```

Na primeira execução, o modelo `all-MiniLM-L6-v2` (~80MB) será baixado automaticamente.

### 2.4 Saída Esperada

O pipeline gera em `data/rag/chromadb/` a base vetorial persistente com ~4389 documentos indexados.

Os testes de busca (`--test`) mostram resultados como:
```
Query: "SSH brute force login attempts"
  1. [mitre_attack] T1110 - Brute Force (distância: 0.35)
  2. [sigma_rules] SSH Brute Force Detection (distância: 0.42)
  3. [mitre_attack] T1021.004 - SSH (distância: 0.48)
```

Se as distâncias estiverem abaixo de ~0.7 e os resultados fizerem sentido semântico, o RAG está funcionando.

---

## Testes

```bash
# Testes do pré-processamento (usa dados sintéticos, não precisa dos CSVs)
pytest tests/test_preprocessor.py -v

# Teste do parsing RAG (precisa ter rodado o download antes)
python -m tests.test_rag_parsing
```

---

## Próximas Etapas

- **Etapa 3:** Integração do pipeline completo (classificação + retrieval + geração com LLM)
- **Etapa 4:** Interface Streamlit + avaliação + documentação final

---

## Equipe

- Igor Mamus dos Santos
- Felipe Ribas Boaretto
- Leonardo dos Santos Marques
- João Vitor Manfrim
