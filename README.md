# Triagem Explicada de Incidentes de Rede com LLM e RAG Local

**Projeto Transformador I** — Bacharelado em Ciência da Computação, PUCPR
Turma 7B — Grupo 3

## Visão Geral

Sistema local de triagem explicada de incidentes de rede, integrando um LLM
especializado em cibersegurança (Foundation-Sec-8B-Instruct) com RAG local para
produzir classificações e explicações fundamentadas em MITRE ATT&CK e Sigma Rules.

## Estrutura do Projeto

```
tcc-triagem/
├── data/
│   ├── raw/                        # Datasets originais (não versionados)
│   │   ├── cic-ids2017/
│   │   └── unsw-nb15/
│   ├── processed/                  # [Etapa 1] Dados pré-processados (Parquet)
│   └── rag/                        # [Etapa 2] Base de conhecimento RAG
│       ├── sources/                # MITRE ATT&CK JSON + Sigma Rules YAMLs
│       └── chromadb/               # Base vetorial persistente
├── src/
│   ├── config.py                   # Caminhos e parâmetros centralizados
│   ├── utils/logger.py
│   ├── data/                       # [Etapa 1] Ingestão e pré-processamento
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── pipeline.py
│   ├── rag/                        # [Etapa 2] Base RAG
│   │   ├── download.py
│   │   ├── sources/{mitre,sigma}.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   ├── retriever.py
│   │   └── pipeline.py
│   ├── llm/                        # [Etapa 3] Triagem com LLM
│   │   ├── text_converter.py       # Registro → descrição textual
│   │   ├── llm_client.py           # Cliente Ollama
│   │   ├── prompts.py              # Templates de prompts
│   │   ├── triage.py               # Orquestrador RAG + LLM
│   │   └── pipeline.py             # CLI da etapa 3
│   ├── app/                        # [Etapa 4] Interface Streamlit
│   └── evaluation/                 # [Etapa 4] Métricas e avaliação
├── tests/
├── outputs/                        # Resultados de triagens
├── requirements.txt
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

**CIC-IDS2017:** https://www.unb.ca/cic/datasets/ids-2017.html → CSVs em `data/raw/cic-ids2017/`

**UNSW-NB15:** https://research.unsw.edu.au/projects/unsw-nb15-dataset → pasta "CSV Files" em `data/raw/unsw-nb15/`

### 1.2 Rodar o Pipeline

```bash
python -m src.data.pipeline                    # ambos os datasets
python -m src.data.pipeline --dataset cic      # só CIC-IDS2017
python -m src.data.pipeline --dataset unsw     # só UNSW-NB15
python -m src.data.pipeline --sample 0.1       # teste rápido com 10%
```

### 1.3 Saída

Em `data/processed/`: arquivos Parquet (`cic_ids2017_clean`, `unsw_nb15_clean`, `unified_dataset`) + JSONs de relatório e parâmetros.

---

## Etapa 2 — Base de Conhecimento RAG

### Pré-requisitos
- **Git** instalado (para clonar Sigma Rules)
- **GPU recomendada** (funciona em CPU, mas é mais lento)
- Dependências já no `requirements.txt`: `sentence-transformers`, `chromadb`, `pyyaml`

### 2.1 Baixar as Fontes

```bash
python -m src.rag.download
```

### 2.2 Testar o Parsing (sem GPU)

```bash
python -m tests.test_rag_parsing
```

### 2.3 Rodar o Pipeline RAG Completo

```bash
python -m src.rag.pipeline --test          # com testes de busca
python -m src.rag.pipeline                 # só indexar
python -m src.rag.pipeline --reset --test  # re-indexar do zero
python -m src.rag.pipeline --skip-download --test
```

### 2.4 Saída

Em `data/rag/chromadb/`: ~4.389 documentos indexados (691 técnicas MITRE + 3.698 regras Sigma).

---

## Etapa 3 — Triagem com LLM

### Pré-requisitos
- **Ollama** instalado e rodando: https://ollama.com/download
- **Modelo Foundation-Sec-8B-Instruct** baixado no Ollama
- **Etapas 1 e 2 já concluídas** (datasets processados + base RAG indexada)
- Dependência nova: `requests` (já no `requirements.txt`)

### 3.1 Instalar e Iniciar o Modelo

```bash
# Iniciar o servidor Ollama (deve estar rodando em background)
ollama serve

# Em outro terminal, baixar o modelo
ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Instruct

# Conferir que está disponível
ollama list
```

> **Nota:** se o modelo do Hugging Face não estiver disponível diretamente no
> Ollama, vocês podem baixar o GGUF e importar manualmente, ou usar outro
> modelo de cibersegurança disponível ajustando a variável `OLLAMA_MODEL` no `.env`.

### 3.2 Configurar (opcional)

Crie um `.env` na raiz se quiser sobrescrever os defaults:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=hf.co/fdtn-ai/Foundation-Sec-8B-Instruct
```

### 3.3 Rodar Triagens

```bash
# Triar 10 registros aleatórios do dataset unificado
python -m src.llm.pipeline --n 10

# Triar 5 registros de cada classe (estratificado)
python -m src.llm.pipeline --n 5 --stratified

# Triar só registros do CIC-IDS2017
python -m src.llm.pipeline --n 10 --dataset cic

# Triar SEM RAG (baseline para avaliação)
python -m src.llm.pipeline --n 10 --no-rag

# Triar um registro específico pelo índice
python -m src.llm.pipeline --index 12345

# Salvar em arquivo específico
python -m src.llm.pipeline --n 20 --output outputs/teste1.json
```

### 3.4 Saída

Cada triagem gera um arquivo JSON em `outputs/` contendo:

```json
{
  "n_records": 10,
  "n_valid": 10,
  "avg_elapsed_seconds": 8.4,
  "results": [
    {
      "attack_type": "DDoS",
      "severity": "critical",
      "confidence": 0.92,
      "mitre_techniques": ["T1498", "T1499.002"],
      "explanation": "O fluxo apresenta...",
      "recommendations": ["Mitigar com WAF", "Habilitar rate limiting"],
      "record_description": "Fluxo TCP duração 0.5s, transferindo...",
      "retrieved_context_titles": ["T1498 - Network DoS", ...],
      "rag_distances": [0.37, 0.40, 0.47, 0.51, 0.55],
      "ground_truth": "DDoS",
      "elapsed_seconds": 7.2
    }
  ]
}
```

E um resumo no console:
```
RESUMO DA TRIAGEM (10 registros)
Acertos vs ground truth: 8/10 (80.0%)
Distribuição de severidade: {'critical': 3, 'high': 2, 'low': 5}
Tempo médio por triagem: 8.4s
```

---

## Comandos Úteis (Resumo)

```bash
# Etapa 1 — Pré-processamento
python -m src.data.pipeline

# Etapa 2 — Construir base RAG
python -m src.rag.download
python -m src.rag.pipeline --test

# Etapa 3 — Rodar triagem
ollama serve  # em outro terminal
python -m src.llm.pipeline --n 10 --stratified

# Testes
python -m tests.test_rag_parsing
pytest tests/test_preprocessor.py -v
```

---

## Próximas Etapas

- **Etapa 4 (final):** Interface Streamlit, avaliação quantitativa (acurácia, precisão, recall, F1) e qualitativa (qualidade da explicação), análise comparativa com vs sem RAG, manuscrito final.

---

## Equipe

- Igor Mamus dos Santos
- Felipe Ribas Boaretto
- Leonardo dos Santos Marques
- João Vitor Manfrim