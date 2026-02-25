# KE-RAG: Chatbot de Lógica de Programação

Chatbot baseado em **KE-RAG (Knowledge Enhanced RAG)** para ajudar alunos iniciantes de Computação na matéria de **Lógica de Programação**. Combina busca semântica em apostilas PDF (RAG clássico) com um Knowledge Graph no Neo4j para enriquecer as respostas com relações entre conceitos.

## Arquitetura

```
Pergunta do Aluno
        │
        ▼
┌───────────────────┐
│   FastAPI (app.py)│
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Chatbot (KE-RAG) │
└────────┬──────────┘
         │
   ┌─────┴──────┐
   │            │
   ▼            ▼
┌──────┐  ┌──────────────┐
│FAISS │  │ Knowledge    │
│Index │  │ Graph (Neo4j)│
└──┬───┘  └──────┬───────┘
   │              │
   │  Chunks      │  Fatos, Pré-requisitos,
   │  relevantes  │  Próximos conceitos
   └──────┬───────┘
          │
          ▼
   ┌─────────────┐
   │ LLaMA 3.3   │  (via Groq API)
   │ 70B (Groq)  │
   └──────┬──────┘
          │
          ▼
   Resposta enriquecida
```

## Stack Tecnológica

| Componente         | Tecnologia                              |
|--------------------|-----------------------------------------|
| LLM                | LLaMA 3.3 70B via Groq API (gratuito)  |
| Framework          | LangChain                               |
| Vector Store       | FAISS                                   |
| Knowledge Graph    | Neo4j Aura Free (cloud)                 |
| PDF Parsing        | PyMuPDF (fitz)                          |
| Interface          | FastAPI (API REST)                      |
| Linguagem          | Python 3.10+                            |
| Embeddings         | sentence-transformers/all-MiniLM-L6-v2  |

## Estrutura de Pastas

```
kerag/
├── data/
│   └── apostilas/              ← coloque os PDFs aqui
├── src/
│   ├── __init__.py
│   ├── ingestion.py            ← carrega, divide e indexa os PDFs no FAISS
│   ├── knowledge_graph.py      ← constrói e consulta o grafo de conceitos
│   ├── retriever.py            ← pipeline KE-RAG (combina RAG + KG)
│   └── chatbot.py              ← lógica principal do chatbot
├── app.py                      ← API REST com FastAPI
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Configuração

### 1. Pré-requisitos

- Python 3.10 ou superior
- Conta gratuita no [Groq](https://console.groq.com) para obter a API key do LLaMA
- Conta gratuita no [Neo4j Aura](https://neo4j.com/cloud/aura) para o Knowledge Graph

### 2. Clonar e instalar dependências

```bash
git clone https://github.com/vitorialeda/kerag.git
cd kerag
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# ou: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o arquivo `.env`:

```env
# Groq API (gratuito em console.groq.com)
GROQ_API_KEY=sua_chave_aqui

# Neo4j Aura (gratuito em neo4j.com/cloud/aura)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=sua_senha_aqui
```

#### Como obter a chave Groq

1. Acesse [console.groq.com](https://console.groq.com)
2. Crie uma conta gratuita
3. Vá em **API Keys** → **Create API Key**
4. Copie a chave gerada

#### Como criar o banco Neo4j Aura

1. Acesse [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura)
2. Crie uma conta gratuita
3. Clique em **Create Free Instance**
4. Anote a URI, usuário e senha gerados

### 4. Adicionar as apostilas

Coloque os arquivos PDF das apostilas de Lógica de Programação na pasta:

```
data/apostilas/
```

Qualquer arquivo `.pdf` colocado nessa pasta será automaticamente indexado.

### 5. Construir o Knowledge Graph

```bash
curl -X POST http://localhost:8000/build-graph
```

## Rodando a API

```bash
python app.py
```

Ou com uvicorn diretamente:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível em `http://localhost:8000`.

Documentação interativa (Swagger): `http://localhost:8000/docs`

## Endpoints

### `POST /chat`

Envia uma pergunta ao chatbot.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "O que é uma variável em programação?",
    "session_id": "aluno123"
  }'
```

Resposta:
```json
{
  "answer": "Uma variável é um espaço na memória do computador onde podemos guardar informações...",
  "prerequisites": [],
  "next_concepts": ["Tipos de Dados"]
}
```

### `POST /index`

Reindexar os PDFs (use quando adicionar novas apostilas).

```bash
curl -X POST http://localhost:8000/index
```

Resposta:
```json
{
  "message": "PDFs reindexados com sucesso!"
}
```

### `POST /build-graph`

(Re)construir o Knowledge Graph no Neo4j.

```bash
curl -X POST http://localhost:8000/build-graph
```

Resposta:
```json
{
  "message": "Knowledge Graph construído com sucesso!"
}
```

### `GET /health`

Verificar se a API está funcionando.

```bash
curl http://localhost:8000/health
```

Resposta:
```json
{
  "status": "ok",
  "chatbot_ready": true,
  "knowledge_graph_ready": true
}
```

## Knowledge Graph de Lógica de Programação

O grafo de conhecimento modela os conceitos do curso e suas relações:

### Conceitos e Dificuldades

| Conceito             | Dificuldade |
|----------------------|-------------|
| Variáveis            | 1 (Básico)  |
| Tipos de Dados       | 1 (Básico)  |
| Operadores           | 1 (Básico)  |
| Entrada e Saída      | 1 (Básico)  |
| Condicionais IF/ELSE | 2 (Médio)   |
| Switch/Case          | 2 (Médio)   |
| Loop FOR             | 2 (Médio)   |
| Loop WHILE           | 2 (Médio)   |
| Loop DO-WHILE        | 2 (Médio)   |
| Funções              | 3 (Avançado)|
| Vetores/Arrays       | 3 (Avançado)|
| Matrizes             | 3 (Avançado)|
| Recursão             | 4 (Expert)  |
| Ordenação            | 4 (Expert)  |

### Tipos de Relacionamentos

- **REQUER** — pré-requisito necessário para entender o conceito
- **É_UM_TIPO_DE** — categorização/herança de conceitos
- **SIMILAR_A** — conceitos com características parecidas
- **LEVA_A** — próximo conceito sugerido na jornada de aprendizado

### Exemplo de caminho de aprendizado

```
Variáveis → Tipos de Dados → Operadores → Entrada e Saída
    → Condicionais IF/ELSE → Loop FOR → Loop WHILE
    → Loop DO-WHILE → Funções → Vetores/Arrays
    → Matrizes → Recursão → Ordenação
```

## Como Funciona o KE-RAG

1. **Pergunta do aluno** chega via API
2. **FAISS** busca os 5 trechos mais relevantes nas apostilas (busca semântica)
3. **Knowledge Graph** identifica o conceito principal e retorna:
   - Fatos relacionados ao conceito
   - Pré-requisitos necessários
   - Próximos conceitos sugeridos
4. **LLaMA 3.3 70B** (via Groq) gera uma resposta enriquecida combinando o contexto das apostilas com o conhecimento do grafo
5. **Histórico** das últimas 5 mensagens é mantido por sessão para contexto contínuo