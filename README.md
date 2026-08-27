# RAGMA 🧠🔎

**RAGMA** is a modular and extensible **Retrieval-Augmented Generation (RAG)** framework built with Python and LangChain.

The project provides an end-to-end pipeline for transforming raw documents into a searchable knowledge base and using semantic retrieval, LLM-based query decomposition, and grounded generation to answer user questions.

The architecture is intentionally kept **simple, explicit, and modular**, making it easy to experiment with different embedding models, vector stores, retrievers, query decomposition strategies, and LLMs.

---

## ✨ Overview

Traditional RAG pipelines often combine document ingestion, retrieval, prompting, and generation into a single tightly coupled chain.

RAGMA takes a more modular approach:

```text
                    ┌─────────────────────┐
                    │     Raw Documents   │
                    │     TXT / PDF       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Document Loader    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Chunking        │
                    │ Recursive Splitter  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Embeddings      │
                    │  EmbeddingGemma     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Qdrant         │
                    │   Vector Store      │
                    └──────────┬──────────┘
                               │
                               │
User Query ────────────────────┤
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Query Decomposition │
                    │        LLM          │
                    └──────────┬──────────┘
                               │
                     Sub-queries
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retrieval  │
                    │      Qdrant         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Context Construction│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM Generation    │
                    │       Ollama        │
                    └──────────┬──────────┘
                               │
                               ▼
                         Final Answer
```

---

## 🎯 Goals

RAGMA is designed around several core goals:

* Build a clean and understandable RAG architecture.
* Separate ingestion from query-time retrieval.
* Support semantic search over heterogeneous documents.
* Improve retrieval for complex questions through query decomposition.
* Keep vector-store implementations interchangeable.
* Keep retrievers independent from the underlying vector database.
* Make LLM and embedding models easy to replace.
* Provide a foundation for more advanced RAG systems and agentic architectures.

---

## 🧩 Key Features

### 📚 Document Ingestion

The ingestion pipeline currently supports:

* `.txt`
* `.pdf`

Documents are automatically loaded and converted into LangChain `Document` objects. The loader also preserves useful metadata such as topic, file type, and source path.

---

### ✂️ Intelligent Chunking

Documents are split using LangChain's `RecursiveCharacterTextSplitter`.

The splitter is configured with separators suitable for both English and Persian text, including the Persian question mark `؟`. Metadata is propagated to the resulting chunks.

Default configuration:

```text
chunk_size    = 1000
chunk_overlap = 200
```

---

### 🧠 Embedding Generation

RAGMA uses the Hugging Face embedding interface and currently initializes:

```text
google/embeddinggemma-300m
```

This embedding model converts document chunks and queries into dense vector representations suitable for semantic search.

The embedding component is deliberately abstracted through LangChain's `Embeddings` interface, making it possible to replace the model later.

---

### 🗄️ Vector Storage

The current vector-store implementation uses **Qdrant**.

RAGMA provides an adapter around Qdrant that handles:

* Collection creation
* Document upsert
* Embedding integration
* Similarity search
* Metadata filtering
* Similarity scores

The current implementation uses cosine similarity for vector search.

The vector store is exposed through an abstract interface and a factory:

```text
vectorstore/
├── base.py
├── qdrant.py
└── vector_store_factory.py
```

This allows additional vector database backends to be added without changing the rest of the RAG pipeline.

---

## 🔍 Semantic Retrieval

The retrieval layer is separated from the vector database.

```text
retriever/
├── base.py
├── vector_retriever.py
└── agents/
```

The `VectorRetriever` performs semantic similarity search through the configured vector-store abstraction.

By default, it retrieves the top `k=5` documents and can optionally apply metadata filters.

Conceptually:

```text
User Query
    │
    ▼
Embedding
    │
    ▼
Vector Search
    │
    ▼
Top-K Documents
```

---

## 🧠 LLM-Based Query Decomposition

One of the main features of RAGMA is **query decomposition**.

Complex questions can be difficult to retrieve with a single semantic-search query.

RAGMA therefore supports an LLM-based decomposer that transforms a complex question into several focused sub-queries.

For example:

```text
Original Query:

"What are the differences between machine learning,
deep learning, and reinforcement learning?"

                    │
                    ▼

             Query Decomposer
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
   What is ML?   What is DL?   What is RL?
```

The current implementation supports up to five sub-queries by default and includes graceful fallback behavior if the LLM response cannot be parsed.

The query decomposition component is located under:

```text
query_decomposition/
├── base.py
└── llm_decomposer.py
```

---

## 🤖 Retrieval-Augmented Generation

The core RAG implementation is located in:

```text
pipeline/rag_pipeline.py
```

The pipeline combines:

1. User query
2. Query decomposition
3. Semantic retrieval
4. Context construction
5. Prompt formatting
6. LLM generation

The implementation intentionally avoids hiding these stages behind a large framework abstraction, making the retrieval flow easy to inspect and modify.

The core workflow is:

```text
                 User Query
                     │
                     ▼
            ┌─────────────────┐
            │ Query Decomposer│
            └────────┬────────┘
                     │
              Multiple Queries
                     │
                     ▼
            ┌─────────────────┐
            │    Retriever    │
            └────────┬────────┘
                     │
                     ▼
            Retrieved Documents
                     │
                     ▼
            ┌─────────────────┐
            │ Context Builder │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Prompt Template │
            └────────┬────────┘
                     │
                     ▼
                  LLM
                     │
                     ▼
               Final Answer
```

---

## 🦙 LLM Backend

The current example uses **Ollama** with:

```text
gemma3:12b
```

and:

```python
temperature=0
```

The model is initialized through LangChain's `ChatOllama` interface.

Because the RAG pipeline depends on LangChain's `BaseChatModel` abstraction, the LLM can be replaced with another compatible backend.

Potential backends include:

* Ollama
* OpenAI-compatible models
* Other LangChain-supported chat models

---

## 📝 Prompt Engineering

Prompts are kept outside the Python implementation.

The main RAG prompt is located at:

```text
prompts/simple_rag.txt
```

Query decomposition uses:

```text
prompts/query_decomposition.txt
```

This separation makes it possible to experiment with prompting strategies without modifying the core RAG implementation.

---

## 🏗️ Architecture

RAGMA separates the system into independent layers:

```text
RAGMA
│
├── ingestion
│   ├── Document Loading
│   ├── Document Chunking
│   └── Embedding & Storage
│
├── vectorstore
│   ├── Base Interface
│   ├── Qdrant Adapter
│   └── Vector Store Factory
│
├── retriever
│   ├── Base Retriever
│   ├── Vector Retriever
│   └── Agents
│
├── query_decomposition
│   ├── Base Interface
│   └── LLM Decomposer
│
├── pipeline
│   ├── Ingestion Pipeline
│   └── RAG Pipeline
│
├── prompts
│   ├── RAG Prompt
│   └── Query Decomposition Prompt
│
└── utils
    └── Shared Utilities
```

The repository currently follows this separation between ingestion, retrieval, vector storage, query decomposition, and orchestration.

---

## 📂 Project Structure

```text
RAGMA/
│
├── ingestion/
│   ├── chunking.py
│   ├── load_documents.py
│   └── upsert_documents.py
│
├── pipeline/
│   ├── ingest_pipeline.py
│   └── rag_pipeline.py
│
├── prompts/
│   ├── query_decomposition.txt
│   └── simple_rag.txt
│
├── query_decomposition/
│   ├── base.py
│   └── llm_decomposer.py
│
├── retriever/
│   ├── agents/
│   ├── base.py
│   └── vector_retriever.py
│
├── vectorstore/
│   ├── base.py
│   ├── qdrant.py
│   └── vector_store_factory.py
│
├── utils/
│   └── ...
│
├── data/
│   └── raw/
│       ├── topic_1/
│       ├── topic_2/
│       └── ...
│
├── main.py
├── store.py
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🛠️ Tech Stack

| Component           | Technology                    |
| ------------------- | ----------------------------- |
| Language            | Python 3.12+                  |
| RAG Framework       | LangChain                     |
| Embeddings          | Hugging Face / EmbeddingGemma |
| Vector Database     | Qdrant                        |
| LLM                 | Ollama / Gemma                |
| Deep Learning       | PyTorch                       |
| Document Processing | PyPDF                         |
| Package Management  | uv                            |
| Containerization    | Docker Compose                |

The current `pyproject.toml` requires Python `>=3.12` and includes LangChain, Qdrant Client, Transformers, PyTorch, PyPDF, Sentence Transformers, LangChain integrations for Hugging Face, Qdrant, OpenAI and Ollama.

---

# 🚀 Installation

## Requirements

Before running RAGMA, make sure you have:

* Python 3.12+
* `uv`
* Docker
* Docker Compose
* Ollama
* A compatible local LLM
* GPU recommended for local LLM inference

---

## 1. Clone the Repository

```bash
git clone https://github.com/miladtavakolii/RAGMA.git
cd RAGMA
```

---

## 2. Install Dependencies

Using `uv`:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

The project is configured for Python 3.12 or newer.

---

## 3. Start Qdrant

The repository includes a Docker Compose configuration for the supporting services.

Start the containers:

```bash
docker compose up -d
```

The default Qdrant configuration used by the application connects to:

```text
localhost:6333
```

The Qdrant adapter creates the configured collection automatically when it does not already exist.

---

## 4. Install and Run Ollama

Install Ollama according to your operating system, then start the service.

Pull the default model:

```bash
ollama pull gemma3:12b
```

Verify that the model is available:

```bash
ollama list
```

The current `main.py` uses `gemma3:12b` as the generation model.

---

# 📚 Preparing Documents

RAGMA expects documents to be organized under:

```text
data/raw/
```

Topics are represented by directories.

For example:

```text
data/raw/
├── machine_learning/
│   ├── machine_learning.txt
│   └── ml_guide.pdf
│
├── deep_learning/
│   ├── neural_networks.pdf
│   └── deep_learning.txt
│
└── nlp/
    └── transformers.pdf
```

The directory name is automatically stored as the document's `topic` metadata. The loader also stores the source path and file type.

---

# 🗂️ Build the Knowledge Base

The ingestion process can be started with:

```bash
python store.py
```

The current implementation uses:

```text
Embedding model:
google/embeddinggemma-300m

Vector store:
Qdrant

Collection:
knowledge_base

Chunk size:
1000

Chunk overlap:
200
```

The ingestion pipeline performs:

```text
Documents
    │
    ▼
Load
    │
    ▼
Chunk
    │
    ▼
Embed
    │
    ▼
Store in Qdrant
```

This workflow is implemented by `IngestPipeline`.

---

# 💬 Running RAG

Once the knowledge base has been created, run:

```bash
python main.py
```

The current example creates the RAG pipeline and asks:

```text
تفاوت یادگیری ماشین و یادگیری عمیق چیست؟
```

The generated answer is then printed to the terminal.

---

## 🔎 Example Query Flow

For a question such as:

```text
تفاوت یادگیری ماشین و یادگیری عمیق چیست و هر کدام چه کاربردهایی دارند؟
```

RAGMA can first decompose the question into smaller retrieval queries:

```text
1. یادگیری ماشین چیست؟
2. یادگیری عمیق چیست؟
3. تفاوت یادگیری ماشین و یادگیری عمیق چیست؟
4. کاربردهای یادگیری ماشین چیست؟
5. کاربردهای یادگیری عمیق چیست؟
```

Each sub-query is independently retrieved from Qdrant, and the resulting documents are combined into the final context before generation.

---

# 🧪 Customizing the RAG Pipeline

RAGMA is designed to make individual components replaceable.

### Change the embedding model

In `main.py`:

```python
embedder = HuggingFaceEmbeddings(
    model_name="your-embedding-model"
)
```

Make sure the embedding dimension matches the Qdrant collection configuration.

---

### Change the LLM

Replace:

```python
llm = ChatOllama(
    model="gemma3:12b",
    temperature=0,
)
```

with another LangChain-compatible chat model.

---

### Change retrieval depth

The default retriever returns five documents:

```python
retriever.retrieve(query, k=5)
```

You can modify this according to the size and quality of your knowledge base.

---

### Change chunking parameters

The ingestion pipeline accepts:

```python
chunk_size=1000
chunk_overlap=200
```

For example:

```python
pipeline = IngestPipeline(
    data_dir="data/raw",
    collection_name="knowledge_base",
    embedder=embedder,
    vectorstore_backend="qdrant",
    chunk_size=800,
    chunk_overlap=150,
)
```

---

# 🧩 Metadata Filtering

Qdrant retrieval supports metadata-based filters in addition to semantic similarity.

For example:

```python
filters = {
    "must": [
        ("topic", "machine_learning"),
    ],
}
```

This can be used to restrict retrieval to a specific topic.

The current vector-store adapter supports:

* `must`
* `should`
* `must_not`

conditions.

This provides a foundation for more advanced routing and domain-specific retrieval.

---

# 🧠 Design Principles

RAGMA follows several architectural principles.

### Separation of Concerns

Each component has a specific responsibility:

```text
Ingestion
    ↓
Vector Storage
    ↓
Retrieval
    ↓
Query Decomposition
    ↓
Generation
```

---

### Interface-Based Components

Core components use abstract interfaces so implementations can be replaced without rewriting the entire pipeline.

For example:

```text
BaseVectorStore
       │
       └── QdrantVectorStoreAdapter
```

and:

```text
BaseRetriever
       │
       └── VectorRetriever
```

---

### Explicit Data Flow

The RAG pipeline intentionally exposes the major processing steps instead of hiding them behind a large chain.

This makes the system easier to:

* Debug
* Benchmark
* Extend
* Test
* Modify

The project's own RAG pipeline describes this as a deliberate design choice to reduce hidden behavior and improve extensibility.

---

# 🤝 Contributing

Contributions, experiments, and research ideas are welcome.

If you find a bug or have an idea for improving the training pipeline, feel free to open an issue or submit a pull request.

---

# 📄 License

License information will be added to the repository.

---

# 👤 Author

**Milad Tavakoli**

GitHub: [@miladtavakolii](https://github.com/miladtavakolii)

---

# ⭐ Acknowledgements

RAGMA builds upon the following open-source technologies:

* LangChain
* Qdrant
* Hugging Face
* Sentence Transformers
* EmbeddingGemma
* Ollama
* Gemma
* PyTorch

---

# 📌 Project Status

RAGMA is currently an **experimental RAG framework and research project**.

The current implementation provides the core building blocks for:

* Document ingestion
* TXT/PDF processing
* Recursive chunking
* Local embedding generation
* Qdrant vector storage
* Semantic retrieval
* Metadata filtering
* LLM-based query decomposition
* Context-grounded generation
* Local LLM inference with Ollama

The architecture is intentionally modular so that more advanced retrieval, reranking, evaluation, memory, and agentic capabilities can be added without fundamentally changing the core pipeline.
