from typing import List
from langchain_core.documents import Document
import numpy as np

from embedding.sentence_transformers_embedding import SentenceTransformersEmbedding
from vectorstore.qdrant import QdrantClientWrapper


class EmbedAndStore:
    '''
    Embeds chunked documents using a local embedding model and stores them in Qdrant.

    This class represents the final step of the ingestion pipeline in a vector-based
    retrieval system (e.g., RAG, semantic search, document QA).

    It is designed to work with:
    - LangChain `Document` objects as input
    - A local embedding model (e.g., embedding-gemma via Sentence-Transformers)
    - A custom Qdrant client wrapper for full control over storage and retrieval

    Architecture Position
    ---------------------
    This class assumes the following pipeline:

        Load Documents  ->  Chunk Documents  ->  Embed & Store

    Responsibilities
    ----------------
    - Extract text and metadata from LangChain `Document` objects
    - Generate dense vector embeddings using a local embedding model
    - Store embeddings + metadata in Qdrant using `QdrantClientWrapper`

    What this class intentionally avoids
    ------------------------------------
    - File I/O (handled by DocumentLoader)
    - Text splitting / chunking (handled by DocumentChunker)
    - Retrieval logic (handled by retriever or search layer)

    This separation keeps the ingestion pipeline modular, testable, and extensible.

    Example Usage (End-to-End)
    --------------------------
    >>> from ingestion.load_documents import DocumentLoader
    >>> from ingestion.chunking import DocumentChunker
    >>> from ingestion.embed_and_store import EmbedAndStore
    >>> from embedding.sentence_transformers_embedding import EmbeddingGemma
    >>>
    >>> # 1. Load documents
    >>> loader = DocumentLoader(data_dir='data/raw')
    >>> documents = loader.load()
    >>>
    >>> # 2. Chunk documents
    >>> chunker = DocumentChunker(chunk_size=800, chunk_overlap=150)
    >>> chunks = chunker.chunk(documents)
    >>>
    >>> # 3. Initialize embedding model
    >>> embedder = EmbeddingGemma(model_name='embedding-gemma-300m')
    >>>
    >>> # 4. Embed and store
    >>> ingest = EmbedAndStore(
    ...     embedder=embedder,
    ...     collection_name='knowledge_base'
    ... )
    >>> ingest.run(chunks)
    '''

    def __init__(
        self,
        embedder: SentenceTransformersEmbedding,
        collection_name: str,
        batch_size: int = 100,
        qdrant_host: str = 'localhost',
        qdrant_port: int = 6333,
    ):
        '''
        Initialize the EmbedAndStore component.

        Parameters
        ----------
        embedder : EmbeddingGemma
            A local embedding model wrapper responsible for converting text
            into L2-normalized dense vectors.

            Expected interface:
            - embed_texts(List[str]) -> np.ndarray

        collection_name : str
            Name of the Qdrant collection where embeddings will be stored.

        qdrant_host : str, default='localhost'
            Hostname or IP address of the Qdrant server.

        qdrant_port : int, default=6333
            Port number on which the Qdrant server is listening.
        '''
        self.embedder = embedder
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

    def run(self, documents: List[Document]) -> None:
        '''
        Embed document chunks and store them in Qdrant.

        This method performs the following steps:
        1. Extracts text (`page_content`) from each LangChain Document
        2. Extracts metadata (`metadata`) for each chunk
        3. Generates embeddings using the configured local embedding model
        4. Creates (or reuses) a Qdrant collection
        5. Upserts vectors and metadata into Qdrant

        Parameters
        ----------
        documents : List[Document]
            A list of LangChain Document objects.

            Each Document must contain:
            - page_content : str
            - metadata : dict

        Raises
        ------
        ValueError
            If the document list is empty.

        Notes
        -----
        - All embeddings are expected to be L2-normalized
        - Metadata is stored as payload in Qdrant
        - Document IDs are auto-generated per ingestion run
        '''
        if not documents:
            raise ValueError(
                'No documents provided for embedding and storage.')

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        embeddings: np.ndarray = self.embedder.embed_texts(texts)
        print('[EmbedAndStore] Generated embeddings for ', len(texts), ' texts')

        # Initialize Qdrant client wrapper
        qdrant = QdrantClientWrapper(
            host=self.qdrant_host,
            port=self.qdrant_port,
            collection_name=self.collection_name,
            vector_dim=embeddings.shape[1],
        )

        # Store vectors
        for i in range(0, len(embeddings), self.batch_size):
            qdrant.upsert_documents(
                embeddings=embeddings[i:i+self.batch_size],
                metadatas=metadatas[i:i+self.batch_size],
            )

        print(
            f'[EmbedAndStore] Successfully stored {len(texts)} chunks '
            f'in Qdrant collection "{self.collection_name}".'
        )
