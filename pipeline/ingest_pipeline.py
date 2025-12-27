from typing import Optional, List
from ingestion.load_documents import DocumentLoader
from ingestion.chunking import DocumentChunker
from ingestion.upsert_documents import EmbedAndStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class IngestPipeline:
    '''
    A modular, end-to-end ingestion pipeline for RAG, semantic search, or document QA projects.

    This class orchestrates the complete workflow from raw documents to vector database storage.
    It integrates:
        - Document loading (supports text files and PDFs)
        - Chunking (splitting documents into manageable segments)
        - Embedding using a local Sentence-Transformers model
        - Storage of embeddings and metadata into a Qdrant collection

    Attributes
    ----------
    data_dir : str
        Root directory containing raw documents organized by topic.
    collection_name : str
        Name of the Qdrant collection where embeddings will be stored.
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between chunks.
    batch_size : int
        Number of embeddings to upsert in a single batch.
    embedder : Embeddings
        Local embedding model for generating dense vector representations.

    Methods
    -------
    run()
        Executes the complete pipeline: load documents, chunk, embed, and store in Qdrant.
    '''

    def __init__(
        self,
        data_dir: str,
        collection_name: str,
        embedder: Embeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 100,
    ):
        '''
        Initialize the IngestPipeline with configurable parameters.

        Parameters
        ----------
        data_dir : str
            Path to the root directory containing raw documents organized by topic.
        collection_name : str
            Name of the Qdrant collection where embeddings will be stored.
        embedder : Embeddings
            Pre-initialized embedding model.
        chunk_size : int, default=800
            Maximum number of characters per document chunk.
        chunk_overlap : int, default=150
            Number of overlapping characters between consecutive chunks.
        batch_size : int, default=100
            Number of embeddings to upsert into Qdrant at once.
        '''
        self.data_dir = data_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.embedder = embedder

    def run(self) -> None:
        '''
        Execute the ingestion pipeline: load, chunk, embed, and store.

        Steps
        -----
        1. Load documents from `data_dir` using DocumentLoader.
        2. Chunk documents using DocumentChunker.
        3. Generate embeddings for each chunk with `embedder`.
        4. Store embeddings and metadata in Qdrant via EmbedAndStore.

        Raises
        ------
        ValueError
            If no documents are found in `data_dir`.

        Example
        -------
        >>> pipeline = IngestPipeline(
        ...     data_dir='data/raw',
        ...     collection_name='knowledge_base'
        ... )
        >>> pipeline.run()
        '''
        # 1. Load documents
        loader = DocumentLoader(data_dir=self.data_dir)
        documents: List[Document] = loader.load()
        if not documents:
            raise ValueError(f'No documents found in {self.data_dir}')

        print(f'[IngestPipeline] Loaded {len(documents)} documents.')

        # 2. Chunk documents
        chunker = DocumentChunker(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks: List[Document] = chunker.chunk(documents)
        print(f'[IngestPipeline] Created {len(chunks)} chunks from documents.')

        # 3. Embed and store
        embed_store = EmbedAndStore(
            embedder=self.embedder,
            collection_name=self.collection_name,
            batch_size=self.batch_size
        )
        embed_store.run(chunks)
        print(
            f'[IngestPipeline] Ingestion complete for collection "{self.collection_name}".')
