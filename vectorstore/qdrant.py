from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List, Dict, Optional
from .base import BaseVectorStore


class QdrantVectorStoreAdapter(BaseVectorStore):
    '''
    A comprehensive wrapper class for QdrantClient to manage vector embeddings storage and retrieval.

    This class simplifies working with Qdrant, a vector database designed for high-performance
    similarity search over dense vector representations of data. It provides methods to:

    - Connect to a Qdrant server, either local or remote.
    - Create a collection if it does not exist, with customizable vector dimensionality and
      similarity metric (currently supports cosine similarity).
    - Upsert embeddings along with associated metadata, allowing for flexible and informative
      document storage.
    - Perform efficient similarity searches for a given query embedding and return the most
      relevant documents along with their similarity scores and metadata.

    The class is designed to integrate seamlessly with embedding models 
    , enabling end-to-end pipelines for retrieval-augmented generation (RAG),
    semantic search, recommendation systems, and other machine learning applications that require
    vector-based reasoning.

    All embeddings stored via this wrapper are expected to be normalized (unit length), which
    ensures that cosine similarity calculations in Qdrant behave as expected and provides
    consistency across different embedding sources.
    '''

    def __init__(
        self,
        embeddings: Embeddings,
        host: str = 'localhost',
        port: int = 6333,
        collection_name: str = 'knowledge_base',
        vector_dim: Optional[int] = None,
    ):
        '''
        Initialize the Qdrant client and optionally create a collection if it does not exist.

        This method connects to a Qdrant server specified by `host` and `port`. If the collection
        named `collection_name` does not exist, it will be created with the given `vector_dim`
        and cosine distance metric. This ensures that the collection is ready to store and search
        embeddings immediately.

        Parameters
        ----------
        host : str
            The hostname or IP address of the Qdrant server. Defaults to 'localhost'.
        port : int
            The port number on which the Qdrant server is listening. Defaults to 6333.
        collection_name : str
            The name of the collection to use or create in Qdrant. Defaults to 'knowledge_base'.
        vector_dim : Optional[int]
            Dimensionality of the vectors to be stored. Must be provided if creating a new
            collection. Ignored if the collection already exists.
        '''
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        # Create collection if not exists
        if collection_name not in [c.name for c in self.client.get_collections().collections]:
            if vector_dim is None:
                raise ValueError(
                    'vector_dim must be specified for new collection.')
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE
                )
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embeddings,
        )

    def upsert_documents(
        self,
        documents: list[Document],
    ) -> None:
        '''
        Upsert LangChain Document objects into Qdrant using the integrated vector store.

        This method provides a high-level, LangChain-native interface for storing documents
        in Qdrant. Instead of manually computing embeddings and passing raw vectors, it relies
        on the configured `QdrantVectorStore` and `Embeddings` implementation to:

        1. Automatically generate embeddings for each document's text (`page_content`)
        2. Store the resulting vectors in the Qdrant collection
        3. Persist document metadata as Qdrant payloads

        This approach is particularly suitable when:
        - You are fully operating inside the LangChain ecosystem
        - You want tight integration with retrievers, chains, and agents
        - You prefer declarative ingestion over low-level vector manipulation

        Parameters
        ----------
        documents : List[langchain_core.documents.Document]
            A list of LangChain `Document` objects to be stored.

            Each Document is expected to contain:
            - page_content : str
                The textual content to be embedded
            - metadata : dict
                Arbitrary key-value metadata (e.g. topic, source, filename, page_number)

        Raises
        ------
        ValueError
            If the vector store has not been initialized with an embedding model.

        Notes
        -----
        - This method requires `embeddings` to be provided when initializing
          `QdrantClientWrapper`
        - Embedding generation and normalization are handled internally by LangChain
        - Document IDs are managed internally by Qdrant unless explicitly configured
        '''
        self.vector_store.add_documents(documents=documents)

    def search(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict]:
        '''
        Perform semantic similarity search using a raw text query.

        This method enables end-to-end semantic retrieval by:
        1. Converting the input query text into an embedding vector
           using the configured LangChain `Embeddings` model
        2. Executing a similarity search against the Qdrant collection
        3. Returning the most relevant documents along with similarity scores

        Unlike `search_vector`, this method abstracts away embedding generation
        and is ideal for user-facing query workflows such as:
        - Question answering
        - Conversational agents
        - RAG pipelines
        - Multi-agent routing systems

        Parameters
        ----------
        query : str
            The input query text to search for semantically similar documents.

        limit : int, default=5
            Maximum number of results to return.

        Returns
        -------
        List[Dict]
            A list of dictionaries, each containing:
            - 'id' : Optional[str | int]
                Document identifier (if available)
            - 'score' : float
                Similarity score between the query and the document
            - 'payload' : dict
                Metadata associated with the matched document

        Notes
        -----
        - Requires `QdrantVectorStore` to be initialized
        - Uses cosine similarity under the hood
        - Scores are higher for more semantically similar documents
        '''
        results = self.vector_store.similarity_search_with_score(
            query, k=limit)

        return [{'id': doc.id, 'score': score, 'payload': doc.metadata} for doc, score in results]
