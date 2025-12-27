from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from typing import List, Dict, Optional
import numpy as np


class QdrantClientWrapper:
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

    Example usage:
    --------------
    >>> from embedding import SentenceTransformersEmbedding
    >>> embedder = SentenceTransformersEmbedding()
    >>> docs = ["شبکه عصبی چیست؟", "یادگیری ماشین چگونه کار می‌کند؟"]
    >>> vectors = embedder.embed_texts(docs)
    >>> metadatas = [{"topic": "technical"}, {"topic": "technical"}]
    >>> qclient = QdrantClientWrapper(vector_dim=vectors.shape[1])
    >>> qclient.upsert_documents(embeddings=vectors, metadatas=metadatas)
    >>> query_vec = embedder.embed_query("مقدمه‌ای بر شبکه‌های عصبی")
    >>> results = qclient.search(query_vec, limit=3)
    >>> print(results)
    '''

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6333,
        collection_name: str = 'knowledge_base',
        vector_dim: Optional[int] = None
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
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE
                )
            )

    def upsert_documents(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict],
        ids: Optional[List[int]] = None
    ) -> None:
        '''
        Upsert multiple documents into the Qdrant collection with their embeddings and metadata.

        This method allows batch insertion or update of documents. Each embedding vector
        corresponds to a single document and can include associated metadata such as topic,
        source, or any other custom information. If IDs are not provided, they will be
        auto-generated as sequential integers.

        Parameters
        ----------
        embeddings : np.ndarray
            2D NumPy array of shape (num_docs, vector_dim) containing embedding vectors
            for each document.
        metadatas : List[Dict]
            List of metadata dictionaries corresponding to each embedding. Each dictionary
            can store arbitrary information about the document.
        ids : Optional[List[int]]
            Optional list of integer IDs for the documents. If None, IDs are auto-assigned.
        '''
        if ids is None:
            last_id = self.client.count('knowledge_base').count
            ids = list(range(last_id + 1, last_id + 1 + len(embeddings)))

        points = [
            PointStruct(id=ids[i], vector=embeddings[i], payload=metadatas[i])
            for i in range(len(embeddings))
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5
    ) -> List[Dict]:
        '''
        Search for the most similar embeddings to a given query vector in the collection.

        This method performs a nearest-neighbor search using cosine similarity and returns
        a list of documents that are most similar to the provided query vector. Each result
        includes the document ID, similarity score, and associated metadata.

        Parameters
        ----------
        query_vector : np.ndarray
            1D NumPy array representing the embedding vector of the query.
        limit : int
            Maximum number of results to return. Defaults to 5.

        Returns
        -------
        List[Dict]
            List of dictionaries, each containing:
            - 'id': The ID of the document in Qdrant.
            - 'score': The similarity score (cosine similarity).
            - 'payload': The metadata dictionary associated with the document.
        '''
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        ).points

        return [{'id': r.id, 'score': r.score, 'payload': r.payload} for r in results]
