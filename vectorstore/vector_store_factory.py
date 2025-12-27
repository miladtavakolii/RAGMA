from langchain_core.embeddings import Embeddings
from .base import BaseVectorStore
from .qdrant import QdrantVectorStoreAdapter
from typing import Any

class VectorStoreFactory:
    '''
    Factory class for creating vector store instances.

    This factory abstracts the creation logic for different vector
    database backends and returns a unified `BaseVectorStore` interface.

    It enables dynamic backend selection at runtime without changing
    application-level code.

    Example
    -------
    >>> vector_store = VectorStoreFactory.create(
    ...     backend='qdrant',
    ...     embeddings=embeddings,
    ...     collection_name='knowledge_base',
    ...     vector_dim=768,
    ... )
    >>> vector_store.add_documents(documents)
    >>> results = vector_store.search('شبکه عصبی چیست؟')
    '''

    @staticmethod
    def create(
        backend: str,
        embeddings: Embeddings,
        **kwargs: Any
    ) -> BaseVectorStore:
        '''
        Create and return a vector store instance for the specified backend.

        Parameters
        ----------
        backend : str
            Identifier of the vector database backend.
            Supported values include:
            - 'qdrant'
            - 'faiss'

        embeddings : langchain_core.embeddings.Embeddings
            Embedding model used to convert text into dense vectors.
            Must be compatible with LangChain's embedding interface.

        **kwargs
            Backend-specific configuration parameters.

            Examples:
            - Qdrant:
                - collection_name : str
                - vector_dim : int
                - host : str
                - port : int
            - FAISS:
                - (no required extra parameters)

        Returns
        -------
        BaseVectorStore
            An initialized vector store adapter implementing
            the `BaseVectorStore` interface.

        Raises
        ------
        ValueError
            If the specified backend is not supported.

        Notes
        -----
        - This method centralizes backend instantiation logic
        - New vector databases can be added by extending this factory
          without modifying existing application code
        '''
        if backend == 'qdrant':
            return QdrantVectorStoreAdapter(
                embeddings=embeddings,
                **kwargs
            )

        raise ValueError(f'Unsupported vector store backend: {backend}')
