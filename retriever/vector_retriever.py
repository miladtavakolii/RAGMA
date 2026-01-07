from typing import Dict
from langchain_core.documents import Document
from .base import BaseRetriever
from vectorstore.base import BaseVectorStore


class VectorRetriever(BaseRetriever):
    '''
    Retriever that performs semantic search
    over a vector store.
    '''

    def __init__(self, vector_store: BaseVectorStore):
        '''
        Parameters
        ----------
        vector_store : BaseVectorStore
            Vector store used for similarity search.
        '''
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5, filters: Dict[str, list[tuple]] | None = None) -> list[Document]:
        '''
        Retrieve documents using vector similarity search.
        '''
        results = self.vector_store.search(query, limit=k, filters=filters)

        return [doc for doc, _ in results]
