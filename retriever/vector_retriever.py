from typing import List
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

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        '''
        Retrieve documents using vector similarity search.
        '''
        results = self.vector_store.search(query, limit=k)

        return [
            Document(
                page_content=r['payload'].get('text', ''),
                metadata=r['payload']
            )
            for r in results
        ]
