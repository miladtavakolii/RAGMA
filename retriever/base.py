from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List


class BaseRetriever(ABC):
    '''
    Abstract interface for all retrievers.

    A retriever is responsible for converting a user query
    into a set of relevant documents, independent of
    the underlying vector database.
    '''

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        '''
        Retrieve top-k relevant documents for a given query.

        Parameters
        ----------
        query : str
            Input query text.

        k : int
            Number of documents to retrieve.

        Returns
        -------
        List[Document]
            Retrieved documents.
        '''
        pass
