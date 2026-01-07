from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List, Dict


class BaseAgent(ABC):
    '''
    Abstract interface for all topic-specific relevance agents.

    Each agent represents a single domain or topic and is responsible
    for estimating how relevant a given question is based on retrieved
    documents.
    '''

    @abstractmethod
    def score_relevance(
        self,
        question: str,
        documents: List[Document],
    ) -> Dict[str, int]:
        '''
        Estimate the relevance of a question to the agent's topic.

        Parameters
        ----------
        question : str
            A single sub-question produced by query decomposition.
        documents : List[Document]
            Documents retrieved from the vector store that are
            specific to the agent's topic.

        Returns
        -------
        Dict[str, int]
            A dictionary containing relevance scores in the range [0, 100].
            The structure is expected to be compatible with downstream
            orchestration logic.
        '''
        pass
