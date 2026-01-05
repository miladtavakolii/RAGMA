from abc import ABC, abstractmethod
from typing import List


class BaseQueryDecomposer(ABC):
    '''
    Abstract base class for query decomposition.

    A query decomposer takes a complex user query and breaks it
    into multiple simpler sub-queries that can be independently
    retrieved from a knowledge base.

    This abstraction allows different decomposition strategies
    (LLM-based, rule-based, heuristic-based) to be used interchangeably.
    '''

    @abstractmethod
    def decompose(self, query: str) -> List[str]:
        '''
        Decompose a complex query into a list of sub-queries.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        List[str]
            A list of simpler sub-queries derived from the input query.
            The list must contain at least one query.
        '''
        raise NotImplementedError
