from typing import List
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from utils.prompt_loader import load_prompt

from .base import BaseQueryDecomposer


class LLMQueryDecomposer(BaseQueryDecomposer):
    '''
    Query decomposer based on a Large Language Model (LLM).

    This decomposer uses an LLM to rewrite a complex question
    into multiple focused and atomic sub-questions suitable
    for retrieval-based systems such as RAG.
    '''

    def __init__(
        self,
        llm: BaseChatModel,
        max_subqueries: int = 5,
    ):
        '''
        Initialize the LLM-based query decomposer.

        Parameters
        ----------
        llm : BaseChatModel
            Language model used to perform query decomposition.

        max_subqueries : int, optional
            Maximum number of sub-queries to generate, by default 5.
        '''
        self.llm = llm
        self.max_subqueries = max_subqueries

    def decompose(self, query: str) -> List[str]:
        '''
        Decompose a query using the LLM.

        The LLM is prompted to return a list of sub-queries,
        one per line, without numbering or additional text.

        Parameters
        ----------
        query : str
            The original user query.

        Returns
        -------
        List[str]
            A list of sub-queries. If decomposition fails,
            the original query is returned as a single-element list.
        '''
        system_msg = SystemMessage(load_prompt('prompts/query_decomposition.txt'))
        human_msg = HumanMessage(query)

        messages = [system_msg, human_msg]
        response = self.llm.invoke(messages)

        subqueries = self._parse_response(response.text)

        return subqueries if subqueries else [query]

    def _parse_response(self, response: str) -> List[str]:
        '''
        Parse the LLM response into individual sub-queries.

        Parameters
        ----------
        response : str
            Raw text output from the LLM.

        Returns
        -------
        List[str]
            Cleaned list of sub-queries.
        '''
        lines = response.splitlines()

        return [
            line.strip()
            for line in lines
            if line.strip()
        ]
