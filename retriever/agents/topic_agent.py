from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from .base import BaseAgent
import json
import re


class TopicAgent(BaseAgent):
    '''
    LLM-based agent that evaluates how relevant a question is
    to a specific topic using retrieved documents as evidence.

    The agent uses:
    - A system prompt to define its domain expertise
    - Retrieved documents as contextual grounding
    - The LLM to produce a numeric relevance score
    '''

    def __init__(
        self,
        topic_name: str,
        llm: BaseChatModel,
        system_prompt_template: str,
    ):
        '''
        Initialize a TopicAgent.

        Parameters
        ----------
        topic_name : str
            Name of the topic this agent represents.
        llm : BaseChatModel
            Language model used to score relevance.
        system_prompt_template : str
            Template used to generate the system prompt
            that defines the agent's role and expertise.
        '''
        self.topic_name: str = topic_name
        self.llm: BaseChatModel = llm
        self.system_prompt_template: str = system_prompt_template

    def score_relevance(
        self,
        question: str,
        documents: List[Document],
    ) -> int:
        '''
        Compute the relevance score of a question to this agent's topic.

        The method:
        - Builds a context from retrieved documents
        - Formats system and user prompts
        - Invokes the LLM
        - Parses the numeric relevance score from the output

        Parameters
        ----------
        question : str
            The sub-question to be evaluated.
        documents : List[Document]
            Topic-specific documents retrieved from the vector store.

        Returns
        -------
        int
            Relevance score in the range [0, 100].
        '''
        context = '\n\n'.join(d.page_content for d in documents)

        system_prompt = self.system_prompt_template.format(
            topic=self.topic_name,
        )
        user_prompt_template = 'question:\n{question}\n\nRetrieved documents:\n{context}'
        user_prompt = user_prompt_template.format(
            question=question,
            context=context,
        )

        system_msg = SystemMessage(system_prompt)
        human_msg = HumanMessage(user_prompt)

        messages = [system_msg, human_msg]

        response = self.llm.invoke(messages)

        # Expected output (JSON):
        # { 'question': percentage }
        
        return self._parse_response(response.text)['question']

    def _parse_response(self, text: str) -> Dict[str, int]:
        '''
        Parse the LLM output into a structured relevance score.

        The LLM is expected to return a JSON-formatted string
        containing relevance values.

        Parameters
        ----------
        text : str
            Raw LLM output.

        Returns
        -------
        Dict[str, int]
            Parsed relevance scores.
        '''
        cleaned = re.sub(r"```(?:json)?", "", str(text)).strip()
        return json.loads(str(cleaned))
