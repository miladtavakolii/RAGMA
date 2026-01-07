from typing import Dict, List
from retriever.vector_retriever import VectorRetriever
from .base import BaseAgent


class AgentOrchestrator:
    '''
    Orchestrates multiple topic-specific agents to estimate
    how relevant each decomposed user question is to each topic.

    Responsibilities:
    - Iterate over all registered topic agents
    - Perform topic-filtered retrieval for each question
    - Delegate relevance scoring to the corresponding agent
    - Aggregate and return relevance scores per topic

    This component acts as a routing and coordination layer
    between query decomposition and downstream RAG execution.
    '''

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        retriever: VectorRetriever,
        top_k: int = 5,
    ):
        '''
        Initialize the AgentOrchestrator.

        Parameters
        ----------
        agents : Dict[str, BaseAgent]
            Mapping from topic name to its corresponding agent.
        retriever : VectorRetriever
            Vector retriever used to fetch topic-specific documents.
        top_k : int, optional
            Number of documents to retrieve per question, by default 5.
        '''        
        self.agents = agents
        self.retriever = retriever
        self.top_k = top_k

    def route_questions(
        self,
        questions: List[str],
    ) -> Dict[str, List[tuple[str, int]]]:
        '''
        Route decomposed questions to all topic agents and
        compute relevance scores.

        For each topic:
        - Each question is searched independently in the vector store
          using a topic filter
        - Retrieved documents are passed to the corresponding agent
        - The agent assigns a relevance score to the question

        Parameters
        ----------
        questions : List[str]
            List of decomposed sub-questions derived from the user query.

        Returns
        -------
        Dict[str, List[tuple[str, int]]]
            Mapping from topic name to a list of (question, relevance_score)
            tuples, where relevance_score is in the range [0, 100].
        '''
        results = {}

        for topic, agent in self.agents.items():
            question_scores: list[tuple[str, int]] = []
            for question in questions:
                docs = self.retriever.retrieve(
                    query=question,
                    k=self.top_k,
                    filters={'must': [('topic', topic)]}
                )

                score = agent.score_relevance(
                    question=question,
                    documents=docs,
                )
                question_scores.append((question, score))


            results[topic] = question_scores

        return results
