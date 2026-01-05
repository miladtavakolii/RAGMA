from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain.messages import AIMessage
from retriever.vector_retriever import VectorRetriever
from utils.prompt_loader import load_prompt


class RAGPipeline:
    '''
    Retrieval-Augmented Generation (RAG) pipeline.

    This class implements a simple yet extensible RAG workflow that combines:
    - A vector-based retriever for fetching relevant documents
    - A prompt template for grounding the LLM on retrieved context
    - A language model for generating the final answer

    Overall flow:
        user query
            → retrieve relevant documents
            → build textual context
            → format prompt
            → invoke LLM
            → return generated answer

    The pipeline is intentionally kept minimal and explicit to:
    - Reduce hidden behavior
    - Improve debuggability
    - Allow easy extension (filters, citations, reranking, agents)
    '''

    def __init__(
        self,
        retriever: VectorRetriever,
        llm: BaseChatModel,
        prompt_path: str,
    ) -> None:
        '''
        Initialize the RAG pipeline.

        Parameters
        ----------
        retriever : VectorRetriever
            Retriever component responsible for fetching relevant documents
            from a vector store based on a user query.

        llm : langchain_core.language_models.BaseChatModel
            Language model used to generate answers. Must implement
            the LangChain `BaseChatModel` interface.

        prompt_path : str
            Path to a prompt template file. The prompt must contain
            the following placeholders:
                - {context}
                - {question}

        Notes
        -----
        - The prompt is loaded once during initialization for efficiency.
        - This design assumes synchronous LLM invocation.
        '''
        self.retriever: VectorRetriever = retriever
        self.llm: BaseChatModel = llm
        self.prompt_template: str = load_prompt(prompt_path)

    def _build_context(self, documents: List[Document]) -> str:
        '''
        Construct a single textual context from retrieved documents.

        Parameters
        ----------
        documents : List[langchain_core.documents.Document]
            List of retrieved documents. Each document's `page_content`
            is concatenated to form the final context.

        Returns
        -------
        str
            A single string containing the combined document contents,
            separated by double newlines.

        Notes
        -----
        - This method defines how retrieved information is presented
          to the LLM.
        - More advanced strategies (chunk scoring, citations,
          section headers) can be implemented here.
        '''
        return '\n\n'.join(doc.page_content for doc in documents)

    def run(self, question: str) -> AIMessage:
        '''
        Execute the RAG pipeline for a single user question.

        Parameters
        ----------
        question : str
            User query or question to be answered.

        Returns
        -------
        str
            The generated answer produced by the language model.

        Workflow
        --------
        1. Retrieve relevant documents using the retriever
        2. Build a textual context from retrieved documents
        3. Format the prompt using the context and question
        4. Invoke the language model
        5. Return the model's response

        Notes
        -----
        - The quality of the answer depends heavily on retrieval quality.
        - This method can be extended to return sources, scores,
          or structured outputs instead of raw text.
        '''
        documents: List[Document] = self.retriever.retrieve(question)
        context: str = self._build_context(documents)

        prompt: str = self.prompt_template.format(
            context=context,
            question=question,
        )

        return self.llm.invoke(prompt)
