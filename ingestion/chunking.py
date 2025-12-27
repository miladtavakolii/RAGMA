from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    '''
    Unified document chunking class based on LangChain.

    This class is responsible for splitting LangChain `Document` objects
    into smaller, semantically coherent chunks suitable for embedding,
    retrieval, and multi-agent RAG pipelines.

    The chunker uses LangChain's `RecursiveCharacterTextSplitter`, which
    attempts to preserve semantic boundaries such as paragraphs and
    sentences before falling back to word-level splitting.

    Key Design Goals:
    -----------------
    - Fully LangChain-native
    - Metadata preservation (topic, file_type, page, filename, etc.)
    - Suitable for multilingual content (including Persian)
    - Robust for long documents and PDFs
    - Simple single-class interface for early-stage projects
    '''

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        '''
        Initialize the DocumentChunker.

        Parameters
        ----------
        chunk_size : int
            Maximum number of characters per chunk.
            Typical values range from 500 to 1000.
        chunk_overlap : int
            Number of overlapping characters between consecutive chunks.
            Overlap helps preserve context across chunk boundaries.
        '''
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                '\n\n',  # paragraph
                '\n',    # line
                '.',     # sentence (English)
                '!',     
                '؟',     # sentence (Persian)
                ' ',
                '',
            ],
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        '''
        Split a list of LangChain documents into smaller chunks.

        The original document metadata is automatically propagated
        to each chunk. No metadata is modified or removed.

        Parameters
        ----------
        documents : List[Document]
            Input LangChain documents loaded from various sources
            (TXT, PDF, etc.).

        Returns
        -------
        List[Document]
            Chunked LangChain documents ready for embedding and storage.
        '''
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        # Optional sanity check / debug log
        print(
            f'Chunked {len(documents)} documents into {len(chunks)} chunks'
        )

        return chunks
