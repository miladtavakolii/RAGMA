import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader, PyPDFLoader


class DocumentLoader:
    '''
    LangChain-compatible multi-format document loader with topic-aware ingestion.

    This loader is designed for professional Retrieval-Augmented Generation (RAG)
    and multi-agent systems where documents originate from heterogeneous sources
    such as TXT and PDF files and are organized by semantic topics.

    Each subdirectory under the root data directory is treated as a topic.
    The topic name is automatically injected into document metadata and can
    later be used for routing, filtering, or agent selection.

    Supported File Types
    --------------------
    - .txt : Loaded via `TextLoader`
    - .pdf : Loaded via `PyPDFLoader` (page-level documents)

    Output
    ------
    Returns a list of LangChain `Document` objects, fully compatible with:
    - Text splitters
    - Embedding pipelines
    - Vector stores (e.g., Qdrant)
    - Multi-agent routing systems

    Expected Directory Structure
    ----------------------------
    data/raw/
    ├── finance/
    │   ├── report.txt
    │   └── balance_sheet.pdf
    ├── ai/
    │   └── transformers.pdf
    '''

    SUPPORTED_EXTENSIONS = {'.txt', '.pdf'}

    def __init__(self, data_dir: str = 'data/raw'):
        '''
        Initialize the DocumentLoader.

        Parameters
        ----------
        data_dir : str
            Root directory containing topic-based subdirectories.
        '''
        self.data_dir = data_dir

    def _load_file(self, file_path: str) -> List[Document]:
        '''
        Load a single file using the appropriate LangChain loader.

        Parameters
        ----------
        file_path : str
            Absolute path to the file.

        Returns
        -------
        List[Document]
            A list of LangChain Document objects extracted from the file.
        '''
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.txt':
            loader: BaseLoader = TextLoader(file_path, encoding='utf-8')
            return loader.load()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_text = "\n".join([doc.page_content for doc in docs])
            single_doc = Document(
                page_content=all_text
            )
            return [single_doc]

        return []

    def load(self) -> List[Document]:
        '''
        Load all supported documents from the dataset and return LangChain Documents.

        This method:
        1. Traverses topic directories
        2. Dispatches file loading based on extension
        3. Injects standardized metadata into each Document
        4. Returns a unified list of documents ready for chunking

        Returns
        -------
        List[Document]
            LangChain Document objects with enriched metadata:
            - topic
            - filename
            - source (relative path)
            - filetype
            - page (if applicable, e.g. PDFs)
        '''
        documents: List[Document] = []

        for topic in os.listdir(self.data_dir):
            topic_path = os.path.join(self.data_dir, topic)
            if not os.path.isdir(topic_path):
                continue

            for root, _, files in os.walk(topic_path):
                for file_name in files:
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext not in self.SUPPORTED_EXTENSIONS:
                        continue

                    file_path = os.path.join(root, file_name)

                    try:
                        loaded_docs = self._load_file(file_path)

                        for doc in loaded_docs:
                            doc.metadata.update(
                                {
                                    'topic': topic,
                                    'filename': file_name,
                                    'filetype': ext,
                                    'source': os.path.relpath(
                                        file_path, self.data_dir
                                    ),
                                }
                            )
                            documents.append(doc)

                    except Exception as e:
                        print(f'[Warning] Failed to load {file_path}: {e}')

        print(f'[DocumentLoader] Loaded {len(documents)} documents')
        return documents
