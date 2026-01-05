from vectorstore.vector_store_factory import VectorStoreFactory
from retriever.vector_retriever import VectorRetriever
from pipeline.rag_pipeline import RAGPipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from query_decomposition.llm_decomposer import LLMQueryDecomposer

def main() -> None:
    embedder = HuggingFaceEmbeddings(model_name='google/embeddinggemma-300m')

    # vector store
    vector_store = VectorStoreFactory.create(
        backend='qdrant',
        embeddings=embedder,
        collection_name='knowledge_base',
        vector_dim=768,
    )

    # retriever
    retriever = VectorRetriever(vector_store)

    # LLM
    llm = ChatOllama(
        model='gemma3:12b',
        temperature=0,
    )
    # RAG
    query_decomposer = LLMQueryDecomposer(llm)
    rag = RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt_path='prompts/simple_rag.txt',
        query_decomposer=query_decomposer
    )

    answer = rag.run('تفاوت یادگیری ماشین و یادگیری عمیق چیست؟')
    print(answer.text)


if __name__ == '__main__':
    main()
