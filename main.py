from vectorstore.vector_store_factory import VectorStoreFactory
from retriever.vector_retriever import VectorRetriever
from pipeline.rag_pipeline import RAGPipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

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
    rag = RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt_path='prompts/simple_rag.txt',
    )

    answer = rag.run('شبکه عصبی چیست؟')
    print(answer)


if __name__ == '__main__':
    main()
