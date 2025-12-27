from pipeline.ingest_pipeline import IngestPipeline
from embedding.sentence_transformers_embedding import SentenceTransformersEmbedding

if __name__ == "__main__":
    embedder = SentenceTransformersEmbedding(model_name="google/embeddinggemma-300m")
    
    pipeline = IngestPipeline(
        data_dir="data/raw",
        collection_name="knowledge_base",
        embedder=embedder,
        chunk_size=1000,
        chunk_overlap=200,
        batch_size=100
    )

    pipeline.run()
