from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class SentenceTransformersEmbedding:
    '''
    Local embedding class using Sentence-Transformers for generating dense vector representations.

    This class provides a convenient interface to convert textual data into L2-normalized
    embeddings using the SentenceTransformersEmbedding model through the Sentence-Transformers library.
    It supports embedding of multiple texts at once or a single query string. These embeddings
    can be directly used for semantic search, clustering, similarity comparison, or as input
    to vector databases such as Qdrant, Pinecone, or FAISS.

    The embeddings generated are normalized (unit length), which makes them suitable for
    cosine similarity calculations and other downstream tasks that rely on vector similarity.
    '''

    def __init__(self, model_name: str = 'google/embeddinggemma-300m'):
        '''
        Initializes the SentenceTransformer model for SentenceTransformersEmbedding.

        This method loads the pretrained Sentence-Transformer model specified by `model_name`.
        The model can be either a Hugging Face model ID or a local path where the model
        is stored. The loaded model is ready to generate embeddings immediately.

        Parameters
        ----------
        model_name : str
            Name or path of the pretrained Sentence-Transformer model. By default, 
            'google/embeddinggemma-300m' is used. The model should be compatible with 
            Sentence-Transformers encoding interface.
        
        Example
        -------
        >>> embedder = EmbeddingGemma(model_name='google/embeddinggemma-300m')
        '''
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        '''
        Generate embeddings for a list of texts.

        This method takes a list of strings and converts each string into a dense,
        L2-normalized vector representation using the Sentence-Transformer model.
        It handles batching internally for efficiency and returns a NumPy array of embeddings.

        Parameters
        ----------
        texts : List[str]
            A list of input strings to embed. Each element in the list is processed
            independently but returned in the same order.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (len(texts), hidden_dim), where `hidden_dim` is
            the dimensionality of the embedding space. All embeddings are L2-normalized.

        Example
        -------
        >>> texts = ["شبکه عصبی چیست؟", "یادگیری ماشین چگونه کار می‌کند؟"]
        >>> vectors = embedder.embed_texts(texts)
        >>> vectors.shape
        (2, 768)
        '''
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        '''
        Generate an embedding for a single query string.

        This method wraps `embed_texts` to handle a single text input. The resulting
        embedding is a 1D NumPy array representing the semantic vector of the input text.
        It is L2-normalized and suitable for immediate use in semantic search, vector
        similarity calculations, or as input to vector databases.

        Parameters
        ----------
        query : str
            A single string to embed. This string will be processed independently
            and converted to its vector representation.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of shape (hidden_dim,) representing the embedding
            of the input query.

        Example
        -------
        >>> query_vec = embedder.embed_query("مقدمه‌ای بر شبکه‌های عصبی")
        >>> query_vec.shape
        (768,)
        '''
        return self.embed_texts([query])[0]
