import logging
import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, util


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    EmbeddingService for text embeddings and similarity calculations
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", normalize_embeddings: bool = True):

        # Model initialization
        self.model_name = model_name
        self.model = EmbeddingService._load_model(model_name)

        # Boolean for embeddings normalization
        self.normalize_embeddings = normalize_embeddings

        logger.info(f"EmbeddingService initialized with model: {model_name}")

    @staticmethod
    def _load_model(model_name: str) -> SentenceTransformer:
        """Helper function to load a SentenceTransformer model by it's name"""
        return SentenceTransformer(model_name)

    def text_to_embedding(self, text: str):
        """
        Generates a single sentence embedding vector for the input text using the all-MiniLM-L6-v2 model.
        """
        return self.model.encode(text,
                                 normalize_embeddings=self.normalize_embeddings,
                                 show_progress_bar=False)


# --- LRU Cache singleton --- #

@lru_cache(maxsize=1)
def get_embedding_service(model_name: str = "all-MiniLM-L6-v2", normalize_embeddings: bool = True):
    """
    Get cached EmbeddingService instance (singleton via LRU cache)
    """
    return EmbeddingService(model_name, normalize_embeddings)

def get_embedding_service_instance():
    """
    Get the default embedding service instance
    Simple wrapper for most common usage
    """
    return get_embedding_service()


