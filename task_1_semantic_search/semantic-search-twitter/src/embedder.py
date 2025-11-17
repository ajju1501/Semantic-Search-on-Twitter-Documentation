"""Embedding generation module using sentence transformers."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using SentenceTransformer.
    
    Attributes:
        model: The embedding model (default: "all-MiniLM-L6-v2")
        device: Device to run model on ("cpu" or "cuda")
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = None
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (num_texts x embedding_dim)
        """
        if not texts:
            raise ValueError("No texts provided")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        self.embeddings = embeddings
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            1D numpy array (embedding_dim,)
        """
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            2D numpy array (num_texts x embedding_dim)
        """
        return self.model.encode(texts, convert_to_numpy=True)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.embeddings is None:
            # Generate a test embedding
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            return test_embedding.shape[1]
        return self.embeddings.shape[1]
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get stored embeddings."""
        return self.embeddings
    
    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length (for cosine similarity).
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
