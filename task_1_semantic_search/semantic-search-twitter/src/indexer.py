"""FAISS index management module."""

import os
import pickle
from typing import List, Optional
import numpy as np
import faiss
from .utils import save_json, load_json, ensure_directory


class FAISSIndexer:
    """
    Manages FAISS vector index for similarity search.
    
    Attributes:
        index_path: Path to save/load the FAISS index
        metadata_path: Path to save/load metadata
    """
    
    def __init__(self, index_path: str = "embeddings/index.faiss", 
                 metadata_path: str = "embeddings/metadata.json"):
        """
        Initialize the indexer.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        ensure_directory(os.path.dirname(index_path))
    
    def build_index(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of shape (num_vectors, embedding_dim)
            metadata: List of metadata dicts for each embedding
        """
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings provided")
        
        embedding_dim = embeddings.shape[1]
        print(f"Building FAISS index with {embeddings.shape[0]} vectors (dim={embedding_dim})")
        
        # Create FAISS index
        # Use L2 distance (you can also use cosine similarity)
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings
        self.index.add(embeddings.astype('float32'))
        self.metadata = metadata
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def save_index(self) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        ensure_directory(os.path.dirname(self.index_path))
        faiss.write_index(self.index, self.index_path)
        print(f"Saved FAISS index to {self.index_path}")
        
        # Save metadata
        save_json({"metadata": self.metadata}, self.metadata_path)
        print(f"Saved metadata to {self.metadata_path}")
    
    def load_index(self) -> bool:
        """
        Load index and metadata from disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.index_path):
            print(f"Index file not found: {self.index_path}")
            return False
        
        try:
            self.index = faiss.read_index(self.index_path)
            print(f"Loaded FAISS index from {self.index_path}")
            
            # Load metadata
            data = load_json(self.metadata_path)
            self.metadata = data.get("metadata", [])
            print(f"Loaded metadata for {len(self.metadata)} vectors")
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[tuple]:
        """
        Search for top-k similar vectors.
        
        Args:
            query_embedding: Query embedding (1D array)
            k: Number of results to return
            
        Returns:
            List of (distance, metadata) tuples
        """
        if self.index is None:
            raise ValueError("Index not loaded")
        
        # Reshape query to (1, embedding_dim)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((float(dist), self.metadata[idx]))
        
        return results
    
    def is_index_available(self) -> bool:
        """Check if index is available in memory."""
        return self.index is not None
    
    def get_index_size(self) -> int:
        """Get number of vectors in index."""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def index_exists(self) -> bool:
        """Check if index file exists on disk."""
        return os.path.exists(self.index_path)
