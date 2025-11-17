"""Document chunking module for splitting markdown files into chunks."""

import re
import json
from typing import List, Tuple
from pathlib import Path
from .utils import get_markdown_files, read_file


class DocumentChunker:
    """
    Splits documents into chunks for embedding.
    
    Attributes:
        chunk_size: Number of characters per chunk (default: 500)
        overlap: Number of overlapping characters between chunks (default: 100)
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self.metadata = []
    
    def load_documents(self, directory: str) -> int:
        """
        Load all markdown and JSON files from a directory.
        
        Args:
            directory: Path to directory containing markdown/JSON files
            
        Returns:
            Number of chunks created
        """
        files = get_markdown_files(directory)
        
        self.chunks = []
        self.metadata = []
        
        for filepath in files:
            if filepath.endswith('.json'):
                num_chunks = self._chunk_json_file(filepath)
            else:
                content = read_file(filepath)
                if content:
                    num_chunks = self._chunk_document(content, filepath)
            
            if num_chunks > 0:
                print(f"Created {num_chunks} chunks from {Path(filepath).name}")
        
        print(f"Total chunks created: {len(self.chunks)}")
        return len(self.chunks)
    
    def _chunk_json_file(self, filepath: str) -> int:
        """
        Extract text from Postman collection JSON and create chunks.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Number of chunks created
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract relevant text from Postman collection
            text_parts = []
            
            # Get collection info
            if 'info' in data:
                info = data['info']
                if 'name' in info:
                    text_parts.append(f"Collection: {info['name']}")
                if 'description' in info:
                    text_parts.append(f"Description: {info['description']}")
            
            # Get items (API endpoints)
            if 'item' in data:
                items = data['item']
                text_parts.extend(self._extract_items_text(items))
            
            content = "\n".join(text_parts)
            
            if content:
                return self._chunk_document(content, filepath)
            return 0
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {filepath}: {e}")
            return 0
        except Exception as e:
            print(f"Error processing JSON file {filepath}: {e}")
            return 0
    
    @staticmethod
    def _extract_items_text(items: list, prefix: str = "") -> List[str]:
        """
        Recursively extract text from Postman collection items.
        
        Args:
            items: List of items from Postman collection
            prefix: Prefix for item names
            
        Returns:
            List of text parts
        """
        text_parts = []
        
        for item in items:
            if isinstance(item, dict):
                # Get item name
                item_name = item.get('name', 'Unnamed')
                full_name = f"{prefix}/{item_name}" if prefix else item_name
                
                # If item has nested items, recurse
                if 'item' in item:
                    text_parts.extend(DocumentChunker._extract_items_text(
                        item['item'], 
                        full_name
                    ))
                
                # Extract request information
                if 'request' in item:
                    request = item['request']
                    text_parts.append(f"\nEndpoint: {full_name}")
                    
                    if 'method' in request:
                        text_parts.append(f"Method: {request['method']}")
                    
                    if 'url' in request:
                        url = request['url']
                        if isinstance(url, dict):
                            raw_url = url.get('raw', '')
                            text_parts.append(f"URL: {raw_url}")
                    
                    if 'description' in request:
                        text_parts.append(f"Description: {request['description']}")
                    
                    if 'header' in request:
                        headers = request['header']
                        if headers:
                            text_parts.append("Headers: " + ", ".join(
                                [h.get('key', '') for h in headers if isinstance(h, dict)]
                            ))
        
        return text_parts
    
    def _chunk_document(self, content: str, source: str) -> int:
        """
        Split a document into chunks.
        
        Args:
            content: Document text content
            source: Source file path
            
        Returns:
            Number of chunks created
        """
        # Clean content
        content = self._clean_text(content)
        
        if len(content) < self.chunk_size:
            # Document is smaller than chunk size, add as single chunk
            self.chunks.append(content)
            self.metadata.append({
                "source": source,
                "chunk_id": len(self.chunks) - 1,
                "start_char": 0,
                "end_char": len(content)
            })
            return 1
        
        # Create overlapping chunks
        chunks_created = 0
        i = 0
        chunk_id = len(self.chunks)
        
        while i < len(content):
            end = min(i + self.chunk_size, len(content))
            chunk = content[i:end]
            
            if chunk.strip():  # Only add non-empty chunks
                self.chunks.append(chunk)
                self.metadata.append({
                    "source": source,
                    "chunk_id": chunk_id,
                    "start_char": i,
                    "end_char": end
                })
                chunks_created += 1
                chunk_id += 1
            
            # Move to next chunk with overlap
            i += self.chunk_size - self.overlap
        
        return chunks_created
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove multiple newlines
        text = re.sub(r'\n\n+', '\n\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def get_chunks(self) -> List[str]:
        """Get all chunks."""
        return self.chunks
    
    def get_metadata(self) -> List[dict]:
        """Get metadata for all chunks."""
        return self.metadata
    
    def get_chunk_with_context(self, chunk_id: int, context_chunks: int = 1) -> str:
        """
        Get a chunk with surrounding context.
        
        Args:
            chunk_id: ID of the chunk
            context_chunks: Number of chunks before/after to include
            
        Returns:
            Chunk with context
        """
        start = max(0, chunk_id - context_chunks)
        end = min(len(self.chunks), chunk_id + context_chunks + 1)
        
        result = []
        for i in range(start, end):
            if i == chunk_id:
                result.append(f"[>>> {self.chunks[i]} <<<]")
            else:
                result.append(self.chunks[i])
        
        return "\n".join(result)
