"""Utility functions for the semantic search project."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any


def get_markdown_files(directory: str) -> List[str]:
    """
    Get all markdown and JSON files from a directory.
    
    Args:
        directory: Path to the directory containing markdown/JSON files
        
    Returns:
        List of file paths
    """
    path = Path(directory)
    if not path.exists():
        print(f"Warning: Directory {directory} does not exist")
        return []
    
    md_files = list(path.glob("*.md"))
    json_files = list(path.glob("*.json"))
    all_files = md_files + json_files
    print(f"Found {len(md_files)} markdown files and {len(json_files)} JSON files in {directory}")
    return [str(f) for f in all_files]


def read_file(filepath: str) -> str:
    """
    Read file contents.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File contents as string
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return ""
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return ""


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save to
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {filepath}")
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    try:
        if not os.path.exists(filepath):
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return {}


def ensure_directory(dirpath: str) -> None:
    """
    Ensure a directory exists, create if needed.
    
    Args:
        dirpath: Path to directory
    """
    Path(dirpath).mkdir(parents=True, exist_ok=True)
