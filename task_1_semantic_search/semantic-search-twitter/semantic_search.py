#!/usr/bin/env python3
"""
Main entry point for semantic search CLI.

Usage:
    python semantic_search.py --query "how to fetch tweets"
    python semantic_search.py --query "authentication" --top-k 5
    python semantic_search.py --query "API endpoints" --json
    python semantic_search.py --query "search tweets" --rebuild-index --json
"""

import argparse
import sys
import json
from src.search import SemanticSearcher


def format_results(results: list, query: str) -> None:
    """
    Pretty print search results.
    
    Args:
        results: List of search results
        query: Original query
    """
    print(f"\n{'='*70}")
    print(f"Search Results for: '{query}'")
    print(f"Found {len(results)} results")
    print(f"{'='*70}\n")
    
    if not results:
        print("No results found.")
        return
    
    for result in results:
        print(f"▶ Result #{result['rank']}")
        print(f"  Similarity: {result['similarity']} | Distance: {result['distance']}")
        print(f"  Source: {result['source']}")
        print(f"  Preview: {result['chunk']}")
        print()


def format_stats(stats: dict) -> None:
    """Pretty print search engine statistics."""
    print(f"\n{'='*70}")
    print("Search Engine Statistics")
    print(f"{'='*70}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Index Size: {stats['index_size']}")
    print(f"Embedding Model: {stats['embedding_model']}")
    print(f"Embedding Dimension: {stats['embedding_dim']}")
    print(f"Index Available: {'Yes' if stats['index_exists'] else 'No'}")
    print(f"Data Directory: {stats['data_directory']}")
    print(f"{'='*70}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic search for Twitter API documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python semantic_search.py --query "how to fetch tweets"
  python semantic_search.py --query "authentication" --top-k 5
  python semantic_search.py --query "API endpoints" --rebuild-index
  python semantic_search.py --query "search tweets" --json
  python semantic_search.py --stats
        """
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Search query"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 3)"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of embeddings index"
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Include surrounding context in results"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show search engine statistics"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw_docs",
        help="Directory containing markdown files (default: data/raw_docs)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    
    args = parser.parse_args()
    
    # Initialize searcher
    if not args.json:
        print("Initializing search engine...")
    
    searcher = SemanticSearcher(
        data_dir=args.data_dir,
        model_name=args.model
    )
    
    # Initialize or load index
    if not searcher.initialize(rebuild=args.rebuild_index):
        if args.json:
            print(json.dumps({
                "success": False,
                "error": "Failed to initialize search engine"
            }))
        else:
            print("Failed to initialize search engine")
        sys.exit(1)
    
    # Handle JSON mode
    if args.json and args.query:
        result = searcher.search_json(args.query, top_k=args.top_k)
        print(json.dumps(result, indent=2))
        return
    
    # Show stats if requested
    if args.stats or not args.query:
        if not args.json:
            stats = searcher.get_stats()
            format_stats(stats)
    
    # Perform search if query provided (non-JSON mode)
    if args.query and not args.json:
        if args.context:
            results = searcher.search_with_context(
                args.query, 
                top_k=args.top_k
            )
        else:
            results = searcher.search(
                args.query, 
                top_k=args.top_k
            )
        
        format_results(results, args.query)
        
        # Save results to output.json
        output_data = {
            "query": args.query,
            "top_k": args.top_k,
            "total_results": len(results),
            "results": results
        }
        with open("output.json", "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to output.json")


if __name__ == "__main__":
    main()
