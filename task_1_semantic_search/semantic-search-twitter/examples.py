#!/usr/bin/env python3
"""
Example script showing how to use the semantic search system programmatically.
Save results to JSON files for further processing.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from src.search import SemanticSearcher


def main():
    """Run semantic search and save results to JSON."""
    
    # Initialize searcher
    print("🔍 Initializing Semantic Search Engine...")
    searcher = SemanticSearcher(
        data_dir="data/raw_docs",
        model_name="all-MiniLM-L6-v2"
    )
    
    # Initialize index
    if not searcher.initialize():
        print("❌ Failed to initialize")
        sys.exit(1)
    
    print("✓ Index loaded successfully\n")
    
    # Example queries
    queries = [
        "how to authenticate",
        "search tweets",
        "user followers",
        "rate limits",
        "pagination",
        "bookmarks",
    ]
    
    # Create output directory
    output_dir = Path("search_results")
    output_dir.mkdir(exist_ok=True)
    
    # Run searches and save results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "search_engine_stats": searcher.get_stats(),
        "queries": []
    }
    
    for query in queries:
        print(f"🔎 Searching: '{query}'")
        result = searcher.search_json(query, top_k=3)
        
        # Add to all_results
        all_results["queries"].append(result)
        
        # Save individual result to file
        safe_filename = query.replace(" ", "_").lower()
        output_file = output_dir / f"{safe_filename}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"   ✓ Saved to {output_file}")
        print(f"   Found {result['total_results']} results\n")
    
    # Save all results combined
    combined_file = output_dir / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ All results saved to {output_dir}/")
    print(f"   📊 Combined results: {combined_file}")
    print(f"   📈 Individual queries: {len(queries)} files")


if __name__ == "__main__":
    main()
