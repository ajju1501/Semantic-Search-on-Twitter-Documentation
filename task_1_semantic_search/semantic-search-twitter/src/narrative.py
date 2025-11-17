"""Narrative building module for creating coherent API documentation flows."""

import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class APIEndpoint:
    """Represents a discovered API endpoint."""
    name: str
    method: str
    url: str
    description: str
    parameters: List[str]
    chunk_id: int
    similarity: float
    source: str


class NarrativeBuilder:
    """
    Builds coherent narratives from semantic search results.
    
    Converts ranked search results into structured narrative flows that
    explain how to accomplish a task step-by-step.
    """
    
    def __init__(self):
        """Initialize narrative builder."""
        self.endpoints = []
        self.relationships = []
    
    def extract_endpoints(self, search_results: List[dict]) -> List[APIEndpoint]:
        """
        Extract API endpoint information from search results.
        
        Args:
            search_results: List of ranked search results
            
        Returns:
            List of APIEndpoint objects
        """
        endpoints = []
        
        for result in search_results:
            chunk = result.get('chunk', '')
            
            # Parse endpoint information from chunk
            endpoint_data = self._parse_endpoint(chunk, result)
            
            if endpoint_data:
                endpoint = APIEndpoint(
                    name=endpoint_data.get('name', 'Unknown'),
                    method=endpoint_data.get('method', 'GET'),
                    url=endpoint_data.get('url', ''),
                    description=endpoint_data.get('description', ''),
                    parameters=endpoint_data.get('parameters', []),
                    chunk_id=result.get('chunk_id', -1),
                    similarity=result.get('similarity', 0.0),
                    source=result.get('source', '')
                )
                endpoints.append(endpoint)
        
        self.endpoints = endpoints
        return endpoints
    
    @staticmethod
    def _parse_endpoint(chunk: str, result: dict) -> Dict[str, Any]:
        """
        Parse endpoint details from chunk text.
        
        Args:
            chunk: Text chunk containing endpoint info
            result: Result metadata
            
        Returns:
            Dictionary with endpoint information
        """
        lines = chunk.split('\n')
        endpoint_data = {
            'name': 'API Endpoint',
            'method': 'GET',
            'url': '',
            'description': '',
            'parameters': []
        }
        
        # Extract endpoint name
        for i, line in enumerate(lines):
            if 'Endpoint:' in line:
                endpoint_data['name'] = line.replace('Endpoint:', '').strip()
                break
            elif 'Method:' in line and i > 0:
                # Look back for endpoint name
                if i > 0 and 'Endpoint:' in lines[i-1]:
                    endpoint_data['name'] = lines[i-1].replace('Endpoint:', '').strip()
                break
        
        # Extract method
        for line in lines:
            if line.startswith('Method:'):
                endpoint_data['method'] = line.replace('Method:', '').strip()
                break
        
        # Extract URL
        for line in lines:
            if line.startswith('URL:'):
                url = line.replace('URL:', '').strip()
                endpoint_data['url'] = url
                # Extract parameters from URL
                if '?' in url:
                    param_str = url.split('?')[1]
                    params = [p.split('=')[0] for p in param_str.split('&')]
                    endpoint_data['parameters'] = params
                break
        
        # Extract description
        for i, line in enumerate(lines):
            if line.startswith('Description:'):
                endpoint_data['description'] = line.replace('Description:', '').strip()
                break
        
        return endpoint_data
    
    def detect_relationships(self) -> List[Dict[str, Any]]:
        """
        Detect relationships between endpoints.
        
        Examples:
        - search/recent → tweets/:id (detail lookup)
        - tweets/:id → users/:id (author lookup)
        
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        for i, source_ep in enumerate(self.endpoints):
            for target_ep in self.endpoints[i+1:]:
                rel = self._detect_relationship(source_ep, target_ep)
                if rel:
                    relationships.append(rel)
        
        self.relationships = relationships
        return relationships
    
    @staticmethod
    def _detect_relationship(ep1: APIEndpoint, ep2: APIEndpoint) -> Dict[str, Any]:
        """
        Detect relationship between two endpoints.
        
        Args:
            ep1: First endpoint
            ep2: Second endpoint
            
        Returns:
            Relationship dictionary or None
        """
        # Shared parameters indicate relationship
        params1 = set(ep1.parameters)
        params2 = set(ep2.parameters)
        shared = params1 & params2
        
        if not shared:
            return None
        
        # Determine relationship type
        rel_type = 'related'
        
        if 'id' in shared and ep1.method == 'GET' and ep2.method == 'GET':
            if 'search' in ep1.name.lower() and 'lookup' in ep2.name.lower():
                rel_type = 'detail_lookup'
            elif 'tweets' in ep1.name.lower() and 'users' in ep2.name.lower():
                rel_type = 'author_lookup'
        
        return {
            'source': ep1.name,
            'target': ep2.name,
            'type': rel_type,
            'shared_parameters': list(shared),
            'similarity': (ep1.similarity + ep2.similarity) / 2
        }
    
    def build_narrative(self, query: str, search_results: List[dict]) -> Dict[str, Any]:
        """
        Build a complete narrative from search results.
        
        Args:
            query: Original user query
            search_results: Ranked search results
            
        Returns:
            Narrative structure with steps and relationships
        """
        # Extract endpoints
        endpoints = self.extract_endpoints(search_results)
        
        # Detect relationships
        relationships = self.detect_relationships()
        
        # Build narrative steps
        steps = self._create_steps(query, endpoints)
        
        # Create narrative structure
        narrative = {
            "query": query,
            "narrative": {
                "summary": self._generate_summary(query, endpoints),
                "steps": steps,
                "relationships": relationships,
                "key_concepts": self._extract_concepts(query, endpoints)
            }
        }
        
        return narrative
    
    def _create_steps(self, query: str, endpoints: List[APIEndpoint]) -> List[Dict[str, Any]]:
        """
        Create step-by-step instructions from endpoints.
        
        Args:
            query: User query
            endpoints: Extracted endpoints
            
        Returns:
            List of instruction steps
        """
        steps = []
        
        for i, endpoint in enumerate(endpoints, 1):
            step = {
                "step": i,
                "title": endpoint.name,
                "endpoint": endpoint.method + " " + endpoint.url,
                "description": endpoint.description,
                "parameters": endpoint.parameters,
                "method": endpoint.method,
                "url": endpoint.url,
                "relevance": {
                    "similarity_score": endpoint.similarity,
                    "rank": i,
                    "reason": self._generate_step_reason(endpoint, query)
                }
            }
            
            # Add related steps
            if i < len(endpoints):
                step['next_step'] = endpoints[i].name
            
            steps.append(step)
        
        return steps
    
    @staticmethod
    def _generate_summary(query: str, endpoints: List[APIEndpoint]) -> str:
        """Generate narrative summary."""
        if not endpoints:
            return f"No relevant endpoints found for: {query}"
        
        summary = f"To {query.lower()}, use the following API endpoints in order:\n\n"
        
        for i, ep in enumerate(endpoints, 1):
            summary += f"{i}. {ep.name}: {ep.method} {ep.url}\n"
        
        return summary
    
    @staticmethod
    def _generate_step_reason(endpoint: APIEndpoint, query: str) -> str:
        """Generate reason why this endpoint is relevant."""
        if 'search' in endpoint.name.lower():
            return "Allows searching for relevant data"
        elif 'lookup' in endpoint.name.lower():
            return "Retrieves details about the data"
        elif 'count' in endpoint.name.lower():
            return "Provides statistics or counts"
        else:
            return "Related to the query"
    
    @staticmethod
    def _extract_concepts(query: str, endpoints: List[APIEndpoint]) -> List[str]:
        """Extract key concepts from query and results."""
        concepts = set()
        
        # Add query terms
        for word in query.lower().split():
            if len(word) > 3:
                concepts.add(word)
        
        # Add endpoint-related concepts
        for ep in endpoints:
            for param in ep.parameters:
                if param not in ['query', 'id']:
                    concepts.add(param)
        
        return sorted(list(concepts))


def build_narrative_from_search(query: str, search_results: List[dict]) -> Dict[str, Any]:
    """
    Convenience function to build narrative from search results.
    
    Args:
        query: User query
        search_results: Ranked results from semantic search
        
    Returns:
        Complete narrative structure
    """
    builder = NarrativeBuilder()
    return builder.build_narrative(query, search_results)
