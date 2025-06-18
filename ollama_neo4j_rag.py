#!/usr/bin/env python3
"""
Ultimate Ollama RAG System for AAPS Projects
Optimized for the Ultimate Multi-Repository Analyzer database structure
Enhanced with better context retrieval and improved performance
"""

import argparse
import json
import logging
import sys
import re
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateNeo4jRAGRetriever:
    """Enhanced RAG retriever optimized for the ultimate analyzer database"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.available_repos = self._get_available_repositories()
        logger.info(f"Connected to Neo4j. Available repositories: {self.available_repos}")
    
    def close(self):
        self.driver.close()
    
    def _get_available_repositories(self) -> List[str]:
        """Get list of available repositories from the database"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (r:Repository) RETURN r.name as name ORDER BY r.name")
                return [record["name"] for record in result]
        except Exception as e:
            logger.warning(f"Could not get repositories: {e}")
            return ["EN_new", "EN_old", "AAPS_source"]  # Fallback
    
    def execute_query(self, query: str, parameters: Dict = None, limit: int = 25) -> List[Dict]:
        """Execute a Cypher query with error handling"""
        try:
            with self.driver.session() as session:
                # Add automatic LIMIT if not present and not a COUNT query
                if "LIMIT" not in query.upper() and "COUNT" not in query.upper():
                    query += f" LIMIT {limit}"
                
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return []
    
    def get_database_overview(self) -> Dict[str, Any]:
        """Get comprehensive database overview"""
        overview = {}
        
        # Repository statistics
        repo_stats = self.execute_query("""
            MATCH (r:Repository)
            RETURN r.name as repository, 
                   r.file_count as files,
                   r.total_loc as loc,
                   r.total_functions as functions
            ORDER BY r.file_count DESC
        """)
        overview['repositories'] = repo_stats
        
        # Global statistics
        global_stats = self.execute_query("""
            MATCH (f:File)
            RETURN count(f) as total_files,
                   sum(f.lines_of_code) as total_loc,
                   sum(f.function_count) as total_functions,
                   avg(f.importance_score) as avg_importance
        """, limit=1)
        overview['global'] = global_stats[0] if global_stats else {}
        
        # File type breakdown
        file_types = self.execute_query("""
            MATCH (f:File)
            RETURN f.file_type as type, count(f) as count
            ORDER BY count DESC
        """)
        overview['file_types'] = file_types
        
        return overview
    
    def search_files_by_keywords(self, keywords: List[str], repository: str = None, limit: int = 20) -> List[Dict]:
        """Enhanced keyword search with better scoring"""
        search_terms = [term.lower() for term in keywords]
        
        # Build repository filter
        repo_filter = ""
        if repository and repository in self.available_repos:
            repo_filter = f"AND f.repository = '{repository}'"
        
        query = f"""
        MATCH (f:File)
        WHERE ANY(term IN $search_terms WHERE 
                  toLower(f.name) CONTAINS term OR 
                  toLower(f.path) CONTAINS term OR 
                  toLower(f.package) CONTAINS term)
        {repo_filter}
        WITH f, 
             [term IN $search_terms WHERE 
              toLower(f.name) CONTAINS term OR 
              toLower(f.path) CONTAINS term OR 
              toLower(f.package) CONTAINS term] as matching_terms
        RETURN f.name as filename, 
               f.path as path, 
               f.repository as repository,
               f.package as package,
               f.file_type as file_type,
               f.lines_of_code as loc, 
               f.function_count as functions,
               f.class_count as classes,
               f.importance_score as importance,
               f.complexity_score as complexity,
               size(matching_terms) as match_score
        ORDER BY match_score DESC, f.importance_score DESC
        LIMIT {limit}
        """
        
        return self.execute_query(query, {"search_terms": search_terms})
    
    def get_file_details(self, filename: str, repository: str = None) -> Dict[str, Any]:
        """Get comprehensive file details with relationships"""
        # Build repository filter
        where_parts = ["f.name = $filename"]
        if repository and repository in self.available_repos:
            where_parts.append(f"f.repository = '{repository}'")
        
        where_clause = " AND ".join(where_parts)
        
        query = f"""
        MATCH (f:File)
        WHERE {where_clause}
        OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
        WHERE out.repository = f.repository
        OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
        WHERE in.repository = f.repository
        WITH f, 
             collect(DISTINCT {{name: target.name, weight: out.weight}}) as calls_to,
             collect(DISTINCT {{name: source.name, weight: in.weight}}) as called_by
        RETURN f.name as filename, 
               f.path as path, 
               f.repository as repository,
               f.package as package,
               f.file_type as file_type,
               f.lines_of_code as loc,
               f.function_count as functions,
               f.class_count as classes,
               f.importance_score as importance,
               f.complexity_score as complexity,
               calls_to,
               called_by,
               size(calls_to) as outgoing_calls,
               size(called_by) as incoming_calls
        """
        
        result = self.execute_query(query, {"filename": filename}, limit=1)
        return result[0] if result else {}
    
    def find_important_files(self, repository: str = None, limit: int = 15) -> List[Dict]:
        """Find most important files with enhanced metadata"""
        repo_filter = ""
        if repository and repository in self.available_repos:
            repo_filter = f"WHERE f.repository = '{repository}'"
        
        query = f"""
        MATCH (f:File)
        {repo_filter}
        OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
        WHERE out.repository = f.repository
        OPTIONAL MATCH (source:File)-[in:CALLS]->(f)  
        WHERE in.repository = f.repository
        WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
        RETURN f.name as filename,
               f.repository as repository,
               f.package as package,
               f.file_type as file_type,
               f.lines_of_code as loc,
               f.function_count as functions,
               f.importance_score as importance,
               f.complexity_score as complexity,
               outgoing,
               incoming,
               (outgoing + incoming) as total_connections
        ORDER BY f.importance_score DESC
        LIMIT {limit}
        """
        
        return self.execute_query(query)
    
    def find_related_files(self, concept: str, repository: str = None, limit: int = 12) -> List[Dict]:
        """Find files related to a concept with enhanced relevance scoring"""
        keywords = concept.lower().split()
        
        repo_filter = ""
        if repository and repository in self.available_repos:
            repo_filter = f"AND f.repository = '{repository}'"
        
        query = f"""
        MATCH (f:File)
        WHERE ANY(term IN $keywords WHERE 
                  toLower(f.name) CONTAINS term OR 
                  toLower(f.path) CONTAINS term OR 
                  toLower(f.package) CONTAINS term)
        {repo_filter}
        WITH f,
             [term IN $keywords WHERE 
              toLower(f.name) CONTAINS term OR 
              toLower(f.path) CONTAINS term OR 
              toLower(f.package) CONTAINS term] as matching_terms
        OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
        WHERE out.repository = f.repository
        OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
        WHERE in.repository = f.repository
        WITH f, matching_terms, count(out) + count(in) as connectivity
        RETURN f.name as filename,
               f.repository as repository,
               f.package as package,
               f.file_type as file_type,
               f.lines_of_code as loc,
               f.function_count as functions,
               f.importance_score as importance,
               connectivity,
               size(matching_terms) as relevance_score
        ORDER BY relevance_score DESC, f.importance_score DESC, connectivity DESC
        LIMIT {limit}
        """
        
        return self.execute_query(query, {"keywords": keywords})
    
    def get_architecture_overview(self, repository: str = None) -> Dict[str, Any]:
        """Get architectural overview with enhanced insights"""
        repo_filter = ""
        if repository and repository in self.available_repos:
            repo_filter = f"WHERE f.repository = '{repository}'"
        
        queries = {
            "summary": f"""
                MATCH (f:File)
                {repo_filter}
                RETURN count(f) as total_files,
                       sum(f.lines_of_code) as total_loc,
                       sum(f.function_count) as total_functions,
                       sum(f.class_count) as total_classes,
                       avg(f.importance_score) as avg_importance,
                       max(f.importance_score) as max_importance
            """,
            
            "critical_files": f"""
                MATCH (f:File)
                {repo_filter}
                OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
                WHERE out.repository = f.repository  
                OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
                WHERE in.repository = f.repository
                WITH f, count(out) + count(in) as connections
                WHERE connections > 2
                RETURN f.name as filename,
                       f.repository as repository,
                       f.importance_score as importance,
                       connections
                ORDER BY connections DESC, f.importance_score DESC
                LIMIT 10
            """,
            
            "entry_points": f"""
                MATCH (f:File)
                {repo_filter}
                OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
                WHERE out.repository = f.repository
                OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
                WHERE in.repository = f.repository  
                WITH f, count(out) as outgoing, count(in) as incoming
                WHERE incoming <= 1 AND outgoing >= 3
                RETURN f.name as filename,
                       f.repository as repository,
                       f.importance_score as importance,
                       outgoing,
                       incoming
                ORDER BY outgoing DESC, f.importance_score DESC
                LIMIT 8
            """,
            
            "packages": f"""
                MATCH (f:File)
                {repo_filter}
                WHERE f.package IS NOT NULL AND f.package <> 'unknown'
                WITH f.package as package,
                     count(f) as file_count,
                     avg(f.importance_score) as avg_importance,
                     sum(f.lines_of_code) as total_loc
                WHERE file_count > 1
                RETURN package, file_count, avg_importance, total_loc
                ORDER BY avg_importance DESC
                LIMIT 10
            """
        }
        
        if not repository:
            queries["repositories"] = """
                MATCH (r:Repository)
                RETURN r.name as repository,
                       r.file_count as files,
                       r.total_loc as loc,
                       r.total_functions as functions
                ORDER BY r.file_count DESC
            """
        
        result = {}
        for key, query in queries.items():
            result[key] = self.execute_query(query)
        
        return result
    
    def find_cross_repository_patterns(self, concept: str) -> List[Dict]:
        """Find patterns across different repositories"""
        keywords = concept.lower().split()
        
        query = """
        MATCH (f:File)
        WHERE ANY(term IN $keywords WHERE 
                  toLower(f.name) CONTAINS term OR 
                  toLower(f.package) CONTAINS term)
        WITH f.repository as repository,
             collect({
                 name: f.name, 
                 importance: f.importance_score, 
                 loc: f.lines_of_code,
                 package: f.package
             }) as files,
             count(f) as total_matching,
             avg(f.importance_score) as avg_importance
        WHERE total_matching > 0
        RETURN repository, 
               files[0..3] as sample_files, 
               total_matching,
               round(avg_importance, 2) as avg_importance
        ORDER BY total_matching DESC, avg_importance DESC
        """
        
        return self.execute_query(query, {"keywords": keywords})
    
    def get_repository_comparison(self, metric: str = "importance") -> List[Dict]:
        """Enhanced repository comparison"""
        metric_field = {
            "importance": "f.importance_score",
            "complexity": "f.complexity_score",
            "size": "f.lines_of_code", 
            "functions": "f.function_count"
        }.get(metric, "f.importance_score")
        
        query = f"""
        MATCH (f:File)
        WITH f.repository as repository,
             avg({metric_field}) as avg_metric,
             max({metric_field}) as max_metric,
             count(f) as file_count,
             sum(f.lines_of_code) as total_loc,
             sum(f.function_count) as total_functions
        RETURN repository, 
               round(avg_metric, 2) as avg_metric,
               round(max_metric, 2) as max_metric,
               file_count,
               total_loc,
               total_functions
        ORDER BY avg_metric DESC
        """
        
        return self.execute_query(query)
    
    def trace_call_path(self, from_file: str, to_file: str, repository: str = None, max_depth: int = 4) -> List[Dict]:
        """Find call paths between files with enhanced path analysis"""
        repo_constraint = ""
        if repository and repository in self.available_repos:
            repo_constraint = f"AND start.repository = '{repository}' AND end.repository = '{repository}'"
        
        query = f"""
        MATCH (start:File {{name: $from_file}}), (end:File {{name: $to_file}})
        WHERE start <> end {repo_constraint}
        MATCH path = shortestPath((start)-[:CALLS*1..{max_depth}]->(end))
        WHERE all(rel in relationships(path) WHERE rel.repository = start.repository)
        RETURN [node in nodes(path) | {{
                   name: node.name, 
                   repository: node.repository,
                   importance: node.importance_score
               }}] as call_path,
               [rel in relationships(path) | rel.weight] as call_weights,
               length(path) as path_length,
               reduce(total = 0, rel in relationships(path) | total + rel.weight) as total_weight
        ORDER BY path_length ASC, total_weight DESC
        LIMIT 5
        """
        
        return self.execute_query(query, {"from_file": from_file, "to_file": to_file})


class EnhancedOllamaClient:
    """Enhanced Ollama client with better error handling and streaming"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:1.5b"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        self.session.timeout = 10
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return self.model in available_models
            return False
        except requests.RequestException:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except requests.RequestException:
            return []
    
    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 2048) -> str:
        """Generate response with enhanced context handling"""
        # Truncate context if it's too long
        max_context_length = 8000  # Leave room for prompt and response
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n\n[Context truncated for length...]"
        
        full_prompt = f"""You are an expert assistant analyzing the AAPS (AndroidAPS) multi-repository codebase. You have access to detailed information about three repositories:

1. **EN_new** - The latest EatingNow variant 
2. **EN_old** - The older EatingNow variant
3. **AAPS_source** - The main AndroidAPS source code

**Context from the project database:**
{context}

**User Question:** {prompt}

**Instructions:**
- Answer based ONLY on the provided context from the project database
- Always mention which repository(ies) you're referring to
- If comparing across repositories, be specific about differences
- If the context doesn't contain relevant information, say so clearly
- Focus on the actual code structure, relationships, and functionality shown in the data
- Provide specific file names, importance scores, and relationships when available

**Response:**"""
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Very low temperature for factual responses
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": max_tokens
                    }
                },
                timeout=180  # 3 minute timeout for complex questions
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error: Ollama API returned status {response.status_code}"
                
        except requests.RequestException as e:
            return f"Error communicating with Ollama: {e}"


class UltimateAAPSRAGSystem:
    """Ultimate RAG system optimized for the new database structure"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 ollama_url: str = "http://localhost:11434", model: str = "deepseek-r1:1.5b"):
        self.retriever = UltimateNeo4jRAGRetriever(neo4j_uri, neo4j_user, neo4j_password)
        self.ollama = EnhancedOllamaClient(ollama_url, model)
        self.available_repos = self.retriever.available_repos
        
    def close(self):
        self.retriever.close()
    
    def extract_keywords_enhanced(self, question: str) -> List[str]:
        """Enhanced keyword extraction with domain-specific terms"""
        # Enhanced stop words including programming terms
        stop_words = {
            'the', 'is', 'are', 'how', 'what', 'where', 'when', 'why', 'who', 
            'does', 'do', 'can', 'could', 'would', 'should', 'will', 'a', 'an', 
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'between', 'across', 'compare', 'difference', 'similar', 'show', 'tell',
            'get', 'find', 'list', 'display', 'give', 'me', 'i', 'my', 'you', 'your'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Boost important domain terms
        domain_terms = {
            'aaps', 'androidaps', 'glucose', 'insulin', 'pump', 'cgm', 'blood',
            'loop', 'algorithm', 'dose', 'bolus', 'basal', 'carb', 'treatment',
            'automation', 'plugin', 'sensor', 'eating', 'now', 'diabetes'
        }
        
        # Prioritize domain-specific terms
        prioritized = []
        for word in keywords:
            if word in domain_terms:
                prioritized.insert(0, word)  # Put domain terms first
            else:
                prioritized.append(word)
        
        return prioritized[:12]  # Return top 12 keywords
    
    def detect_repository_context(self, question: str) -> Optional[str]:
        """Enhanced repository detection"""
        question_lower = question.lower()
        
        # Direct repository mentions
        for repo in self.available_repos:
            if repo.lower() in question_lower:
                return repo
        
        # Contextual clues
        if any(term in question_lower for term in ['en_new', 'new version', 'latest', 'current']):
            return 'EN_new'
        elif any(term in question_lower for term in ['en_old', 'old version', 'previous', 'original eating']):
            return 'EN_old'
        elif any(term in question_lower for term in ['aaps_source', 'main aaps', 'base aaps', 'nightscout', 'source']):
            return 'AAPS_source'
        
        return None
    
    def detect_question_type(self, question: str) -> str:
        """Detect the type of question being asked"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['compare', 'difference', 'vs', 'versus', 'between']):
            return 'comparison'
        elif any(word in question_lower for word in ['how', 'algorithm', 'work', 'process']):
            return 'explanation'
        elif any(word in question_lower for word in ['find', 'show', 'list', 'get', 'which']):
            return 'search'
        elif any(word in question_lower for word in ['overview', 'summary', 'architecture', 'structure']):
            return 'overview'
        elif any(word in question_lower for word in ['call', 'relationship', 'connect', 'link']):
            return 'relationships'
        else:
            return 'general'
    
    def retrieve_context_enhanced(self, question: str) -> str:
        """Enhanced context retrieval with intelligent selection"""
        keywords = self.extract_keywords_enhanced(question)
        target_repo = self.detect_repository_context(question)
        question_type = self.detect_question_type(question)
        
        context_parts = []
        
        # Add repository context
        if target_repo:
            context_parts.append(f"🎯 FOCUSED ON REPOSITORY: {target_repo}")
        else:
            context_parts.append(f"📚 AVAILABLE REPOSITORIES: {', '.join(self.available_repos)}")
        
        # Get relevant files based on question type
        if question_type == 'overview' or any(word in keywords for word in ['architecture', 'structure']):
            arch_info = self.retriever.get_architecture_overview(target_repo)
            self._add_architecture_context(context_parts, arch_info, target_repo)
        
        if question_type == 'comparison' and not target_repo:
            comparison = self.retriever.get_repository_comparison()
            self._add_comparison_context(context_parts, comparison)
        
        # Always get relevant files for keywords
        if keywords:
            relevant_files = self.retriever.search_files_by_keywords(keywords, target_repo, 12)
            if relevant_files:
                self._add_relevant_files_context(context_parts, relevant_files)
        
        # Get important files for context
        important_files = self.retriever.find_important_files(target_repo, 8)
        if important_files:
            self._add_important_files_context(context_parts, important_files)
        
        # Handle specific concept searches
        for keyword in keywords:
            if len(keyword) > 4:  # Only substantial keywords
                related_files = self.retriever.find_related_files(keyword, target_repo, 6)
                if related_files:
                    self._add_concept_files_context(context_parts, keyword, related_files)
        
        # Handle cross-repository questions
        if question_type == 'comparison' or any(word in keywords for word in ['across', 'between']):
            for keyword in keywords[:3]:  # Top 3 keywords only
                if len(keyword) > 4:
                    cross_patterns = self.retriever.find_cross_repository_patterns(keyword)
                    if cross_patterns:
                        self._add_cross_repo_context(context_parts, keyword, cross_patterns)
        
        # Handle relationship questions
        if question_type == 'relationships':
            # Look for file mentions to trace paths
            file_mentions = [word for word in keywords if word.endswith('.java') or word.endswith('.kt')]
            if len(file_mentions) >= 2:
                paths = self.retriever.trace_call_path(file_mentions[0], file_mentions[1], target_repo)
                if paths:
                    self._add_path_context(context_parts, file_mentions[0], file_mentions[1], paths)
        
        return "\n".join(context_parts)
    
    def _add_architecture_context(self, context_parts: List[str], arch_info: Dict, target_repo: str):
        """Add architectural overview to context"""
        context_parts.append("\n🏗️ ARCHITECTURE OVERVIEW:")
        
        if arch_info.get('summary'):
            summary = arch_info['summary'][0]
            repo_text = f" ({target_repo})" if target_repo else ""
            context_parts.append(f"📊 Project Summary{repo_text}:")
            context_parts.append(f"   Files: {summary.get('total_files', 0):,}")
            context_parts.append(f"   Lines of Code: {summary.get('total_loc', 0):,}")
            context_parts.append(f"   Functions: {summary.get('total_functions', 0):,}")
            context_parts.append(f"   Average Importance: {summary.get('avg_importance', 0):.1f}")
        
        if arch_info.get('critical_files'):
            context_parts.append("\n🔥 Most Critical Files:")
            for file_info in arch_info['critical_files'][:5]:
                repo = file_info.get('repository', 'unknown')
                context_parts.append(f"   • {file_info['filename']} ({repo}): {file_info['connections']} connections, importance: {file_info.get('importance', 0):.1f}")
        
        if arch_info.get('repositories'):
            context_parts.append("\n📚 Repository Breakdown:")
            for repo_info in arch_info['repositories']:
                context_parts.append(f"   • {repo_info['repository']}: {repo_info['files']} files, {repo_info['functions']} functions")
    
    def _add_comparison_context(self, context_parts: List[str], comparison: List[Dict]):
        """Add repository comparison to context"""
        context_parts.append("\n🔄 REPOSITORY COMPARISON:")
        for comp in comparison:
            context_parts.append(f"   📦 {comp['repository']}:")
            context_parts.append(f"      Files: {comp['file_count']}, LOC: {comp['total_loc']:,}")
            context_parts.append(f"      Functions: {comp['total_functions']:,}, Avg Importance: {comp['avg_metric']:.1f}")
    
    def _add_relevant_files_context(self, context_parts: List[str], relevant_files: List[Dict]):
        """Add relevant files to context"""
        context_parts.append("\n🎯 MOST RELEVANT FILES:")
        for file_info in relevant_files[:8]:
            repo = file_info.get('repository', 'unknown')
            context_parts.append(f"   • {file_info['filename']} ({repo})")
            context_parts.append(f"     Package: {file_info.get('package', 'unknown')}, Importance: {file_info.get('importance', 0):.1f}")
            context_parts.append(f"     {file_info.get('functions', 0)} functions, {file_info.get('loc', 0)} LOC")
    
    def _add_important_files_context(self, context_parts: List[str], important_files: List[Dict]):
        """Add important files to context"""
        context_parts.append("\n⭐ MOST IMPORTANT FILES:")
        for file_info in important_files[:6]:
            repo = file_info.get('repository', 'unknown')
            connections = file_info.get('total_connections', 0)
            context_parts.append(f"   • {file_info['filename']} ({repo})")
            context_parts.append(f"     Importance: {file_info.get('importance', 0):.1f}, Connections: {connections}")
    
    def _add_concept_files_context(self, context_parts: List[str], keyword: str, related_files: List[Dict]):
        """Add concept-related files to context"""
        context_parts.append(f"\n🔍 FILES RELATED TO '{keyword.upper()}':")
        for file_info in related_files[:5]:
            repo = file_info.get('repository', 'unknown')
            context_parts.append(f"   • {file_info['filename']} ({repo}): importance {file_info.get('importance', 0):.1f}")
    
    def _add_cross_repo_context(self, context_parts: List[str], keyword: str, cross_patterns: List[Dict]):
        """Add cross-repository patterns to context"""
        context_parts.append(f"\n🌐 '{keyword.upper()}' ACROSS REPOSITORIES:")
        for pattern in cross_patterns[:4]:
            sample_files = pattern.get('sample_files', [])
            if sample_files:
                top_file = sample_files[0]
                context_parts.append(f"   • {pattern['repository']}: {pattern['total_matching']} files")
                context_parts.append(f"     Top file: {top_file.get('name', 'unknown')} (importance: {top_file.get('importance', 0):.1f})")
    
    def _add_path_context(self, context_parts: List[str], from_file: str, to_file: str, paths: List[Dict]):
        """Add call path information to context"""
        context_parts.append(f"\n🔗 CALL PATHS FROM {from_file} TO {to_file}:")
        for path in paths[:3]:
            path_nodes = path.get('call_path', [])
            if path_nodes:
                path_names = [node.get('name', 'unknown') for node in path_nodes]
                context_parts.append(f"   • {' → '.join(path_names)} (length: {path.get('path_length', 0)})")
    
    def answer_question(self, question: str) -> str:
        """Answer a question using enhanced RAG approach"""
        logger.info(f"Processing question: {question}")
        
        # Retrieve enhanced context
        context = self.retrieve_context_enhanced(question)
        
        if not context.strip():
            return "I couldn't find relevant information in the project database for your question."
        
        logger.info(f"Retrieved context length: {len(context)} characters")
        
        # Generate response using enhanced Ollama client
        response = self.ollama.generate_response(question, context)
        
        return response
    
    def interactive_mode(self):
        """Enhanced interactive Q&A session"""
        print(f"🤖 AAPS Ultimate Multi-Repository RAG System")
        print(f"🧠 Model: {self.ollama.model}")
        print(f"📚 Available repositories: {', '.join(self.available_repos)}")
        print(f"💡 Enhanced with intelligent context retrieval and analysis")
        print(f"\nType 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                question = input("❓ Ask about the AAPS projects: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    self.show_enhanced_help()
                    continue
                elif question.lower() == 'stats':
                    self.show_enhanced_stats()
                    continue
                elif question.lower() == 'repos':
                    self.show_repositories()
                    continue
                elif question.lower() == 'examples':
                    self.show_examples()
                    continue
                elif not question:
                    continue
                
                print("🔍 Analyzing question and searching database...")
                
                # Show detected context
                target_repo = self.detect_repository_context(question)
                question_type = self.detect_question_type(question)
                keywords = self.extract_keywords_enhanced(question)
                
                print(f"🎯 Detected: {question_type} question", end="")
                if target_repo:
                    print(f" about {target_repo}")
                else:
                    print(" (cross-repository)")
                print(f"🔑 Key terms: {', '.join(keywords[:5])}")
                
                answer = self.answer_question(question)
                print(f"\n🤖 {answer}\n")
                print("-" * 80)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("👋 Goodbye!")
    
    def show_enhanced_help(self):
        """Show enhanced help information"""
        help_text = f"""
🤖 AAPS Ultimate RAG System - Enhanced Help

📚 Available Repositories:
{chr(10).join([f'   • {repo}' for repo in self.available_repos])}

💡 Smart Features:
   • Automatic repository detection from your questions
   • Intelligent question type classification
   • Enhanced keyword extraction with domain knowledge
   • Cross-repository comparison capabilities
   • Call path tracing between files

🎯 Question Types Supported:
   • Comparison: "Compare pump algorithms between EN_new and AAPS_source"
   • Explanation: "How does the insulin calculation work in EN_new?"
   • Search: "Find all files related to glucose monitoring"
   • Overview: "What's the architecture of the EN_old repository?"
   • Relationships: "What files call MainActivity.kt in AAPS_source?"

📝 Commands:
   • 'stats' - Show enhanced database statistics
   • 'repos' - Show detailed repository information  
   • 'examples' - Show example questions
   • 'help' - Show this help
   • 'quit' - Exit

🔧 Repository-Specific Questions:
   • Start with repository name: "In EN_new, what files handle..."
   • Use contextual terms: "latest version", "original AAPS", "old eating now"

🌐 Cross-Repository Questions:
   • "Compare [feature] between repositories"
   • "Which repository has the best [functionality]?"
   • "What are the differences in [component] across versions?"
"""
        print(help_text)
    
    def show_enhanced_stats(self):
        """Show enhanced project statistics"""
        print("📊 Enhanced Database Statistics:")
        
        overview = self.retriever.get_database_overview()
        
        # Global stats
        if overview.get('global'):
            global_stats = overview['global']
            print(f"\n🌍 Global Overview:")
            print(f"   Total Files: {global_stats.get('total_files', 0):,}")
            print(f"   Total LOC: {global_stats.get('total_loc', 0):,}")
            print(f"   Total Functions: {global_stats.get('total_functions', 0):,}")
            print(f"   Average Importance: {global_stats.get('avg_importance', 0):.1f}")
        
        # Repository breakdown
        if overview.get('repositories'):
            print(f"\n📚 Repository Breakdown:")
            for repo in overview['repositories']:
                print(f"   📦 {repo['repository']}:")
                print(f"      Files: {repo['files']:,}")
                print(f"      LOC: {repo['loc']:,}")
                print(f"      Functions: {repo['functions']:,}")
        
        # File types
        if overview.get('file_types'):
            print(f"\n📄 File Types:")
            for file_type in overview['file_types']:
                print(f"   • {file_type['type']}: {file_type['count']:,} files")
        
        print()
    
    def show_repositories(self):
        """Show detailed repository information"""
        print("📚 Detailed Repository Information:")
        
        for repo_name in self.available_repos:
            print(f"\n📦 {repo_name}:")
            
            # Get repository-specific overview
            arch_info = self.retriever.get_architecture_overview(repo_name)
            
            if arch_info.get('summary'):
                summary = arch_info['summary'][0]
                print(f"   📊 Statistics:")
                print(f"      Files: {summary.get('total_files', 0):,}")
                print(f"      LOC: {summary.get('total_loc', 0):,}")
                print(f"      Functions: {summary.get('total_functions', 0):,}")
                print(f"      Avg Importance: {summary.get('avg_importance', 0):.1f}")
            
            # Top files
            important_files = self.retriever.find_important_files(repo_name, 3)
            if important_files:
                print(f"   ⭐ Top Files:")
                for file_info in important_files:
                    print(f"      • {file_info['filename']}: importance {file_info.get('importance', 0):.1f}")
            
            # Top packages
            if arch_info.get('packages'):
                print(f"   📦 Key Packages:")
                for pkg in arch_info['packages'][:3]:
                    print(f"      • {pkg['package']}: {pkg['file_count']} files")
        
        print()
    
    def show_examples(self):
        """Show example questions"""
        examples = [
            "🔍 Search Examples:",
            "   • What files handle blood glucose calculations?",
            "   • Find all automation-related files in EN_new",
            "   • Show me pump communication files",
            "",
            "🔄 Comparison Examples:", 
            "   • Compare insulin algorithms between EN_new and AAPS_source",
            "   • What are the differences between EN_old and EN_new?",
            "   • Which repository has better glucose monitoring?",
            "",
            "🏗️ Architecture Examples:",
            "   • What's the overall architecture of AAPS_source?",
            "   • Show me the most important files in EN_new",
            "   • What are the entry points in the EN_old codebase?",
            "",
            "🔗 Relationship Examples:",
            "   • What files call MainActivity.kt?",
            "   • How is PumpPlugin.java connected to other files?",
            "   • Trace the call path from X.java to Y.kt",
            "",
            "💡 Explanation Examples:",
            "   • How does the loop algorithm work?",
            "   • Explain the bolus calculation process",
            "   • How does CGM data flow through the system?"
        ]
        
        print("\n".join(examples))
        print()


def main():
    parser = argparse.ArgumentParser(description="AAPS Ultimate Multi-Repository RAG System")
    parser.add_argument("--model", default="deepseek-r1:1.5b", 
                       help="Ollama model to use (default: deepseek-r1:1.5b)")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687",
                       help="Neo4j URI (default: bolt://localhost:7687)")
    parser.add_argument("--neo4j-user", default="neo4j",
                       help="Neo4j username (default: neo4j)")
    parser.add_argument("--neo4j-password", default="password",
                       help="Neo4j password (default: password)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("--question", type=str,
                       help="Single question to ask (if not provided, starts interactive mode)")
    parser.add_argument("--repository", type=str,
                       help="Focus on specific repository (EN_new, EN_old, AAPS_source)")
    
    args = parser.parse_args()
    
    print("🚀 AAPS Ultimate Multi-Repository RAG System")
    print("💡 Enhanced for the Ultimate Analyzer database structure")
    print("="*60)
    
    # Initialize RAG system
    try:
        rag_system = UltimateAAPSRAGSystem(
            args.neo4j_uri, args.neo4j_user, args.neo4j_password,
            args.ollama_url, args.model
        )
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("Make sure Neo4j is running and credentials are correct")
        print("Also ensure the Ultimate Analyzer has been run to populate the database")
        sys.exit(1)
    
    # Check available repositories
    if not rag_system.available_repos:
        print("❌ No repositories found in database")
        print("Make sure the Ultimate Analyzer (aaps_analyzer.py) has been run")
        sys.exit(1)
    
    print(f"✅ Connected to database with repositories: {', '.join(rag_system.available_repos)}")
    
    # Check Ollama availability
    if not rag_system.ollama.is_available():
        print(f"❌ Model '{args.model}' not available in Ollama")
        available_models = rag_system.ollama.get_available_models()
        if available_models:
            print("Available models:")
            for model in available_models[:10]:  # Show first 10
                print(f"  - {model}")
        else:
            print("  Ollama may not be running")
        sys.exit(1)
    
    print(f"✅ Ollama model '{args.model}' is available")
    
    # Single question mode or interactive mode
    if args.question:
        if args.repository:
            # Modify question to include repository context
            args.question = f"In repository {args.repository}: {args.question}"
        
        print(f"\nQuestion: {args.question}")
        print("="*60)
        answer = rag_system.answer_question(args.question)
        print(answer)
    else:
        rag_system.interactive_mode()
    
    rag_system.close()

if __name__ == "__main__":
    main()
