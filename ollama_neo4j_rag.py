#!/usr/bin/env python3
"""
Enhanced Ultimate Ollama RAG System for AAPS Projects
WITH SOURCE CODE ACCESS, EATING NOW PRIORITIZATION, AND CODE GENERATION CACHING
Optimized for the Enhanced Multi-Repository Analyzer database structure
Features automatic code detection, caching, and eating-now-focused responses
"""

import argparse
import json
import logging
import sys
import re
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeCache:
    """Smart code caching system for generated code snippets"""
    
    def __init__(self, cache_dir: str = "./generated_code_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session_cache = {}
        
    def detect_code_in_response(self, response: str) -> List[Dict[str, str]]:
        """Detect code blocks in AI response"""
        code_blocks = []
        
        # Detect code blocks with language specification
        code_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        for lang, code in matches:
            if len(code.strip()) > 50:  # Only cache substantial code
                code_blocks.append({
                    'language': lang or 'unknown',
                    'code': code.strip(),
                    'type': 'code_block'
                })
        
        # Detect inline code that looks like functions/classes
        inline_patterns = [
            r'(class\s+\w+.*?\{.*?\})',
            r'(fun\s+\w+\(.*?\)\s*\{.*?\})',
            r'(public\s+\w+\s+\w+\(.*?\)\s*\{.*?\})',
            r'(private\s+\w+\s+\w+\(.*?\)\s*\{.*?\})',
        ]
        
        for pattern in inline_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                if len(match.strip()) > 30:
                    # Determine language from syntax
                    if 'fun ' in match:
                        lang = 'kotlin'
                    elif 'public ' in match or 'private ' in match:
                        lang = 'java'
                    else:
                        lang = 'unknown'
                    
                    code_blocks.append({
                        'language': lang,
                        'code': match.strip(),
                        'type': 'inline_function'
                    })
        
        return code_blocks
    
    def cache_code(self, code_blocks: List[Dict[str, str]], context: str = "") -> List[str]:
        """Cache generated code blocks and return file paths"""
        cached_files = []
        timestamp = int(time.time())
        
        for i, block in enumerate(code_blocks):
            # Generate filename
            lang = block['language']
            code_type = block['type']
            
            # Extract function/class name if possible
            code = block['code']
            name_match = re.search(r'(?:class|fun|function)\s+(\w+)', code)
            name = name_match.group(1) if name_match else f"generated_{i}"
            
            # Determine file extension
            ext_map = {
                'kotlin': 'kt',
                'java': 'java',
                'javascript': 'js',
                'python': 'py',
                'typescript': 'ts'
            }
            ext = ext_map.get(lang, 'txt')
            
            filename = f"{timestamp}_{name}.{ext}"
            filepath = self.cache_dir / filename
            
            # Create code file with metadata
            full_content = f"""/*
 * Generated Code - AAPS Eating Now Plugin Development
 * Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
 * Language: {lang}
 * Type: {code_type}
 * Context: {context[:100]}...
 */

{code}
"""
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(full_content)
                
                cached_files.append(str(filepath))
                
                # Also store in session cache for quick access
                self.session_cache[filename] = {
                    'code': code,
                    'language': lang,
                    'type': code_type,
                    'context': context,
                    'timestamp': timestamp
                }
                
                logger.info(f"💾 Cached generated code: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to cache code: {e}")
        
        return cached_files
    
    def get_cached_code_summary(self) -> str:
        """Get summary of cached code for next session"""
        if not self.session_cache:
            return ""
        
        summary_parts = ["\n🔧 GENERATED CODE FROM THIS SESSION:"]
        for filename, info in self.session_cache.items():
            lang = info['language']
            code_type = info['type']
            preview = info['code'][:100].replace('\n', ' ')
            summary_parts.append(f"   • {filename} ({lang}, {code_type}): {preview}...")
        
        summary_parts.append("\n💡 These files are available in ./generated_code_cache/")
        return "\n".join(summary_parts)


class EnhancedNeo4jRAGRetriever:
    """Enhanced RAG retriever with source code access and eating now prioritization"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.available_repos = self._get_available_repositories()
        self.fulltext_available = self._test_fulltext_availability()
        logger.info(f"Connected to enhanced Neo4j. Available repositories: {self.available_repos}")
    
    def close(self):
        self.driver.close()
    
    def _test_fulltext_availability(self) -> bool:
        """Test if full-text search is actually working"""
        try:
            with self.driver.session() as session:
                # Test with a simple search that should return results
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('file_source_idx', 'bolus') 
                    YIELD node 
                    RETURN count(node) as count
                    LIMIT 1
                """)
                
                record = result.single()
                count = record["count"] if record else 0
                
                if count > 0:
                    logger.info(f"✅ Full-text search confirmed working: {count} results for 'bolus'")
                    return True
                else:
                    logger.warning("⚠️  Full-text index exists but returns no results")
                    return False
                    
        except Exception as e:
            logger.warning(f"⚠️  Full-text search test failed: {e}")
            return False

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
    
    def get_eating_now_focused_overview(self) -> Dict[str, Any]:
        """Get database overview with strong eating now focus"""
        overview = {}
        
        # Top eating now files globally
        top_eating_now = self.execute_query("""
            MATCH (f:File)
            WHERE f.eating_now_score > 50
            RETURN f.name as filename, 
                   f.repository as repository,
                   f.eating_now_score as eating_now_score,
                   f.importance_score as importance,
                   f.package as package,
                   f.function_count as functions,
                   f.has_source_code as has_source,
                   substring(f.source_code, 0, 300) as source_preview
            ORDER BY f.eating_now_score DESC
            LIMIT 15
        """)
        overview['top_eating_now_files'] = top_eating_now
        
        # Eating now repositories comparison
        repo_comparison = self.execute_query("""
            MATCH (r:Repository)
            WHERE r.is_eating_now_repo = true
            RETURN r.name as repository,
                   r.avg_eating_now_score as avg_eating_now,
                   r.file_count as files,
                   r.total_functions as functions,
                   r.files_with_source_code as files_with_source
            ORDER BY r.avg_eating_now_score DESC
        """)
        overview['eating_now_repositories'] = repo_comparison
        
        # Key eating now packages
        eating_packages = self.execute_query("""
            MATCH (f:File)
            WHERE f.eating_now_score > 100 AND f.package IS NOT NULL
            WITH f.package as package, 
                 f.repository as repository,
                 count(f) as file_count,
                 avg(f.eating_now_score) as avg_score,
                 collect(f.name)[0..3] as sample_files
            RETURN package, repository, file_count, avg_score, sample_files
            ORDER BY avg_score DESC
            LIMIT 10
        """)
        overview['eating_now_packages'] = eating_packages
        
        return overview
    
    def search_eating_now_source_code(self, keywords: List[str], repository: str = None, limit: int = 12) -> List[Dict]:
        """Enhanced search prioritizing eating now files with source code - FIXED VERSION"""
        search_terms = [term.lower() for term in keywords]
        
        # Build repository filter
        repo_filter = ""
        if repository and repository in self.available_repos:
            repo_filter = f"AND node.repository = '{repository}'"
        
        # Try full-text search first with better error handling
        fulltext_query = f"""
        CALL db.index.fulltext.queryNodes('file_source_idx', $search_string) YIELD node, score
        WHERE node.eating_now_score >= 0 {repo_filter}
        RETURN node.name as filename,
               node.repository as repository,
               node.package as package,
               node.eating_now_score as eating_now_score,
               node.importance_score as importance,
               node.source_code as source_code,
               node.key_snippets as key_snippets,
               node.functions as functions,
               node.has_source_code as has_source,
               score * (1 + node.eating_now_score * 0.01) as weighted_score
        ORDER BY weighted_score DESC, node.eating_now_score DESC
        LIMIT {limit}
        """
        
        # Try different search string formats
        search_variations = [
            " OR ".join(search_terms[:5]),      # OR search (most likely to succeed)
            " AND ".join(search_terms[:3]),     # AND search (more restrictive)
            search_terms[0] if search_terms else "bolus"  # Single term fallback
        ]
        
        for search_string in search_variations:
            try:
                logger.info(f"Trying full-text search with: '{search_string}'")
                fulltext_results = self.execute_query(fulltext_query, {"search_string": search_string})
                
                if fulltext_results:
                    logger.info(f"✅ Full-text search successful: {len(fulltext_results)} results")
                    return fulltext_results
                else:
                    logger.info(f"Full-text search returned no results for: '{search_string}'")
                    
            except Exception as e:
                logger.warning(f"Full-text search failed for '{search_string}': {e}")
                continue
        
        # Only fall back to property search if ALL full-text attempts failed
        logger.info("Full-text search not available, using property-based search fallback")
        
        property_search = f"""
        MATCH (f:File)
        WHERE ANY(term IN $search_terms WHERE 
                  toLower(f.name) CONTAINS term OR 
                  toLower(f.path) CONTAINS term OR 
                  toLower(f.package) CONTAINS term OR
                  (f.source_code IS NOT NULL AND toLower(f.source_code) CONTAINS term))
        {repo_filter.replace('node.', 'f.')}
        RETURN f.name as filename,
               f.repository as repository,
               f.package as package,
               f.eating_now_score as eating_now_score,
               f.importance_score as importance,
               f.source_code as source_code,
               f.key_snippets as key_snippets,
               f.functions as functions,
               f.has_source_code as has_source,
               f.eating_now_score as weighted_score
        ORDER BY f.eating_now_score DESC, f.importance_score DESC
        LIMIT {limit}
        """
        
        property_results = self.execute_query(property_search, {"search_terms": search_terms})
        logger.info(f"✅ Property-based search returned: {len(property_results)} results")
        return property_results
    
    def get_eating_now_source_code(self, filename: str, repository: str = None) -> Dict[str, Any]:
        """Get complete source code for eating now files"""
        where_parts = ["f.name = $filename"]
        if repository and repository in self.available_repos:
            where_parts.append(f"f.repository = '{repository}'")
        
        where_clause = " AND ".join(where_parts)
        
        query = f"""
        MATCH (f:File)
        WHERE {where_clause}
        OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
        WHERE out.repository = f.repository AND target.eating_now_score > 0
        OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
        WHERE in.repository = f.repository AND source.eating_now_score > 0
        WITH f, 
             collect(DISTINCT {{name: target.name, eating_score: target.eating_now_score}}) as calls_to_eating_now,
             collect(DISTINCT {{name: source.name, eating_score: source.eating_now_score}}) as called_by_eating_now
        RETURN f.name as filename,
               f.repository as repository,
               f.package as package,
               f.eating_now_score as eating_now_score,
               f.importance_score as importance,
               f.source_code as source_code,
               f.key_snippets as key_snippets,
               f.functions as functions,
               f.classes as classes,
               f.imports as imports,
               f.has_source_code as has_source,
               calls_to_eating_now,
               called_by_eating_now
        """
        
        result = self.execute_query(query, {"filename": filename}, limit=1)
        return result[0] if result else {}
    
    def find_eating_now_templates(self, functionality: str = "eating", include_source: bool = True) -> List[Dict]:
        """Find eating now plugin templates with full source code"""
        functionality_terms = {
            'eating': ['eating', 'eatnow', 'meal', 'food'],
            'bolus': ['bolus', 'dose', 'insulin'],
            'carb': ['carb', 'carbohydrate', 'nutrition'],
            'calculation': ['calc', 'algorithm', 'compute', 'formula'],
            'plugin': ['plugin', 'extension', 'module']
        }
        
        terms = functionality_terms.get(functionality.lower(), [functionality])
        
        query = """
        MATCH (f:File)
        WHERE f.eating_now_score > 80
        AND (ANY(term IN $terms WHERE toLower(f.name) CONTAINS term)
             OR ANY(term IN $terms WHERE toLower(f.package) CONTAINS term)
             OR f.is_eating_now_critical = true)
        OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
        WHERE out.repository = f.repository
        OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
        WHERE in.repository = f.repository
        WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
        RETURN f.name as filename,
               f.repository as repository,
               f.package as package,
               f.eating_now_score as eating_now_score,
               f.importance_score as importance,
               f.source_code as source_code,
               f.key_snippets as key_snippets,
               f.functions as functions,
               f.classes as classes,
               f.has_source_code as has_source,
               (outgoing + incoming) as connections,
               f.lines_of_code as loc
        ORDER BY f.eating_now_score DESC, connections DESC
        LIMIT 8
        """
        
        return self.execute_query(query, {"terms": terms})


class EnhancedOllamaClient:
    """Enhanced Ollama client with better prompting for eating now context"""
    
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
    
    def generate_eating_now_response(self, prompt: str, context: str = "", cached_code_summary: str = "", max_tokens: int = 3072) -> str:
        """Generate response with enhanced eating now context and code awareness"""
        # Truncate context if it's too long
        max_context_length = 15000  # Increased for source code
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n\n[Context truncated for length...]"
        
        full_prompt = f"""You are an expert AAPS (AndroidAPS) plugin developer specializing in EATING NOW functionality. You have access to the complete source code of three repositories:

1. **EN_new** - Latest EatingNow variant (HIGHEST PRIORITY)
2. **EN_old** - Older EatingNow variant 
3. **AAPS_source** - Main AndroidAPS source code

**EATING NOW CONTEXT (PRIORITIZED):**
{context}

{cached_code_summary}

**User Question:** {prompt}

**EXPERT INSTRUCTIONS:**
- PRIORITIZE eating now functionality and files with high eating_now_scores
- When showing code examples, use ACTUAL source code from the database
- For plugin development questions, provide complete, working code based on existing patterns
- Always specify which repository (EN_new, EN_old, AAPS_source) code comes from
- Focus on bolus calculation, carb counting, meal timing, and insulin dosing
- Provide concrete file names, class names, and function names from the actual codebase
- If generating new code, base it closely on existing eating now implementations
- Use proper Kotlin/Java syntax following AAPS patterns

**RESPONSE GUIDELINES:**
- Start with the most relevant eating now files/functions
- Include actual source code snippets when helpful
- Explain how code relates to eating now functionality
- Provide step-by-step implementation guidance
- Mention specific packages and imports needed

**Response:**"""
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Low temperature for more factual, code-focused responses
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": max_tokens
                    }
                },
                timeout=240  # 4 minute timeout for complex code generation
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error: Ollama API returned status {response.status_code}"
                
        except requests.RequestException as e:
            return f"Error communicating with Ollama: {e}"


class EnhancedAAPSRAGSystem:
    """Enhanced RAG system with eating now focus, source code access, and code caching"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 ollama_url: str = "http://localhost:11434", model: str = "deepseek-r1:1.5b"):
        self.retriever = EnhancedNeo4jRAGRetriever(neo4j_uri, neo4j_user, neo4j_password)
        self.ollama = EnhancedOllamaClient(ollama_url, model)
        self.code_cache = CodeCache()
        self.available_repos = self.retriever.available_repos
        
    def close(self):
        self.retriever.close()
    
    def extract_eating_now_keywords(self, question: str) -> List[str]:
        """Enhanced keyword extraction prioritizing eating now terms"""
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
        
        # SUPER BOOST eating now terms
        eating_now_terms = {
            'eating', 'eatnow', 'eatingnow', 'bolus', 'carb', 'carbs', 'carbohydrate',
            'meal', 'food', 'nutrition', 'insulin', 'dose', 'dosing', 'calculation',
            'treatment', 'therapy', 'algorithm', 'plugin', 'glucose'
        }
        
        # Prioritize eating now terms first
        prioritized = []
        for word in keywords:
            if word in eating_now_terms:
                prioritized.insert(0, word)  # Put eating now terms first
            else:
                prioritized.append(word)
        
        return prioritized[:15]  # Return top 15 keywords
    
    def detect_code_generation_request(self, question: str) -> bool:
        """Detect if user is asking for code generation"""
        code_indicators = [
            'generate', 'create', 'write', 'build', 'implement', 'develop',
            'code', 'plugin', 'class', 'function', 'method', 'algorithm',
            'example', 'template', 'sample', 'show me how', 'help me write'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in code_indicators)
    
    def retrieve_eating_now_context(self, question: str) -> str:
        """Enhanced context retrieval with strong eating now focus"""
        keywords = self.extract_eating_now_keywords(question)
        target_repo = self._detect_repository_context(question)
        is_code_request = self.detect_code_generation_request(question)
        
        context_parts = []
        
        # Add repository context
        if target_repo:
            context_parts.append(f"🎯 FOCUSED ON REPOSITORY: {target_repo}")
        else:
            context_parts.append(f"📚 EATING NOW REPOSITORIES: {', '.join([r for r in self.available_repos if 'EN' in r])}")
        
        # Get eating now overview first
        overview = self.retriever.get_eating_now_focused_overview()
        self._add_eating_now_overview_context(context_parts, overview, target_repo)
        
        # Get relevant source code
        if keywords:
            relevant_files = self.retriever.search_eating_now_source_code(keywords, target_repo, 8)
            if relevant_files:
                self._add_source_code_context(context_parts, relevant_files, is_code_request)
        
        # Get eating now templates if this is a code generation request
        if is_code_request:
            for keyword in keywords[:3]:  # Top 3 keywords
                if keyword in ['eating', 'bolus', 'carb', 'calculation', 'plugin']:
                    templates = self.retriever.find_eating_now_templates(keyword)
                    if templates:
                        self._add_template_context(context_parts, keyword, templates)
        
        return "\n".join(context_parts)
    
    def _detect_repository_context(self, question: str) -> Optional[str]:
        """Enhanced repository detection with eating now priority"""
        question_lower = question.lower()
        
        # Direct repository mentions
        for repo in self.available_repos:
            if repo.lower() in question_lower:
                return repo
        
        # Contextual clues - prioritize eating now repos
        if any(term in question_lower for term in ['en_new', 'new eating', 'latest eating', 'new version']):
            return 'EN_new'
        elif any(term in question_lower for term in ['en_old', 'old eating', 'previous eating', 'original eating']):
            return 'EN_old'
        elif any(term in question_lower for term in ['aaps_source', 'main aaps', 'base aaps', 'source']):
            return 'AAPS_source'
        
        # Default to EN_new for eating now questions
        eating_indicators = ['eating', 'bolus', 'carb', 'meal', 'food']
        if any(term in question_lower for term in eating_indicators):
            return 'EN_new'  # Default to latest eating now implementation
        
        return None
    
    def _add_eating_now_overview_context(self, context_parts: List[str], overview: Dict, target_repo: str):
        """Add eating now overview context"""
        context_parts.append("\n🍽️ EATING NOW OVERVIEW (HIGHEST PRIORITY):")
        
        # Top eating now files
        if overview.get('top_eating_now_files'):
            context_parts.append("\n🔥 TOP EATING NOW FILES:")
            for file_info in overview['top_eating_now_files'][:6]:
                repo = file_info.get('repository', 'unknown')
                score = file_info.get('eating_now_score', 0)
                has_source = file_info.get('has_source', False)
                source_indicator = "💾" if has_source else ""
                context_parts.append(f"   • {source_indicator}{file_info['filename']} ({repo}): Score {score:.1f}")
                # Add source preview
                preview = file_info.get('source_preview', '')
                if preview:
                    cleaned_preview = preview.replace('\n', ' ').strip()
                    context_parts.append(f"     Preview: {cleaned_preview}...")
        
        # Eating now repositories
        if overview.get('eating_now_repositories'):
            context_parts.append("\n📚 EATING NOW REPOSITORIES:")
            for repo_info in overview['eating_now_repositories']:
                files_with_source = repo_info.get('files_with_source', 0)
                context_parts.append(f"   • {repo_info['repository']}: Avg Score {repo_info['avg_eating_now']:.1f}, "
                                   f"{repo_info['files']} files, {files_with_source} with source")
        
        # Key packages
        if overview.get('eating_now_packages'):
            context_parts.append("\n📦 KEY EATING NOW PACKAGES:")
            for pkg in overview['eating_now_packages'][:4]:
                context_parts.append(f"   • {pkg['package']} ({pkg['repository']}): {pkg['file_count']} files, Score {pkg['avg_score']:.1f}")
    
    def _add_source_code_context(self, context_parts: List[str], relevant_files: List[Dict], include_full_source: bool = False):
        """Add source code context with actual code"""
        context_parts.append("\n💾 RELEVANT SOURCE CODE:")
        
        for file_info in relevant_files[:5]:  # Top 5 files
            repo = file_info.get('repository', 'unknown')
            score = file_info.get('eating_now_score', 0)
            filename = file_info['filename']
            has_source = file_info.get('has_source', False)
            
            context_parts.append(f"\n📄 {filename} ({repo}) - Eating Score: {score:.1f} {'💾' if has_source else ''}")
            context_parts.append(f"   Package: {file_info.get('package', 'unknown')}")
            
            # Add functions list
            functions = file_info.get('functions', [])
            if functions:
                context_parts.append(f"   Functions: {', '.join(functions[:5])}")
            
            # Add source code preview or snippets
            if include_full_source and has_source:
                source_code = file_info.get('source_code', '')
                if source_code:
                    # Show first 1000 characters of source code
                    preview = source_code[:1000]
                    context_parts.append(f"   Source Code Preview:\n{preview}")
                    if len(source_code) > 1000:
                        context_parts.append("   [... source code continues ...]")
            
            # Add key snippets
            key_snippets = file_info.get('key_snippets', '')
            if key_snippets:
                try:
                    snippets = json.loads(key_snippets) if isinstance(key_snippets, str) else key_snippets
                    if snippets:
                        context_parts.append("   Key Code Snippets:")
                        for snippet_name, snippet_code in list(snippets.items())[:2]:  # Show top 2 snippets
                            context_parts.append(f"     {snippet_name}:")
                            lines = snippet_code.split('\n')[:4]  # First 4 lines
                            for line in lines:
                                if line.strip():
                                    context_parts.append(f"       {line}")
                except:
                    pass
    
    def _add_template_context(self, context_parts: List[str], keyword: str, templates: List[Dict]):
        """Add plugin template context with source code"""
        context_parts.append(f"\n🔧 {keyword.upper()} PLUGIN TEMPLATES:")
        
        for template in templates[:3]:  # Top 3 templates
            repo = template.get('repository', 'unknown')
            score = template.get('eating_now_score', 0)
            has_source = template.get('has_source', False)
            
            context_parts.append(f"\n   📄 {template['filename']} ({repo}) - Score: {score:.1f} {'💾' if has_source else ''}")
            context_parts.append(f"      Package: {template.get('package', 'unknown')}")
            context_parts.append(f"      Functions: {template.get('functions', 0)}, LOC: {template.get('loc', 0)}")
            
            # Add actual source code for templates
            if has_source:
                source_code = template.get('source_code', '')
                if source_code:
                    # Show class declaration and first method
                    lines = source_code.split('\n')
                    in_class = False
                    shown_lines = 0
                    context_parts.append("      Template Code:")
                    
                    for line in lines:
                        if shown_lines >= 15:  # Limit template preview
                            context_parts.append("      [... template continues ...]")
                            break
                        
                        if 'class ' in line or 'interface ' in line or in_class:
                            in_class = True
                            context_parts.append(f"        {line}")
                            shown_lines += 1
                            
                            if line.strip().endswith('}') and shown_lines > 5:
                                break
    
    def answer_question_enhanced(self, question: str) -> str:
        """Answer question with enhanced eating now focus and code caching"""
        logger.info(f"Processing eating now focused question: {question}")
        
        # Retrieve enhanced context with source code
        context = self.retrieve_eating_now_context(question)
        
        if not context.strip():
            return "I couldn't find relevant eating now information in the project database for your question."
        
        logger.info(f"Retrieved eating now context length: {len(context)} characters")
        
        # Get cached code summary
        cached_summary = self.code_cache.get_cached_code_summary()
        
        # Generate response using enhanced Ollama client
        response = self.ollama.generate_eating_now_response(question, context, cached_summary)
        
        # Detect and cache any generated code
        code_blocks = self.code_cache.detect_code_in_response(response)
        if code_blocks:
            cached_files = self.code_cache.cache_code(code_blocks, context=question[:200])
            if cached_files:
                response += f"\n\n💾 Generated code has been cached to:\n"
                for file_path in cached_files:
                    response += f"   • {file_path}\n"
        
        return response
    
    def interactive_mode(self):
        """Enhanced interactive Q&A session with eating now focus"""
        print(f"🤖 AAPS Enhanced Multi-Repository RAG System")
        print(f"🍽️ EATING NOW PRIORITIZED WITH SOURCE CODE ACCESS")
        print(f"🧠 Model: {self.ollama.model}")
        print(f"📚 Available repositories: {', '.join(self.available_repos)}")
        print(f"💾 Code caching: Enabled (./generated_code_cache/)")
        print(f"💡 Focused on eating now functionality and plugin development")
        print(f"\nType 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                question = input("🍽️ Ask about AAPS eating now: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    self.show_enhanced_help()
                    continue
                elif question.lower() == 'stats':
                    self.show_eating_now_stats()
                    continue
                elif question.lower() == 'templates':
                    self.show_templates()
                    continue
                elif question.lower() == 'cache':
                    self.show_cached_code()
                    continue
                elif question.lower().startswith('source '):
                    filename = question[7:].strip()
                    self.show_source_code(filename)
                    continue
                elif not question:
                    continue
                
                print("🔍 Searching eating now database with source code...")
                
                # Show detected context
                target_repo = self._detect_repository_context(question)
                is_code_request = self.detect_code_generation_request(question)
                keywords = self.extract_eating_now_keywords(question)
                
                print(f"🎯 Repository: {target_repo or 'All (eating now focused)'}")
                print(f"💻 Code generation: {'Yes' if is_code_request else 'No'}")
                print(f"🔑 Key eating now terms: {', '.join(keywords[:5])}")
                
                answer = self.answer_question_enhanced(question)
                print(f"\n🤖 {answer}\n")
                print("-" * 80)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Show session summary
        cached_summary = self.code_cache.get_cached_code_summary()
        if cached_summary:
            print(cached_summary)
        
        print("👋 Goodbye!")
    
    def show_enhanced_help(self):
        """Show enhanced help information"""
        help_text = f"""
🤖 AAPS Enhanced RAG System - Eating Now Focused Help

🍽️ EATING NOW PRIORITIZED:
   • All responses prioritize eating now functionality (EN_new, EN_old)
   • Source code from actual files is included in responses
   • Plugin development templates are automatically suggested

📚 Available Repositories (Eating Now Focused):
{chr(10).join([f'   • {repo} {"🍽️" if "EN" in repo else ""}' for repo in self.available_repos])}

💡 Smart Features:
   • Automatic eating now file prioritization
   • Complete source code access and search
   • Code generation with automatic caching
   • Plugin template recommendations

🎯 Question Types Optimized for Eating Now:
   • Plugin Development: "How do I create an eating now plugin for AAPS_source?"
   • Code Examples: "Show me bolus calculation code from EN_new"
   • Templates: "What eating now files can I use as templates?"
   • Source Code: "Get source code for BolusCalculatorPlugin.kt"
   • Implementation: "How does carb counting work in EN_old?"

📝 Commands:
   • 'stats' - Show eating now database statistics
   • 'templates' - Show available plugin templates
   • 'cache' - Show cached generated code
   • 'source <filename>' - Show source code for specific file
   • 'help' - Show this help
   • 'quit' - Exit

🔧 Code Generation Examples:
   • "Create a bolus calculation plugin based on EN_new"
   • "Generate carb counting functions using existing templates"
   • "Write an eating now timer plugin"
   • "Show me how to implement meal detection"

💾 Automatic Code Caching:
   • Generated code is automatically saved to ./generated_code_cache/
   • Each session's code is tracked and summarized
   • Files include metadata and context for easy reference

🌐 Eating Now Focused Queries:
   • Always mentions which eating now repository code comes from
   • Prioritizes files with high eating_now_scores
   • Includes actual working code from the database
   • Provides concrete implementation guidance
"""
        print(help_text)
    
    def show_eating_now_stats(self):
        """Show eating now focused statistics"""
        print("📊 Eating Now Database Statistics:")
        
        overview = self.retriever.get_eating_now_focused_overview()
        
        # Eating now repositories
        if overview.get('eating_now_repositories'):
            print(f"\n🍽️ Eating Now Repositories:")
            for repo in overview['eating_now_repositories']:
                files_with_source = repo.get('files_with_source', 0)
                print(f"   📦 {repo['repository']}:")
                print(f"      Files: {repo['files']:,}")
                print(f"      Functions: {repo['functions']:,}")
                print(f"      Avg Eating Now Score: {repo['avg_eating_now']:.1f}")
                print(f"      Files with Source Code: {files_with_source:,}")
        
        # Top eating now files
        if overview.get('top_eating_now_files'):
            print(f"\n🔥 Top Eating Now Files:")
            for i, file_info in enumerate(overview['top_eating_now_files'][:8], 1):
                repo = file_info.get('repository', 'unknown')
                score = file_info.get('eating_now_score', 0)
                has_source = file_info.get('has_source', False)
                source_indicator = "💾" if has_source else ""
                print(f"   {i:2d}. {source_indicator}{file_info['filename']} ({repo}): {score:.1f}")
        
        # Key packages
        if overview.get('eating_now_packages'):
            print(f"\n📦 Key Eating Now Packages:")
            for pkg in overview['eating_now_packages'][:5]:
                print(f"   • {pkg['package']} ({pkg['repository']}): {pkg['file_count']} files")
        
        print()
    
    def show_templates(self):
        """Show available plugin templates"""
        print("🔧 Available Eating Now Plugin Templates:")
        
        template_types = ['eating', 'bolus', 'carb', 'calculation']
        
        for template_type in template_types:
            templates = self.retriever.find_eating_now_templates(template_type)
            if templates:
                print(f"\n🍽️ {template_type.upper()} Templates:")
                for template in templates[:3]:
                    repo = template.get('repository', 'unknown')
                    score = template.get('eating_now_score', 0)
                    has_source = template.get('has_source', False)
                    source_indicator = "💾" if has_source else ""
                    print(f"   • {source_indicator}{template['filename']} ({repo}): Score {score:.1f}")
                    print(f"     Package: {template.get('package', 'unknown')}")
                    print(f"     Functions: {template.get('functions', 0)}, LOC: {template.get('loc', 0)}")
        
        print(f"\n💡 Use these files as templates for your eating now plugin development!")
        print()
    
    def show_cached_code(self):
        """Show cached generated code from this session"""
        if not self.code_cache.session_cache:
            print("💾 No code has been generated and cached in this session.")
            return
        
        print("💾 Generated Code Cache (This Session):")
        print("="*50)
        
        for filename, info in self.code_cache.session_cache.items():
            lang = info['language']
            code_type = info['type']
            timestamp = time.strftime('%H:%M:%S', time.localtime(info['timestamp']))
            
            print(f"\n📄 {filename}")
            print(f"   Language: {lang}")
            print(f"   Type: {code_type}")
            print(f"   Generated: {timestamp}")
            print(f"   Context: {info['context'][:100]}...")
            
            # Show code preview
            code_lines = info['code'].split('\n')
            print(f"   Code Preview:")
            for line in code_lines[:5]:
                if line.strip():
                    print(f"     {line}")
            if len(code_lines) > 5:
                print("     [... more code ...]")
        
        print(f"\n📁 All files saved to: {self.code_cache.cache_dir}")
        print()
    
    def show_source_code(self, filename: str):
        """Show source code for a specific file"""
        print(f"💾 Retrieving source code for: {filename}")
        
        source_data = self.retriever.get_eating_now_source_code(filename)
        
        if not source_data:
            print(f"❌ File '{filename}' not found in eating now database.")
            return
        
        print(f"\n💾 SOURCE CODE: {source_data['filename']}")
        print("="*60)
        print(f"📦 Repository: {source_data.get('repository', 'unknown')}")
        print(f"📁 Package: {source_data.get('package', 'unknown')}")
        print(f"🍽️ Eating Now Score: {source_data.get('eating_now_score', 0):.1f}")
        print(f"⭐ Importance: {source_data.get('importance', 0):.1f}")
        print(f"💾 Has Source Code: {source_data.get('has_source', False)}")
        print("="*60)
        
        source_code = source_data.get('source_code', '')
        if source_code:
            # Show first 50 lines
            lines = source_code.split('\n')
            for i, line in enumerate(lines[:50], 1):
                print(f"{i:3d}: {line}")
            
            if len(lines) > 50:
                print(f"\n[... showing first 50 of {len(lines)} lines ...]")
                print("Full source code is available in the database.")
        else:
            print("❌ No source code available for this file.")
        
        # Show eating now connections
        calls_to = source_data.get('calls_to_eating_now', [])
        called_by = source_data.get('called_by_eating_now', [])
        
        if calls_to:
            print(f"\n🔗 Calls to eating now files:")
            for call in calls_to[:5]:
                print(f"   → {call['name']} (score: {call['eating_score']:.1f})")
        
        if called_by:
            print(f"\n🔗 Called by eating now files:")
            for call in called_by[:5]:
                print(f"   ← {call['name']} (score: {call['eating_score']:.1f})")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="AAPS Enhanced Multi-Repository RAG System")
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
    parser.add_argument("--cache-dir", default="./generated_code_cache",
                       help="Directory for caching generated code")
    
    args = parser.parse_args()
    
    print("🚀 AAPS Enhanced Multi-Repository RAG System")
    print("🍽️ EATING NOW PRIORITIZED WITH SOURCE CODE ACCESS")
    print("💾 AUTOMATIC CODE GENERATION AND CACHING")
    print("="*70)
    
    # Initialize enhanced RAG system
    try:
        rag_system = EnhancedAAPSRAGSystem(
            args.neo4j_uri, args.neo4j_user, args.neo4j_password,
            args.ollama_url, args.model
        )
        
        # Set custom cache directory if specified
        if args.cache_dir != "./generated_code_cache":
            rag_system.code_cache = CodeCache(args.cache_dir)
        
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("Make sure Neo4j is running and the Enhanced Analyzer has been run")
        sys.exit(1)
    
    # Check available repositories
    if not rag_system.available_repos:
        print("❌ No repositories found in database")
        print("Make sure the Enhanced Analyzer (aaps_analyzer.py) has been run")
        sys.exit(1)
    
    print(f"✅ Connected to enhanced database with repositories: {', '.join(rag_system.available_repos)}")
    
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
    print(f"💾 Code caching enabled: {rag_system.code_cache.cache_dir}")
    
    # Single question mode or interactive mode
    if args.question:
        if args.repository:
            # Modify question to include repository context
            args.question = f"In repository {args.repository}: {args.question}"
        
        print(f"\nEating Now Question: {args.question}")
        print("="*70)
        answer = rag_system.answer_question_enhanced(args.question)
        print(answer)
        
        # Show any cached code
        cached_summary = rag_system.code_cache.get_cached_code_summary()
        if cached_summary:
            print(cached_summary)
    else:
        rag_system.interactive_mode()
    
    rag_system.close()

if __name__ == "__main__":
    main()
