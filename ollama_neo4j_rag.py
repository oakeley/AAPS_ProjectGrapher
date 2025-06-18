#!/usr/bin/env python3
"""
Ollama Neo4j RAG System for AAPS EatingNow Project
Allows Ollama models to query the Neo4j database and answer questions based on actual project structure
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

class Neo4jRAGRetriever:
    """Retrieves relevant information from Neo4j database based on user queries"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return []
    
    def search_files_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """Search for files containing specific keywords"""
        search_terms = [term.lower() for term in keywords]
        
        query = """
        MATCH (f:File)
        WHERE ANY(term IN $search_terms WHERE 
                  toLower(f.name) CONTAINS term OR 
                  toLower(f.path) CONTAINS term)
        OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)
        OPTIONAL MATCH (f)-[:CONTAINS]->(c:Class)
        RETURN f.name as filename, f.path as path,
               f.lines_of_code as loc, f.function_count as func_count,
               collect(DISTINCT fn.name) as functions,
               collect(DISTINCT c.name) as classes
        ORDER BY f.function_count DESC
        LIMIT 20
        """
        
        return self.execute_query(query, {"search_terms": search_terms})
    
    def get_file_details(self, filename: str) -> Dict[str, Any]:
        """Get detailed information about a specific file"""
        query = """
        MATCH (f:File {name: $filename})
        OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)
        OPTIONAL MATCH (f)-[:CONTAINS]->(c:Class)
        OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
        OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
        RETURN f.name as filename, f.path as path,
               f.lines_of_code as loc, f.function_count as func_count,
               collect(DISTINCT fn.name) as functions,
               collect(DISTINCT c.name) as classes,
               collect(DISTINCT {name: target.name, function: out.function}) as calls_to,
               collect(DISTINCT {name: source.name, function: in.function}) as called_by
        """
        
        result = self.execute_query(query, {"filename": filename})
        return result[0] if result else {}
    
    def find_related_files(self, concept: str, max_results: int = 10) -> List[Dict]:
        """Find files related to a concept (broader search)"""
        keywords = concept.lower().split()
        
        # Search in file names, paths, and function names
        query = """
        MATCH (f:File)
        WHERE ANY(term IN $keywords WHERE 
                  toLower(f.name) CONTAINS term OR 
                  toLower(f.path) CONTAINS term)
        OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)
        WHERE ANY(term IN $keywords WHERE toLower(fn.name) CONTAINS term)
        WITH f, collect(DISTINCT fn.name) as matching_functions
        OPTIONAL MATCH (f)-[out:CALLS]->()
        OPTIONAL MATCH ()-[in:CALLS]->(f)
        WITH f, matching_functions, count(out) + count(in) as connectivity
        RETURN f.name as filename, f.path as path,
               f.lines_of_code as loc, f.function_count as func_count,
               matching_functions, connectivity
        ORDER BY connectivity DESC, f.function_count DESC
        LIMIT $max_results
        """
        
        return self.execute_query(query, {"keywords": keywords, "max_results": max_results})
    
    def get_architecture_overview(self) -> Dict[str, Any]:
        """Get high-level architecture information"""
        queries = {
            "summary": """
                MATCH (f:File)
                RETURN count(f) as total_files,
                       sum(f.lines_of_code) as total_loc,
                       sum(f.function_count) as total_functions,
                       sum(f.class_count) as total_classes
            """,
            
            "critical_files": """
                MATCH (f:File)
                OPTIONAL MATCH (f)-[out:CALLS]->()
                OPTIONAL MATCH ()-[in:CALLS]->(f)
                WITH f, count(out) + count(in) as connections
                WHERE connections > 5
                RETURN f.name as filename, connections, f.function_count as functions
                ORDER BY connections DESC
                LIMIT 10
            """,
            
            "entry_points": """
                MATCH (f:File)
                OPTIONAL MATCH (f)-[out:CALLS]->()
                OPTIONAL MATCH ()-[in:CALLS]->(f)
                WITH f, count(out) as outgoing, count(in) as incoming
                WHERE incoming <= 2 AND outgoing >= 5
                RETURN f.name as filename, outgoing, incoming
                ORDER BY outgoing DESC
                LIMIT 5
            """
        }
        
        result = {}
        for key, query in queries.items():
            result[key] = self.execute_query(query)
        
        return result
    
    def trace_call_path(self, from_file: str, to_file: str, max_depth: int = 5) -> List[Dict]:
        """Find call paths between two files"""
        query = f"""
        MATCH (start:File {{name: $from_file}}), (end:File {{name: $to_file}})
        MATCH path = shortestPath((start)-[:CALLS*1..{max_depth}]->(end))
        WHERE start <> end
        RETURN [node in nodes(path) | node.name] as call_path,
               [rel in relationships(path) | rel.function] as functions_called,
               length(path) as path_length
        """
        
        return self.execute_query(query, {"from_file": from_file, "to_file": to_file})

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:1.5b"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return self.model in available_models
            return False
        except requests.RequestException:
            return False
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama"""
        full_prompt = f"""Context from AAPS EatingNow project database:
{context}

User question: {prompt}

Please answer based ONLY on the provided context from the project database. If the context doesn't contain relevant information, say so clearly. Focus on the actual project structure and relationships shown in the data."""
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more factual responses
                        "top_p": 0.9,
                        "top_k": 40
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Error: Ollama API returned status {response.status_code}"
                
        except requests.RequestException as e:
            return f"Error communicating with Ollama: {e}"

class AAPSRAGSystem:
    """Main RAG system combining Neo4j retrieval with Ollama generation"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 ollama_url: str = "http://localhost:11434", model: str = "deepseek-r1:1.5b"):
        self.retriever = Neo4jRAGRetriever(neo4j_uri, neo4j_user, neo4j_password)
        self.ollama = OllamaClient(ollama_url, model)
        
    def close(self):
        self.retriever.close()
    
    def extract_keywords(self, question: str) -> List[str]:
        """Extract relevant keywords from user question"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'is', 'are', 'how', 'what', 'where', 'when', 'why', 'who', 
                     'does', 'do', 'can', 'could', 'would', 'should', 'will', 'a', 'an', 
                     'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        # Extract words, convert to lowercase, and filter
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to most important keywords
    
    def retrieve_context(self, question: str) -> str:
        """Retrieve relevant context from Neo4j based on the question"""
        keywords = self.extract_keywords(question)
        context_parts = []
        
        # Get relevant files
        relevant_files = self.retriever.search_files_by_keywords(keywords)
        if relevant_files:
            context_parts.append("RELEVANT FILES:")
            for file_info in relevant_files[:5]:  # Top 5 most relevant
                functions = file_info.get('functions', [])
                func_str = f", functions: {functions[:5]}" if functions else ""
                context_parts.append(
                    f"- {file_info['filename']}: {file_info['func_count']} functions, "
                    f"{file_info['loc']} LOC{func_str}"
                )
        
        # For specific file questions, get detailed info
        file_mentions = [word for word in keywords if word.endswith('.java') or word.endswith('.kt')]
        for file_mention in file_mentions:
            file_details = self.retriever.get_file_details(file_mention)
            if file_details:
                context_parts.append(f"\nDETAILS FOR {file_mention}:")
                context_parts.append(f"Path: {file_details['path']}")
                context_parts.append(f"Functions: {file_details.get('functions', [])}")
                context_parts.append(f"Classes: {file_details.get('classes', [])}")
                if file_details.get('calls_to'):
                    context_parts.append(f"Calls to: {[call['name'] for call in file_details['calls_to'][:5]]}")
        
        # Get architecture overview for general questions
        if any(word in keywords for word in ['architecture', 'structure', 'overview', 'main', 'entry']):
            arch_info = self.retriever.get_architecture_overview()
            context_parts.append(f"\nPROJECT OVERVIEW:")
            summary = arch_info['summary'][0] if arch_info['summary'] else {}
            context_parts.append(f"Total files: {summary.get('total_files', 'N/A')}")
            context_parts.append(f"Total functions: {summary.get('total_functions', 'N/A')}")
            context_parts.append(f"Total LOC: {summary.get('total_loc', 'N/A')}")
            
            if arch_info['critical_files']:
                context_parts.append("Critical files (most connected):")
                for file_info in arch_info['critical_files'][:3]:
                    context_parts.append(f"- {file_info['filename']}: {file_info['connections']} connections")
        
        # Search for concept-related files
        for keyword in keywords:
            if len(keyword) > 4:  # Only for substantial keywords
                related_files = self.retriever.find_related_files(keyword, 3)
                if related_files:
                    context_parts.append(f"\nFILES RELATED TO '{keyword.upper()}':")
                    for file_info in related_files:
                        context_parts.append(f"- {file_info['filename']}: {file_info['func_count']} functions")
        
        return "\n".join(context_parts)
    
    def answer_question(self, question: str) -> str:
        """Answer a question using RAG approach"""
        logger.info(f"Processing question: {question}")
        
        # Retrieve relevant context
        context = self.retrieve_context(question)
        
        if not context.strip():
            return "I couldn't find relevant information in the project database for your question."
        
        logger.info(f"Retrieved context length: {len(context)} characters")
        
        # Generate response using Ollama
        response = self.ollama.generate_response(question, context)
        
        return response
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print(f"🤖 AAPS EatingNow RAG System")
        print(f"Model: {self.ollama.model}")
        print(f"Type 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                question = input("❓ Ask about the AAPS project: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    self.show_help()
                    continue
                elif question.lower() == 'stats':
                    self.show_stats()
                    continue
                elif not question:
                    continue
                
                print("🔍 Searching project database...")
                answer = self.answer_question(question)
                print(f"\n🤖 {answer}\n")
                print("-" * 80)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")
    
    def show_help(self):
        """Show help information"""
        help_text = """
Available commands:
- Ask any question about the AAPS EatingNow project
- 'stats' - Show project statistics
- 'help' - Show this help
- 'quit' - Exit

Example questions:
- "What files handle blood glucose calculations?"
- "How does the insulin algorithm work?"
- "What are the main entry points?"
- "Show me files related to pump communication"
- "What functions are in MainActivity.kt?"
"""
        print(help_text)
    
    def show_stats(self):
        """Show project statistics"""
        arch_info = self.retriever.get_architecture_overview()
        summary = arch_info['summary'][0] if arch_info['summary'] else {}
        
        print(f"\n📊 AAPS Project Statistics:")
        print(f"Total files: {summary.get('total_files', 'N/A')}")
        print(f"Total functions: {summary.get('total_functions', 'N/A')}")
        print(f"Total classes: {summary.get('total_classes', 'N/A')}")
        print(f"Total lines of code: {summary.get('total_loc', 'N/A')}")
        
        if arch_info['critical_files']:
            print(f"\nMost connected files:")
            for file_info in arch_info['critical_files'][:5]:
                print(f"- {file_info['filename']}: {file_info['connections']} connections")
        print()

def main():
    parser = argparse.ArgumentParser(description="AAPS EatingNow RAG System with Ollama")
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
    
    args = parser.parse_args()
    
    # Initialize RAG system
    try:
        rag_system = AAPSRAGSystem(
            args.neo4j_uri, args.neo4j_user, args.neo4j_password,
            args.ollama_url, args.model
        )
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("Make sure Neo4j is running and credentials are correct")
        sys.exit(1)
    
    # Check Ollama availability
    if not rag_system.ollama.is_available():
        print(f"❌ Model '{args.model}' not available in Ollama")
        print("Available models:")
        try:
            response = requests.get(f"{args.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    print(f"  - {model['name']}")
            else:
                print("  Could not retrieve model list")
        except:
            print("  Ollama may not be running")
        sys.exit(1)
    
    # Single question mode or interactive mode
    if args.question:
        answer = rag_system.answer_question(args.question)
        print(answer)
    else:
        rag_system.interactive_mode()
    
    rag_system.close()

if __name__ == "__main__":
    main()
