#!/usr/bin/env python3
"""
Enhanced Multi-Repository Neo4j Query Utilities for AAPS Projects
WITH SOURCE CODE STORAGE AND EATING NOW PRIORITIZATION
Works with the Enhanced Analyzer database structure
Provides source code access for better RAG performance
"""

import json
from typing import Dict, List, Any, Optional
import logging
import sys
import re

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("❌ Neo4j driver not available. Install with: pip install neo4j")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMultiRepoQueries:
    """Enhanced multi-repository Neo4j queries with source code access and eating now focus"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.available_repos = self._get_available_repositories()
        
    def close(self):
        self.driver.close()
    
    def _get_available_repositories(self) -> List[str]:
        """Get available repositories from database"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (r:Repository) RETURN r.name as name ORDER BY r.name")
                return [record["name"] for record in result]
        except Exception as e:
            logger.warning(f"Could not get repositories: {e}")
            return []
    
    def execute_query_safe(self, query: str, parameters: Dict = None, limit: int = 25) -> List[Dict]:
        """Execute a query safely with memory limits"""
        try:
            with self.driver.session() as session:
                if "LIMIT" not in query.upper() and "COUNT" not in query.upper():
                    query += f" LIMIT {limit}"
                
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_enhanced_database_overview(self) -> Dict[str, Any]:
        """Get comprehensive database overview with eating now metrics"""
        stats = {}
        
        # Repository counts and details with eating now focus
        repo_stats = self.execute_query_safe("""
            MATCH (r:Repository)
            RETURN r.name as repository, 
                   r.file_count as files,
                   r.total_loc as loc,
                   r.total_functions as functions,
                   r.avg_eating_now_score as avg_eating_now,
                   r.is_eating_now_repo as is_eating_now_repo,
                   r.files_with_source_code as files_with_source
            ORDER BY r.avg_eating_now_score DESC
        """)
        stats['repositories'] = repo_stats
        
        # Global counts with eating now metrics
        global_stats = self.execute_query_safe("""
            MATCH (f:File)
            RETURN count(f) as total_files,
                   sum(f.lines_of_code) as total_loc,
                   sum(f.function_count) as total_functions,
                   avg(f.importance_score) as avg_importance,
                   avg(f.eating_now_score) as avg_eating_now,
                   max(f.eating_now_score) as max_eating_now,
                   count(CASE WHEN f.eating_now_score > 100 THEN 1 END) as critical_eating_now_files,
                   count(CASE WHEN f.has_source_code = true THEN 1 END) as files_with_source
        """, limit=1)
        stats['global'] = global_stats[0] if global_stats else {}
        
        # File type breakdown
        file_types = self.execute_query_safe("""
            MATCH (f:File)
            RETURN f.file_type as type, 
                   count(f) as count,
                   avg(f.eating_now_score) as avg_eating_now_score
            ORDER BY avg_eating_now_score DESC
        """)
        stats['file_types'] = file_types
        
        return stats
    
    def find_top_eating_now_files(self, repository: str = None, limit: int = 20) -> List[Dict]:
        """Find files with highest eating now scores - critical for plugin development"""
        if repository and repository in self.available_repos:
            query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            WHERE f.eating_now_score IS NOT NULL AND f.eating_now_score > 0
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            RETURN f.name as filename, 
                   f.package as package, 
                   f.repository as repository,
                   f.eating_now_score as eating_now_score,
                   f.importance_score as importance,
                   f.complexity_score as complexity,
                   f.function_count as functions,
                   f.lines_of_code as loc,
                   f.file_type as file_type,
                   f.is_eating_now_critical as is_critical,
                   f.has_source_code as has_source,
                   outgoing,
                   incoming,
                   (outgoing + incoming) as total_connections
            ORDER BY f.eating_now_score DESC
            LIMIT {limit}
            """
        else:
            query = f"""
            MATCH (f:File)
            WHERE f.eating_now_score IS NOT NULL AND f.eating_now_score > 0
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            RETURN f.name as filename, 
                   f.package as package, 
                   f.repository as repository,
                   f.eating_now_score as eating_now_score,
                   f.importance_score as importance,
                   f.complexity_score as complexity,
                   f.function_count as functions,
                   f.lines_of_code as loc,
                   f.file_type as file_type,
                   f.is_eating_now_critical as is_critical,
                   f.has_source_code as has_source,
                   outgoing,
                   incoming,
                   (outgoing + incoming) as total_connections
            ORDER BY f.eating_now_score DESC
            LIMIT {limit}
            """
        
        return self.execute_query_safe(query)
    
    def get_source_code_for_file(self, filename: str, repository: str = None) -> Dict[str, Any]:
        """Get complete source code for a specific file"""
        where_parts = ["f.name = $filename"]
        if repository and repository in self.available_repos:
            where_parts.append(f"f.repository = '{repository}'")
        
        where_clause = " AND ".join(where_parts)
        
        query = f"""
        MATCH (f:File)
        WHERE {where_clause}
        RETURN f.name as filename,
               f.path as path,
               f.repository as repository,
               f.package as package,
               f.eating_now_score as eating_now_score,
               f.importance_score as importance,
               f.source_code as source_code,
               f.key_snippets as key_snippets,
               f.functions as functions,
               f.classes as classes,
               f.imports as imports,
               f.has_source_code as has_source
        """
        
        result = self.execute_query_safe(query, {"filename": filename}, limit=1)
        return result[0] if result else {}
    
    def search_source_code(self, search_term: str, repository: str = None, limit: int = 15) -> List[Dict]:
        """Search within source code content using full-text search with fallback"""
        # First try full-text search
        try:
            if repository and repository in self.available_repos:
                query = f"""
                CALL db.index.fulltext.queryNodes('file_source_idx', $search_term) YIELD node, score
                WHERE node.repository = '{repository}'
                RETURN node.name as filename,
                       node.repository as repository,
                       node.package as package,
                       node.eating_now_score as eating_now_score,
                       node.importance_score as importance,
                       node.has_source_code as has_source,
                       score,
                       substring(node.source_code, 0, 500) as source_preview
                ORDER BY score DESC, node.eating_now_score DESC
                LIMIT {limit}
                """
            else:
                query = f"""
                CALL db.index.fulltext.queryNodes('file_source_idx', $search_term) YIELD node, score
                RETURN node.name as filename,
                       node.repository as repository,
                       node.package as package,
                       node.eating_now_score as eating_now_score,
                       node.importance_score as importance,
                       node.has_source_code as has_source,
                       score,
                       substring(node.source_code, 0, 500) as source_preview
                ORDER BY score DESC, node.eating_now_score DESC
                LIMIT {limit}
                """
            
            results = self.execute_query_safe(query, {"search_term": search_term})
            if results:
                return results
        except:
            pass  # Fall back to property search
        
        # Fallback to property-based search if full-text search fails
        logger.info("Full-text search not available, using property-based search fallback")
        
        if repository and repository in self.available_repos:
            fallback_query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            WHERE f.source_code IS NOT NULL 
            AND (toLower(f.source_code) CONTAINS toLower($search_term)
                 OR toLower(f.name) CONTAINS toLower($search_term)
                 OR toLower(f.package) CONTAINS toLower($search_term))
            RETURN f.name as filename,
                   f.repository as repository,
                   f.package as package,
                   f.eating_now_score as eating_now_score,
                   f.importance_score as importance,
                   f.has_source_code as has_source,
                   f.eating_now_score as score,
                   substring(f.source_code, 0, 500) as source_preview
            ORDER BY f.eating_now_score DESC, f.importance_score DESC
            LIMIT {limit}
            """
        else:
            fallback_query = f"""
            MATCH (f:File)
            WHERE f.source_code IS NOT NULL 
            AND (toLower(f.source_code) CONTAINS toLower($search_term)
                 OR toLower(f.name) CONTAINS toLower($search_term)
                 OR toLower(f.package) CONTAINS toLower($search_term))
            RETURN f.name as filename,
                   f.repository as repository,
                   f.package as package,
                   f.eating_now_score as eating_now_score,
                   f.importance_score as importance,
                   f.has_source_code as has_source,
                   f.eating_now_score as score,
                   substring(f.source_code, 0, 500) as source_preview
            ORDER BY f.eating_now_score DESC, f.importance_score DESC
            LIMIT {limit}
            """
        
        return self.execute_query_safe(fallback_query, {"search_term": search_term})
    
    def get_eating_now_architecture(self, repository: str = None) -> Dict[str, Any]:
        """Get eating now specific architecture insights"""
        insights = {}
        
        # Core eating now files
        if repository and repository in self.available_repos:
            core_files_query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            WHERE f.eating_now_score > 100
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            RETURN f.name as filename,
                   f.repository as repository,
                   f.eating_now_score as eating_now_score,
                   f.importance_score as importance,
                   f.package as package,
                   f.has_source_code as has_source,
                   outgoing,
                   incoming,
                   (outgoing + incoming) as total_connections
            ORDER BY f.eating_now_score DESC
            LIMIT 15
            """
        else:
            core_files_query = """
            MATCH (f:File)
            WHERE f.eating_now_score > 100
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            RETURN f.name as filename,
                   f.repository as repository,
                   f.eating_now_score as eating_now_score,
                   f.importance_score as importance,
                   f.package as package,
                   f.has_source_code as has_source,
                   outgoing,
                   incoming,
                   (outgoing + incoming) as total_connections
            ORDER BY f.eating_now_score DESC
            LIMIT 15
            """
        insights['core_eating_now_files'] = self.execute_query_safe(core_files_query)
        
        # Eating now packages
        if repository and repository in self.available_repos:
            packages_query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            WHERE f.eating_now_score > 50 AND f.package IS NOT NULL AND f.package <> 'unknown'
            WITH f.package as package,
                 count(f) as file_count,
                 avg(f.eating_now_score) as avg_eating_now,
                 max(f.eating_now_score) as max_eating_now,
                 sum(f.lines_of_code) as total_loc
            WHERE file_count > 0
            RETURN package,
                   file_count,
                   round(avg_eating_now, 2) as avg_eating_now,
                   round(max_eating_now, 2) as max_eating_now,
                   total_loc
            ORDER BY avg_eating_now DESC
            LIMIT 10
            """
        else:
            packages_query = """
            MATCH (f:File)
            WHERE f.eating_now_score > 50 AND f.package IS NOT NULL AND f.package <> 'unknown'
            WITH f.package as package,
                 f.repository as repository,
                 count(f) as file_count,
                 avg(f.eating_now_score) as avg_eating_now,
                 max(f.eating_now_score) as max_eating_now,
                 sum(f.lines_of_code) as total_loc
            WHERE file_count > 0
            RETURN package,
                   repository,
                   file_count,
                   round(avg_eating_now, 2) as avg_eating_now,
                   round(max_eating_now, 2) as max_eating_now,
                   total_loc
            ORDER BY avg_eating_now DESC
            LIMIT 15
            """
        insights['eating_now_packages'] = self.execute_query_safe(packages_query)
        
        # Files that depend on eating now functionality
        dependency_query = """
        MATCH (f1:File)-[:CALLS]->(f2:File)
        WHERE f2.eating_now_score > 100
        WITH f1, count(DISTINCT f2) as eating_now_dependencies, 
             collect(DISTINCT f2.name) as dependent_files
        RETURN f1.name as filename,
               f1.repository as repository,
               f1.eating_now_score as own_eating_now_score,
               eating_now_dependencies,
               dependent_files[0..3] as sample_dependencies
        ORDER BY eating_now_dependencies DESC, f1.eating_now_score DESC
        LIMIT 10
        """
        insights['eating_now_dependents'] = self.execute_query_safe(dependency_query)
        
        return insights
    
    def find_plugin_templates(self, functionality: str = "eating", limit: int = 10) -> List[Dict]:
        """Find files that can serve as templates for plugin development"""
        search_terms = {
            'eating': ['eating', 'bolus', 'carb', 'meal'],
            'pump': ['pump', 'insulin', 'dose'],
            'cgm': ['cgm', 'glucose', 'sensor'],
            'automation': ['automation', 'rule', 'trigger']
        }
        
        terms = search_terms.get(functionality.lower(), [functionality])
        
        query = """
        MATCH (f:File)
        WHERE f.eating_now_score > 50 
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
               f.function_count as functions,
               f.lines_of_code as loc,
               f.has_source_code as has_source,
               substring(f.source_code, 0, 1000) as source_preview,
               (outgoing + incoming) as connections
        ORDER BY f.eating_now_score DESC, connections DESC
        LIMIT $limit
        """
        
        return self.execute_query_safe(query, {"terms": terms, "limit": limit})
    
    def get_code_examples_for_concept(self, concept: str, repository: str = None, limit: int = 10) -> List[Dict]:
        """Get specific code examples for a concept with fallback search"""
        # First try full-text search
        try:
            if repository and repository in self.available_repos:
                query = f"""
                CALL db.index.fulltext.queryNodes('file_source_idx', $concept) YIELD node, score
                WHERE node.repository = '{repository}' AND node.eating_now_score > 0
                RETURN node.name as filename,
                       node.repository as repository,
                       node.package as package,
                       node.eating_now_score as eating_now_score,
                       node.has_source_code as has_source,
                       score,
                       node.source_code as full_source_code,
                       node.key_snippets as key_snippets
                ORDER BY score DESC, node.eating_now_score DESC
                LIMIT {limit}
                """
            else:
                query = f"""
                CALL db.index.fulltext.queryNodes('file_source_idx', $concept) YIELD node, score
                WHERE node.eating_now_score > 0
                RETURN node.name as filename,
                       node.repository as repository,
                       node.package as package,
                       node.eating_now_score as eating_now_score,
                       node.has_source_code as has_source,
                       score,
                       node.source_code as full_source_code,
                       node.key_snippets as key_snippets
                ORDER BY score DESC, node.eating_now_score DESC
                LIMIT {limit}
                """
            
            results = self.execute_query_safe(query, {"concept": concept})
            if results:
                return results
        except:
            pass  # Fall back to property search
        
        # Fallback to property-based search
        logger.info("Using property-based search for code examples")
        
        if repository and repository in self.available_repos:
            fallback_query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            WHERE f.eating_now_score > 0 
            AND f.source_code IS NOT NULL
            AND (toLower(f.source_code) CONTAINS toLower($concept)
                 OR toLower(f.name) CONTAINS toLower($concept)
                 OR toLower(f.package) CONTAINS toLower($concept))
            RETURN f.name as filename,
                   f.repository as repository,
                   f.package as package,
                   f.eating_now_score as eating_now_score,
                   f.has_source_code as has_source,
                   f.eating_now_score as score,
                   f.source_code as full_source_code,
                   f.key_snippets as key_snippets
            ORDER BY f.eating_now_score DESC
            LIMIT {limit}
            """
        else:
            fallback_query = f"""
            MATCH (f:File)
            WHERE f.eating_now_score > 0 
            AND f.source_code IS NOT NULL
            AND (toLower(f.source_code) CONTAINS toLower($concept)
                 OR toLower(f.name) CONTAINS toLower($concept)
                 OR toLower(f.package) CONTAINS toLower($concept))
            RETURN f.name as filename,
                   f.repository as repository,
                   f.package as package,
                   f.eating_now_score as eating_now_score,
                   f.has_source_code as has_source,
                   f.eating_now_score as score,
                   f.source_code as full_source_code,
                   f.key_snippets as key_snippets
            ORDER BY f.eating_now_score DESC
            LIMIT {limit}
            """
        
        return self.execute_query_safe(fallback_query, {"concept": concept})
    
    def compare_eating_now_implementations(self) -> List[Dict]:
        """Compare eating now implementations across repositories"""
        query = """
        MATCH (f:File)
        WHERE f.eating_now_score > 100 
        AND (toLower(f.name) CONTAINS 'eating' OR toLower(f.name) CONTAINS 'bolus')
        WITH f.repository as repository,
             collect({
                 name: f.name,
                 eating_now_score: f.eating_now_score,
                 importance: f.importance_score,
                 loc: f.lines_of_code,
                 functions: f.function_count,
                 package: f.package,
                 has_source: f.has_source_code
             }) as files,
             avg(f.eating_now_score) as avg_eating_now,
             max(f.eating_now_score) as max_eating_now,
             count(f) as file_count
        RETURN repository,
               file_count,
               round(avg_eating_now, 2) as avg_eating_now,
               round(max_eating_now, 2) as max_eating_now,
               files
        ORDER BY avg_eating_now DESC
        """
        
        return self.execute_query_safe(query)
    
    def generate_enhanced_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report with eating now focus"""
        logger.info("Generating enhanced multi-repository report with eating now focus...")
        
        report = {
            "database_overview": self.get_enhanced_database_overview(),
            "top_eating_now_files": {
                "global": self.find_top_eating_now_files()[:20],
                "by_repository": {}
            },
            "eating_now_architecture": {
                "global": self.get_eating_now_architecture(),
                "by_repository": {}
            },
            "plugin_templates": {
                "eating": self.find_plugin_templates("eating"),
                "bolus": self.find_plugin_templates("bolus"),
                "carb": self.find_plugin_templates("carb")
            },
            "repository_comparison": self.compare_eating_now_implementations(),
            "code_examples": {
                "bolus_calculation": self.get_code_examples_for_concept("bolus calculation"),
                "carb_counting": self.get_code_examples_for_concept("carb counting"),
                "meal_timing": self.get_code_examples_for_concept("meal timing")
            }
        }
        
        # Add per-repository analysis
        for repo_name in self.available_repos:
            report["top_eating_now_files"]["by_repository"][repo_name] = self.find_top_eating_now_files(repo_name)[:10]
            report["eating_now_architecture"]["by_repository"][repo_name] = self.get_eating_now_architecture(repo_name)
        
        return report


class EnhancedInteractiveExplorer:
    """Enhanced interactive database explorer with source code access"""
    
    def __init__(self, queries: EnhancedMultiRepoQueries):
        self.queries = queries
    
    def run(self):
        """Run enhanced interactive exploration session"""
        print(f"🔍 AAPS Enhanced Multi-Repository Database Explorer")
        print(f"🍽️ WITH EATING NOW FOCUS AND SOURCE CODE ACCESS")
        print(f"📚 Available repositories: {', '.join(self.queries.available_repos)}")
        print(f"Type 'help' for commands, 'quit' to exit\n")
        
        while True:
            try:
                command = input("🔍 Enhanced Explorer> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                elif command == 'help':
                    self.show_enhanced_help()
                elif command == 'overview':
                    self.show_enhanced_overview()
                elif command == 'eating':
                    self.show_eating_now_files()
                elif command.startswith('eating '):
                    parts = command.split()
                    repo = parts[1] if len(parts) > 1 and parts[1] in self.queries.available_repos else None
                    self.show_eating_now_files(repo)
                elif command.startswith('source '):
                    parts = command.split(maxsplit=1)
                    if len(parts) > 1:
                        self.show_source_code(parts[1])
                    else:
                        print("Usage: source <filename>")
                elif command.startswith('search '):
                    parts = command.split(maxsplit=1)
                    if len(parts) > 1:
                        self.search_source_code(parts[1])
                    else:
                        print("Usage: search <term>")
                elif command == 'templates':
                    self.show_plugin_templates()
                elif command.startswith('templates '):
                    parts = command.split()
                    concept = parts[1] if len(parts) > 1 else 'eating'
                    self.show_plugin_templates(concept)
                elif command == 'architecture':
                    self.show_eating_now_architecture()
                elif command == 'compare':
                    self.show_repository_comparison()
                elif command.startswith('examples '):
                    parts = command.split(maxsplit=1)
                    if len(parts) > 1:
                        self.show_code_examples(parts[1])
                    else:
                        print("Usage: examples <concept>")
                elif not command:
                    continue
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("👋 Goodbye!")
    
    def show_enhanced_help(self):
        """Show enhanced help information"""
        help_text = """
🔍 Enhanced Database Explorer Commands:

🍽️ Eating Now Focused:
   eating [repo]     - Show top eating now files (optionally for specific repo)
   templates [type]  - Show plugin templates (eating, bolus, carb, pump, cgm)
   architecture      - Show eating now architecture insights
   examples <concept> - Get code examples for concept (e.g., "bolus calculation")

💾 Source Code Access:
   source <filename> - Show complete source code for a file
   search <term>     - Search within source code using full-text search

📊 Analysis:
   overview         - Show enhanced database overview with eating now metrics
   compare          - Compare eating now implementations across repositories

💡 Examples:
   eating EN_new           - Top eating now files in EN_new
   source BolusPlugin.kt   - Show source code for BolusPlugin.kt
   search "bolus calculation" - Search for bolus calculation in source code
   templates bolus         - Show bolus-related plugin templates
   examples "carb counting" - Get carb counting code examples

📚 Repository-specific examples:
   eating EN_new     - Eating now files in EN_new
   eating AAPS_source - Eating now files in AAPS_source

💡 Other:
   help - Show this help
   quit - Exit explorer
"""
        print(help_text)
    
    def show_enhanced_overview(self):
        """Show enhanced database overview with eating now metrics"""
        overview = self.queries.get_enhanced_database_overview()
        
        print("\n📊 ENHANCED DATABASE OVERVIEW:")
        print("="*60)
        
        # Global stats with eating now
        if overview.get('global'):
            global_stats = overview['global']
            print(f"🌍 Global Statistics:")
            print(f"   Total Files: {global_stats.get('total_files', 0):,}")
            print(f"   Total LOC: {global_stats.get('total_loc', 0):,}")
            print(f"   Total Functions: {global_stats.get('total_functions', 0):,}")
            print(f"   Average Importance: {global_stats.get('avg_importance', 0):.1f}")
            print(f"   🍽️ Average Eating Now Score: {global_stats.get('avg_eating_now', 0):.1f}")
            print(f"   🍽️ Max Eating Now Score: {global_stats.get('max_eating_now', 0):.1f}")
            print(f"   🍽️ Critical Eating Now Files: {global_stats.get('critical_eating_now_files', 0):,}")
            print(f"   💾 Files with Source Code: {global_stats.get('files_with_source', 0):,}")
        
        # Repository breakdown with eating now focus
        if overview.get('repositories'):
            print(f"\n📚 Repository Breakdown (Eating Now Focused):")
            for repo in overview['repositories']:
                is_eating_repo = repo.get('is_eating_now_repo', False)
                repo_indicator = "🍽️" if is_eating_repo else "📦"
                print(f"   {repo_indicator} {repo['repository']}:")
                print(f"      Files: {repo['files']:,}")
                print(f"      LOC: {repo['loc']:,}")
                print(f"      Functions: {repo['functions']:,}")
                if repo.get('avg_eating_now') is not None:
                    print(f"      🍽️ Avg Eating Now Score: {repo['avg_eating_now']:.1f}")
                if repo.get('files_with_source') is not None:
                    print(f"      💾 Files with Source: {repo['files_with_source']:,}")
        
        print()
    
    def show_eating_now_files(self, repository: str = None):
        """Show top eating now files"""
        files = self.queries.find_top_eating_now_files(repository, 15)
        
        repo_text = f" in {repository}" if repository else ""
        print(f"\n🍽️ TOP EATING NOW FILES{repo_text} (CRITICAL FOR PLUGIN DEVELOPMENT):")
        print("="*80)
        
        if not files:
            print("No eating now files found with significant scores.")
            return
        
        for i, file_info in enumerate(files, 1):
            repo = file_info.get('repository', 'unknown')
            eating_score = file_info.get('eating_now_score', 0)
            is_critical = file_info.get('is_critical', False)
            has_source = file_info.get('has_source', False)
            critical_indicator = "🔥" if is_critical else "  "
            source_indicator = "💾" if has_source else "  "
            
            print(f"{i:2d}.{critical_indicator}{source_indicator} {file_info['filename']} ({repo})")
            print(f"       🍽️ Eating Now Score: {eating_score:.1f}")
            print(f"       ⭐ Importance: {file_info.get('importance', 0):.1f}")
            print(f"       📦 Package: {file_info.get('package', 'unknown')}")
            print(f"       📊 Functions: {file_info.get('functions', 0)}, LOC: {file_info.get('loc', 0)}")
            print(f"       🔗 Connections: {file_info.get('total_connections', 0)}")
            print(f"       💾 Has Source Code: {'Yes' if has_source else 'No'}")
            print()
    
    def show_source_code(self, filename: str):
        """Show complete source code for a file"""
        source_data = self.queries.get_source_code_for_file(filename)
        
        if not source_data:
            print(f"❌ File '{filename}' not found in database.")
            return
        
        print(f"\n💾 SOURCE CODE FOR: {source_data['filename']}")
        print("="*80)
        print(f"📦 Repository: {source_data.get('repository', 'unknown')}")
        print(f"📁 Package: {source_data.get('package', 'unknown')}")
        print(f"🍽️ Eating Now Score: {source_data.get('eating_now_score', 0):.1f}")
        print(f"⭐ Importance: {source_data.get('importance', 0):.1f}")
        print(f"💾 Has Source Code: {source_data.get('has_source', False)}")
        print("="*80)
        
        source_code = source_data.get('source_code', '')
        if source_code:
            # Show first 100 lines
            lines = source_code.split('\n')
            for i, line in enumerate(lines[:100], 1):
                print(f"{i:3d}: {line}")
            
            if len(lines) > 100:
                print(f"\n[... showing first 100 of {len(lines)} lines ...]")
                print("Full source code is available in the database.")
        else:
            print("❌ No source code available for this file.")
        
        # Show key snippets if available
        key_snippets = source_data.get('key_snippets')
        if key_snippets:
            try:
                snippets = json.loads(key_snippets) if isinstance(key_snippets, str) else key_snippets
                if snippets:
                    print(f"\n🔧 AVAILABLE KEY SNIPPETS:")
                    for snippet_name in snippets.keys():
                        print(f"   • {snippet_name}")
            except:
                pass
        
        print()
    
    def search_source_code(self, search_term: str):
        """Search within source code"""
        results = self.queries.search_source_code(search_term, None, 10)
        
        print(f"\n🔍 SOURCE CODE SEARCH RESULTS for '{search_term}':")
        print("="*80)
        
        if not results:
            print("No matches found in source code.")
            return
        
        for i, result in enumerate(results, 1):
            repo = result.get('repository', 'unknown')
            eating_score = result.get('eating_now_score', 0)
            score = result.get('score', 0)
            has_source = result.get('has_source', False)
            
            print(f"{i:2d}. {result['filename']} ({repo})")
            print(f"     🍽️ Eating Now Score: {eating_score:.1f}")
            print(f"     🔍 Search Score: {score:.2f}")
            print(f"     📦 Package: {result.get('package', 'unknown')}")
            print(f"     💾 Has Source: {'Yes' if has_source else 'No'}")
            
            preview = result.get('source_preview', '')
            if preview:
                # Clean up preview and highlight search term
                cleaned_preview = preview.replace('\n', ' ').replace('\t', ' ')
                # Simple highlighting (case insensitive)
                highlighted = re.sub(
                    f'({re.escape(search_term)})', 
                    r'>>>\1<<<', 
                    cleaned_preview, 
                    flags=re.IGNORECASE
                )
                print(f"     📝 Preview: {highlighted[:200]}...")
            print()
    
    def show_plugin_templates(self, template_type: str = 'eating'):
        """Show plugin templates for specific functionality"""
        templates = self.queries.find_plugin_templates(template_type, 8)
        
        print(f"\n🔧 PLUGIN TEMPLATES FOR '{template_type.upper()}':")
        print("="*60)
        
        if not templates:
            print(f"No templates found for '{template_type}'.")
            return
        
        for i, template in enumerate(templates, 1):
            repo = template.get('repository', 'unknown')
            eating_score = template.get('eating_now_score', 0)
            connections = template.get('connections', 0)
            has_source = template.get('has_source', False)
            
            print(f"{i:2d}. {template['filename']} ({repo})")
            print(f"     🍽️ Eating Now Score: {eating_score:.1f}")
            print(f"     ⭐ Importance: {template.get('importance', 0):.1f}")
            print(f"     📦 Package: {template.get('package', 'unknown')}")
            print(f"     📊 Functions: {template.get('functions', 0)}, LOC: {template.get('loc', 0)}")
            print(f"     🔗 Connections: {connections}")
            print(f"     💾 Has Source Code: {'Yes' if has_source else 'No'}")
            
            preview = template.get('source_preview', '')
            if preview:
                # Show class or function declaration
                lines = preview.split('\n')
                for line in lines[:3]:
                    if 'class ' in line or 'fun ' in line or 'public ' in line:
                        print(f"     💻 {line.strip()}")
                        break
            print()
    
    def show_eating_now_architecture(self):
        """Show eating now architecture insights"""
        insights = self.queries.get_eating_now_architecture()
        
        print("\n🏗️ EATING NOW ARCHITECTURE INSIGHTS:")
        print("="*60)
        
        # Core eating now files
        if insights.get('core_eating_now_files'):
            print("\n🔥 Core Eating Now Files (Score > 100):")
            for file_info in insights['core_eating_now_files'][:8]:
                repo = file_info.get('repository', 'unknown')
                eating_score = file_info.get('eating_now_score', 0)
                connections = file_info.get('total_connections', 0)
                has_source = file_info.get('has_source', False)
                source_indicator = "💾" if has_source else "  "
                print(f"   •{source_indicator} {file_info['filename']} ({repo})")
                print(f"     🍽️ Score: {eating_score:.1f}, 🔗 Connections: {connections}")
        
        # Eating now packages
        if insights.get('eating_now_packages'):
            print("\n📦 Eating Now Packages:")
            for pkg in insights['eating_now_packages'][:6]:
                repo_info = f" ({pkg['repository']})" if 'repository' in pkg else ""
                print(f"   • {pkg['package']}{repo_info}")
                print(f"     Files: {pkg['file_count']}, Avg Score: {pkg['avg_eating_now']:.1f}")
        
        # Dependencies
        if insights.get('eating_now_dependents'):
            print("\n🔗 Files Depending on Eating Now Functionality:")
            for dep in insights['eating_now_dependents'][:5]:
                repo = dep.get('repository', 'unknown')
                deps_count = dep.get('eating_now_dependencies', 0)
                print(f"   • {dep['filename']} ({repo})")
                print(f"     Depends on {deps_count} eating now files")
        
        print()
    
    def show_repository_comparison(self):
        """Show eating now implementation comparison across repositories"""
        comparison = self.queries.compare_eating_now_implementations()
        
        print("\n🔄 EATING NOW IMPLEMENTATION COMPARISON:")
        print("="*70)
        
        for repo_data in comparison:
            repo = repo_data['repository']
            file_count = repo_data['file_count']
            avg_score = repo_data['avg_eating_now']
            max_score = repo_data['max_eating_now']
            
            print(f"\n📦 {repo}:")
            print(f"   Eating Now Files: {file_count}")
            print(f"   Average Score: {avg_score:.1f}")
            print(f"   Max Score: {max_score:.1f}")
            
            # Show top files
            files = repo_data.get('files', [])
            if files:
                print(f"   Top Files:")
                for file_info in files[:3]:
                    has_source = file_info.get('has_source', False)
                    source_indicator = "💾" if has_source else "  "
                    print(f"     •{source_indicator} {file_info['name']}: {file_info['eating_now_score']:.1f}")
        
        print()
    
    def show_code_examples(self, concept: str):
        """Show code examples for a specific concept"""
        examples = self.queries.get_code_examples_for_concept(concept, None, 5)
        
        print(f"\n💻 CODE EXAMPLES FOR '{concept.upper()}':")
        print("="*60)
        
        if not examples:
            print(f"No code examples found for '{concept}'.")
            return
        
        for i, example in enumerate(examples, 1):
            repo = example.get('repository', 'unknown')
            eating_score = example.get('eating_now_score', 0)
            score = example.get('score', 0)
            has_source = example.get('has_source', False)
            
            print(f"{i}. {example['filename']} ({repo})")
            print(f"   🍽️ Eating Now Score: {eating_score:.1f}")
            print(f"   🔍 Relevance Score: {score:.2f}")
            print(f"   💾 Has Source Code: {'Yes' if has_source else 'No'}")
            
            # Show code snippets if available
            key_snippets = example.get('key_snippets')
            if key_snippets:
                try:
                    snippets = json.loads(key_snippets) if isinstance(key_snippets, str) else key_snippets
                    if snippets:
                        # Show first relevant snippet
                        for snippet_name, snippet_code in snippets.items():
                            if any(term in snippet_name.lower() for term in concept.lower().split()):
                                print(f"   💻 Relevant code snippet ({snippet_name}):")
                                lines = snippet_code.split('\n')
                                for line in lines[:5]:  # Show first 5 lines
                                    if line.strip():
                                        print(f"      {line}")
                                if len(lines) > 5:
                                    print(f"      ... (truncated)")
                                break
                except:
                    pass
            print()


def main():
    """Main execution with enhanced analysis"""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change to your Neo4j password
    
    print("🔍 AAPS Enhanced Multi-Repository Neo4j Utilities")
    print("🍽️ WITH EATING NOW FOCUS AND SOURCE CODE ACCESS")
    print("🗄️ Enhanced Analyzer Database Explorer")
    print("="*70)
    
    try:
        # Initialize enhanced queries
        queries = EnhancedMultiRepoQueries(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Check database connectivity and content
        overview = queries.get_enhanced_database_overview()
        if not overview.get('global', {}).get('total_files'):
            print("❌ Database appears to be empty or not populated.")
            print("Run the Enhanced Analyzer first: python aaps_analyzer.py")
            return
        
        print(f"✅ Connected to enhanced database with {len(queries.available_repos)} repositories")
        
        # Generate enhanced report
        print("📊 Generating enhanced analysis report with eating now focus...")
        report = queries.generate_enhanced_report()
        
        # Save report
        with open('aaps_enhanced_database_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display key findings
        print("\n" + "="*80)
        print("📊 AAPS ENHANCED DATABASE ANALYSIS REPORT")
        print("🍽️ WITH EATING NOW PRIORITIZATION")
        print("="*80)
        
        # Database overview
        if report['database_overview'].get('global'):
            global_stats = report['database_overview']['global']
            print(f"\n🌍 GLOBAL OVERVIEW:")
            print(f"   Total Files: {global_stats.get('total_files', 0):,}")
            print(f"   🍽️ Critical Eating Now Files: {global_stats.get('critical_eating_now_files', 0):,}")
            print(f"   🍽️ Average Eating Now Score: {global_stats.get('avg_eating_now', 0):.1f}")
            print(f"   🍽️ Max Eating Now Score: {global_stats.get('max_eating_now', 0):.1f}")
            print(f"   💾 Files with Source Code: {global_stats.get('files_with_source', 0):,}")
        
        # Top eating now files
        print(f"\n🍽️ TOP EATING NOW FILES (CRITICAL FOR PLUGIN DEVELOPMENT):")
        for i, file_info in enumerate(report['top_eating_now_files']['global'][:8], 1):
            repo = file_info.get('repository', 'unknown')
            eating_score = file_info.get('eating_now_score', 0)
            has_source = file_info.get('has_source', False)
            source_indicator = "💾" if has_source else "  "
            print(f"   {i:2d}.{source_indicator} {file_info['filename']} ({repo}): {eating_score:.1f}")
        
        # Repository comparison
        print(f"\n📚 EATING NOW REPOSITORY COMPARISON:")
        for repo_data in report['repository_comparison']:
            repo = repo_data['repository']
            files_with_source = sum(1 for f in repo_data['files'] if f.get('has_source', False))
            print(f"   📦 {repo}: {repo_data['file_count']} files, avg score: {repo_data['avg_eating_now']:.1f}, source: {files_with_source}")
        
        # Plugin templates
        print(f"\n🔧 AVAILABLE PLUGIN TEMPLATES:")
        for template_type, templates in report['plugin_templates'].items():
            if templates:
                top_template = templates[0]
                repo = top_template.get('repository', 'unknown')
                score = top_template.get('eating_now_score', 0)
                has_source = top_template.get('has_source', False)
                source_indicator = "💾" if has_source else "  "
                print(f"   🍽️ {template_type.title()}:{source_indicator} {top_template['filename']} ({repo}) - Score: {score:.1f}")
        
        print(f"\n📁 GENERATED FILES:")
        print("   📊 aaps_enhanced_database_report.json - Complete enhanced report with source code")
        
        print(f"\n💡 ENHANCED USAGE EXAMPLES:")
        print("   🔍 Interactive explorer: python neo4j_utilities.py")
        print("   🤖 Enhanced RAG: python ollama_neo4j_rag.py")
        print("   💾 Get source code: MATCH (f:File {name: 'BolusPlugin.kt'}) RETURN f.source_code")
        
        print(f"\n🔍 EXAMPLE ENHANCED QUERIES:")
        print("   // Top eating now files with source code")
        print("   MATCH (f:File) WHERE f.eating_now_score > 100 AND f.has_source_code = true")
        print("   RETURN f.name, f.eating_now_score, f.repository ORDER BY f.eating_now_score DESC")
        print("   ")
        print("   // Search source code")
        print("   CALL db.index.fulltext.queryNodes('file_source_idx', 'bolus calculation')")
        print("   YIELD node RETURN node.name, node.source_code, node.eating_now_score")
        print("   ")
        print("   // Files with full source code")
        print("   MATCH (f:File) WHERE f.has_source_code = true")
        print("   RETURN f.name, f.repository, f.eating_now_score ORDER BY f.eating_now_score DESC")
        
        print("\n" + "="*80)
        print("📊 Enhanced analysis complete!")
        
        # Start enhanced interactive explorer
        response = input("\nStart enhanced interactive database explorer? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            explorer = EnhancedInteractiveExplorer(queries)
            explorer.run()
        
        queries.close()
        
    except Exception as e:
        logger.error(f"Error running enhanced analysis: {e}")
        print(f"❌ Enhanced analysis failed: {e}")
        print("Make sure:")
        print("1. Neo4j is running and accessible")
        print("2. Database credentials are correct")
        print("3. Enhanced Analyzer has been run to populate the database")

if __name__ == "__main__":
    main()
