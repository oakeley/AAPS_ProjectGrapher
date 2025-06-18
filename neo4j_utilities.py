#!/usr/bin/env python3
"""
Multi-Repository Neo4j Query Utilities for AAPS Projects
Works with the Ultimate Analyzer database structure
Run standalone for database exploration and debugging
"""

import json
from typing import Dict, List, Any, Optional
import logging
import sys

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("❌ Neo4j driver not available. Install with: pip install neo4j")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateMultiRepoQueries:
    """Ultimate multi-repository Neo4j queries optimized for the new database structure"""
    
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
    
    def get_database_overview(self) -> Dict[str, Any]:
        """Get comprehensive database overview"""
        stats = {}
        
        # Repository counts and details
        repo_stats = self.execute_query_safe("""
            MATCH (r:Repository)
            RETURN r.name as repository, 
                   r.file_count as files,
                   r.total_loc as loc,
                   r.total_functions as functions
            ORDER BY r.file_count DESC
        """)
        stats['repositories'] = repo_stats
        
        # Global counts
        global_stats = self.execute_query_safe("""
            MATCH (f:File)
            RETURN count(f) as total_files,
                   sum(f.lines_of_code) as total_loc,
                   sum(f.function_count) as total_functions,
                   avg(f.importance_score) as avg_importance,
                   max(f.importance_score) as max_importance
        """, limit=1)
        stats['global'] = global_stats[0] if global_stats else {}
        
        # File type breakdown
        file_types = self.execute_query_safe("""
            MATCH (f:File)
            RETURN f.file_type as type, count(f) as count
            ORDER BY count DESC
        """)
        stats['file_types'] = file_types
        
        # Call relationships
        calls_stats = self.execute_query_safe("""
            MATCH ()-[c:CALLS]->()
            RETURN count(c) as total_calls,
                   count(DISTINCT c.repository) as repositories_with_calls
        """, limit=1)
        stats['calls'] = calls_stats[0] if calls_stats else {}
        
        return stats
    
    def find_most_important_files(self, repository: str = None, limit: int = 20) -> List[Dict]:
        """Find files with highest importance scores"""
        if repository and repository in self.available_repos:
            query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            WHERE f.importance_score IS NOT NULL
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            RETURN f.name as filename, 
                   f.package as package, 
                   f.repository as repository,
                   f.importance_score as importance,
                   f.complexity_score as complexity,
                   f.function_count as functions,
                   f.lines_of_code as loc,
                   f.file_type as file_type,
                   outgoing,
                   incoming,
                   (outgoing + incoming) as total_connections
            ORDER BY f.importance_score DESC
            LIMIT {limit}
            """
        else:
            query = f"""
            MATCH (f:File)
            WHERE f.importance_score IS NOT NULL
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            RETURN f.name as filename, 
                   f.package as package, 
                   f.repository as repository,
                   f.importance_score as importance,
                   f.complexity_score as complexity,
                   f.function_count as functions,
                   f.lines_of_code as loc,
                   f.file_type as file_type,
                   outgoing,
                   incoming,
                   (outgoing + incoming) as total_connections
            ORDER BY f.importance_score DESC
            LIMIT {limit}
            """
        
        return self.execute_query_safe(query)
    
    def find_most_connected_files(self, repository: str = None, limit: int = 20) -> List[Dict]:
        """Find files with most connections within their repository"""
        if repository and repository in self.available_repos:
            query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            WHERE outgoing > 0 OR incoming > 0
            RETURN f.name as filename, 
                   f.package as package, 
                   f.repository as repository,
                   f.importance_score as importance,
                   outgoing, 
                   incoming,
                   (outgoing + incoming) as total_connections
            ORDER BY total_connections DESC, f.importance_score DESC
            LIMIT {limit}
            """
        else:
            query = f"""
            MATCH (f:File)
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            WHERE outgoing > 0 OR incoming > 0
            RETURN f.name as filename, 
                   f.package as package, 
                   f.repository as repository,
                   f.importance_score as importance,
                   outgoing, 
                   incoming,
                   (outgoing + incoming) as total_connections
            ORDER BY total_connections DESC, f.importance_score DESC
            LIMIT {limit}
            """
        
        return self.execute_query_safe(query)
    
    def find_package_overview(self, repository: str = None, limit: int = 25) -> List[Dict]:
        """Get overview by package"""
        if repository and repository in self.available_repos:
            query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            WHERE f.package IS NOT NULL AND f.package <> 'unknown'
            WITH f.package as package, 
                 f.repository as repository,
                 count(f) as file_count,
                 sum(f.function_count) as total_functions,
                 sum(f.lines_of_code) as total_loc,
                 avg(f.importance_score) as avg_importance,
                 max(f.importance_score) as max_importance
            WHERE file_count > 1
            RETURN package, 
                   repository, 
                   file_count, 
                   total_functions, 
                   total_loc, 
                   round(avg_importance, 2) as avg_importance,
                   round(max_importance, 2) as max_importance
            ORDER BY avg_importance DESC
            LIMIT {limit}
            """
        else:
            query = f"""
            MATCH (f:File)
            WHERE f.package IS NOT NULL AND f.package <> 'unknown'
            WITH f.package as package, 
                 f.repository as repository,
                 count(f) as file_count,
                 sum(f.function_count) as total_functions,
                 sum(f.lines_of_code) as total_loc,
                 avg(f.importance_score) as avg_importance,
                 max(f.importance_score) as max_importance
            WHERE file_count > 1
            RETURN package, 
                   repository, 
                   file_count, 
                   total_functions, 
                   total_loc, 
                   round(avg_importance, 2) as avg_importance,
                   round(max_importance, 2) as max_importance
            ORDER BY avg_importance DESC
            LIMIT {limit}
            """
        
        return self.execute_query_safe(query)
    
    def search_files_by_name(self, search_term: str, repository: str = None, limit: int = 20) -> List[Dict]:
        """Search files by name pattern"""
        if repository and repository in self.available_repos:
            query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            WHERE toLower(f.name) CONTAINS toLower($search_term)
               OR toLower(f.package) CONTAINS toLower($search_term)
               OR toLower(f.path) CONTAINS toLower($search_term)
            RETURN f.name as filename, 
                   f.package as package, 
                   f.repository as repository,
                   f.function_count as functions, 
                   f.lines_of_code as loc,
                   f.importance_score as importance,
                   f.complexity_score as complexity,
                   f.file_type as file_type
            ORDER BY f.importance_score DESC
            LIMIT {limit}
            """
        else:
            query = f"""
            MATCH (f:File)
            WHERE toLower(f.name) CONTAINS toLower($search_term)
               OR toLower(f.package) CONTAINS toLower($search_term)
               OR toLower(f.path) CONTAINS toLower($search_term)
            RETURN f.name as filename, 
                   f.package as package, 
                   f.repository as repository,
                   f.function_count as functions, 
                   f.lines_of_code as loc,
                   f.importance_score as importance,
                   f.complexity_score as complexity,
                   f.file_type as file_type
            ORDER BY f.importance_score DESC
            LIMIT {limit}
            """
        
        return self.execute_query_safe(query, {'search_term': search_term})
    
    def get_repository_comparison(self) -> List[Dict]:
        """Compare all repositories"""
        query = """
        MATCH (r:Repository)
        OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
        OPTIONAL MATCH (f1:File {repository: r.name})-[c:CALLS {repository: r.name}]->(f2:File {repository: r.name})
        WITH r, 
             count(DISTINCT f) as file_count,
             sum(f.lines_of_code) as total_loc,
             sum(f.function_count) as total_functions,
             avg(f.importance_score) as avg_importance,
             max(f.importance_score) as max_importance,
             count(DISTINCT c) as internal_calls
        RETURN r.name as repository,
               file_count,
               total_loc,
               total_functions,
               round(avg_importance, 2) as avg_importance,
               round(max_importance, 2) as max_importance,
               internal_calls
        ORDER BY avg_importance DESC
        """
        
        return self.execute_query_safe(query)
    
    def find_cross_repository_similarities(self) -> List[Dict]:
        """Find files with similar names across repositories"""
        query = """
        MATCH (f1:File), (f2:File)
        WHERE f1.repository <> f2.repository 
          AND f1.name = f2.name
          AND f1.repository < f2.repository
        RETURN f1.name as filename, 
               f1.repository as repo1, 
               f1.importance_score as importance1,
               f1.lines_of_code as loc1,
               f2.repository as repo2, 
               f2.importance_score as importance2,
               f2.lines_of_code as loc2,
               abs(f1.importance_score - f2.importance_score) as importance_diff
        ORDER BY f1.name, importance_diff DESC
        """
        
        return self.execute_query_safe(query, limit=30)
    
    def get_architecture_insights(self, repository: str = None) -> Dict[str, Any]:
        """Get architectural insights"""
        insights = {}
        
        # Entry points (files with few incoming calls but many outgoing)
        if repository and repository in self.available_repos:
            entry_points_query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            WHERE incoming <= 1 AND outgoing >= 3
            RETURN f.name as filename,
                   f.repository as repository,
                   f.importance_score as importance,
                   outgoing,
                   incoming
            ORDER BY outgoing DESC, f.importance_score DESC
            LIMIT 10
            """
        else:
            entry_points_query = """
            MATCH (f:File)
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            WHERE incoming <= 1 AND outgoing >= 3
            RETURN f.name as filename,
                   f.repository as repository,
                   f.importance_score as importance,
                   outgoing,
                   incoming
            ORDER BY outgoing DESC, f.importance_score DESC
            LIMIT 10
            """
        insights['entry_points'] = self.execute_query_safe(entry_points_query)
        
        # Core files (highly connected and important)
        if repository and repository in self.available_repos:
            core_files_query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            WHERE f.importance_score > 30 AND (outgoing + incoming) > 5
            RETURN f.name as filename,
                   f.repository as repository,
                   f.importance_score as importance,
                   (outgoing + incoming) as total_connections
            ORDER BY f.importance_score DESC, total_connections DESC
            LIMIT 10
            """
        else:
            core_files_query = """
            MATCH (f:File)
            OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
            WHERE out.repository = f.repository
            OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
            WHERE in.repository = f.repository
            WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
            WHERE f.importance_score > 30 AND (outgoing + incoming) > 5
            RETURN f.name as filename,
                   f.repository as repository,
                   f.importance_score as importance,
                   (outgoing + incoming) as total_connections
            ORDER BY f.importance_score DESC, total_connections DESC
            LIMIT 10
            """
        insights['core_files'] = self.execute_query_safe(core_files_query)
        
        # Isolated files (no connections)
        if repository and repository in self.available_repos:
            isolated_query = f"""
            MATCH (f:File {{repository: '{repository}'}})
            WHERE NOT (f)-[:CALLS]->() AND NOT ()-[:CALLS]->(f)
            RETURN f.name as filename,
                   f.repository as repository,
                   f.importance_score as importance,
                   f.lines_of_code as loc
            ORDER BY f.importance_score DESC
            LIMIT 10
            """
        else:
            isolated_query = """
            MATCH (f:File)
            WHERE NOT (f)-[:CALLS]->() AND NOT ()-[:CALLS]->(f)
            RETURN f.name as filename,
                   f.repository as repository,
                   f.importance_score as importance,
                   f.lines_of_code as loc
            ORDER BY f.importance_score DESC
            LIMIT 10
            """
        insights['isolated_files'] = self.execute_query_safe(isolated_query)
        
        return insights
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("Generating comprehensive multi-repository report...")
        
        report = {
            "database_overview": self.get_database_overview(),
            "repository_comparison": self.get_repository_comparison(),
            "most_important_files": {
                "global": self.find_most_important_files()[:15],
                "by_repository": {}
            },
            "most_connected_files": {
                "global": self.find_most_connected_files()[:15],
                "by_repository": {}
            },
            "package_analysis": {
                "global": self.find_package_overview()[:20],
                "by_repository": {}
            },
            "architecture_insights": {
                "global": self.get_architecture_insights(),
                "by_repository": {}
            },
            "cross_repository_analysis": {
                "similarities": self.find_cross_repository_similarities()
            }
        }
        
        # Add per-repository analysis
        for repo_name in self.available_repos:
            report["most_important_files"]["by_repository"][repo_name] = self.find_most_important_files(repo_name)[:10]
            report["most_connected_files"]["by_repository"][repo_name] = self.find_most_connected_files(repo_name)[:10]
            report["package_analysis"]["by_repository"][repo_name] = self.find_package_overview(repo_name)[:15]
            report["architecture_insights"]["by_repository"][repo_name] = self.get_architecture_insights(repo_name)
        
        return report


class InteractiveExplorer:
    """Interactive database explorer"""
    
    def __init__(self, queries: UltimateMultiRepoQueries):
        self.queries = queries
    
    def run(self):
        """Run interactive exploration session"""
        print(f"🔍 AAPS Multi-Repository Database Explorer")
        print(f"📚 Available repositories: {', '.join(self.queries.available_repos)}")
        print(f"Type 'help' for commands, 'quit' to exit\n")
        
        while True:
            try:
                command = input("🔍 Explorer> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                elif command == 'help':
                    self.show_help()
                elif command == 'overview':
                    self.show_overview()
                elif command == 'repos':
                    self.show_repositories()
                elif command.startswith('important'):
                    parts = command.split()
                    repo = parts[1] if len(parts) > 1 and parts[1] in self.queries.available_repos else None
                    self.show_important_files(repo)
                elif command.startswith('connected'):
                    parts = command.split()
                    repo = parts[1] if len(parts) > 1 and parts[1] in self.queries.available_repos else None
                    self.show_connected_files(repo)
                elif command.startswith('search'):
                    parts = command.split(maxsplit=1)
                    if len(parts) > 1:
                        self.search_files(parts[1])
                    else:
                        print("Usage: search <term>")
                elif command == 'packages':
                    self.show_packages()
                elif command == 'compare':
                    self.show_comparison()
                elif command == 'similarities':
                    self.show_similarities()
                elif command == 'insights':
                    self.show_insights()
                elif not command:
                    continue
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("👋 Goodbye!")
    
    def show_help(self):
        """Show help information"""
        help_text = """
🔍 Database Explorer Commands:

📊 General:
   overview     - Show database overview
   repos        - Show repository details
   compare      - Compare repositories
   similarities - Show cross-repository similarities
   insights     - Show architectural insights

🔍 Search & Analysis:
   important [repo]  - Show most important files (optionally for specific repo)
   connected [repo]  - Show most connected files (optionally for specific repo)
   search <term>     - Search files by name/package
   packages          - Show package analysis

📚 Repository-specific examples:
   important EN_new     - Important files in EN_new
   connected AAPS_source - Connected files in AAPS_source

💡 Other:
   help - Show this help
   quit - Exit explorer
"""
        print(help_text)
    
    def show_overview(self):
        """Show database overview"""
        overview = self.queries.get_database_overview()
        
        print("\n📊 DATABASE OVERVIEW:")
        print("="*50)
        
        # Global stats
        if overview.get('global'):
            global_stats = overview['global']
            print(f"🌍 Global Statistics:")
            print(f"   Total Files: {global_stats.get('total_files', 0):,}")
            print(f"   Total LOC: {global_stats.get('total_loc', 0):,}")
            print(f"   Total Functions: {global_stats.get('total_functions', 0):,}")
            print(f"   Average Importance: {global_stats.get('avg_importance', 0):.1f}")
            print(f"   Max Importance: {global_stats.get('max_importance', 0):.1f}")
        
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
        
        # Calls
        if overview.get('calls'):
            calls = overview['calls']
            print(f"\n🔗 Call Relationships:")
            print(f"   Total Calls: {calls.get('total_calls', 0):,}")
            print(f"   Repositories with Calls: {calls.get('repositories_with_calls', 0)}")
        
        print()
    
    def show_repositories(self):
        """Show detailed repository information"""
        comparison = self.queries.get_repository_comparison()
        
        print("\n📚 REPOSITORY DETAILS:")
        print("="*60)
        
        for repo in comparison:
            print(f"\n📦 {repo['repository']}:")
            print(f"   Files: {repo['file_count']:,}")
            print(f"   Lines of Code: {repo['total_loc']:,}")
            print(f"   Functions: {repo['total_functions']:,}")
            print(f"   Average Importance: {repo.get('avg_importance', 0):.1f}")
            print(f"   Max Importance: {repo.get('max_importance', 0):.1f}")
            print(f"   Internal Calls: {repo.get('internal_calls', 0):,}")
        
        print()
    
    def show_important_files(self, repository: str = None):
        """Show most important files"""
        files = self.queries.find_most_important_files(repository, 15)
        
        repo_text = f" in {repository}" if repository else ""
        print(f"\n⭐ MOST IMPORTANT FILES{repo_text}:")
        print("="*60)
        
        for i, file_info in enumerate(files, 1):
            repo = file_info.get('repository', 'unknown')
            print(f"{i:2d}. {file_info['filename']} ({repo})")
            print(f"     Importance: {file_info.get('importance', 0):.1f}")
            print(f"     Package: {file_info.get('package', 'unknown')}")
            print(f"     Functions: {file_info.get('functions', 0)}, LOC: {file_info.get('loc', 0)}")
            print(f"     Connections: {file_info.get('total_connections', 0)} (in: {file_info.get('incoming', 0)}, out: {file_info.get('outgoing', 0)})")
            print()
    
    def show_connected_files(self, repository: str = None):
        """Show most connected files"""
        files = self.queries.find_most_connected_files(repository, 15)
        
        repo_text = f" in {repository}" if repository else ""
        print(f"\n🔗 MOST CONNECTED FILES{repo_text}:")
        print("="*60)
        
        for i, file_info in enumerate(files, 1):
            repo = file_info.get('repository', 'unknown')
            total = file_info.get('total_connections', 0)
            incoming = file_info.get('incoming', 0)
            outgoing = file_info.get('outgoing', 0)
            
            print(f"{i:2d}. {file_info['filename']} ({repo})")
            print(f"     Connections: {total} (incoming: {incoming}, outgoing: {outgoing})")
            print(f"     Importance: {file_info.get('importance', 0):.1f}")
            print(f"     Package: {file_info.get('package', 'unknown')}")
            print()
    
    def search_files(self, search_term: str):
        """Search files by term"""
        files = self.queries.search_files_by_name(search_term, None, 20)
        
        print(f"\n🔍 SEARCH RESULTS for '{search_term}':")
        print("="*60)
        
        if not files:
            print("No files found matching the search term.")
            return
        
        for i, file_info in enumerate(files, 1):
            repo = file_info.get('repository', 'unknown')
            print(f"{i:2d}. {file_info['filename']} ({repo})")
            print(f"     Package: {file_info.get('package', 'unknown')}")
            print(f"     Importance: {file_info.get('importance', 0):.1f}")
            print(f"     Functions: {file_info.get('functions', 0)}, LOC: {file_info.get('loc', 0)}")
            print()
    
    def show_packages(self):
        """Show package analysis"""
        packages = self.queries.find_package_overview(None, 20)
        
        print("\n📦 PACKAGE ANALYSIS:")
        print("="*60)
        
        for i, pkg in enumerate(packages, 1):
            print(f"{i:2d}. {pkg['package']} ({pkg.get('repository', 'unknown')})")
            print(f"     Files: {pkg['file_count']}, Functions: {pkg.get('total_functions', 0)}")
            print(f"     LOC: {pkg.get('total_loc', 0):,}")
            print(f"     Avg Importance: {pkg.get('avg_importance', 0):.1f}")
            print(f"     Max Importance: {pkg.get('max_importance', 0):.1f}")
            print()
    
    def show_comparison(self):
        """Show repository comparison"""
        comparison = self.queries.get_repository_comparison()
        
        print("\n🔄 REPOSITORY COMPARISON:")
        print("="*80)
        print(f"{'Repository':<15} {'Files':<8} {'LOC':<10} {'Functions':<10} {'Avg Imp':<8} {'Max Imp':<8} {'Calls':<8}")
        print("-" * 80)
        
        for repo in comparison:
            print(f"{repo['repository']:<15} "
                  f"{repo['file_count']:<8} "
                  f"{repo['total_loc']:<10,} "
                  f"{repo['total_functions']:<10,} "
                  f"{repo.get('avg_importance', 0):<8.1f} "
                  f"{repo.get('max_importance', 0):<8.1f} "
                  f"{repo.get('internal_calls', 0):<8,}")
        
        print()
    
    def show_similarities(self):
        """Show cross-repository similarities"""
        similarities = self.queries.find_cross_repository_similarities()
        
        print("\n🌐 CROSS-REPOSITORY SIMILARITIES:")
        print("="*60)
        
        if not similarities:
            print("No files with identical names found across repositories.")
            return
        
        for i, sim in enumerate(similarities, 1):
            print(f"{i:2d}. {sim['filename']}")
            print(f"     {sim['repo1']}: importance {sim.get('importance1', 0):.1f}, LOC {sim.get('loc1', 0)}")
            print(f"     {sim['repo2']}: importance {sim.get('importance2', 0):.1f}, LOC {sim.get('loc2', 0)}")
            print(f"     Importance difference: {sim.get('importance_diff', 0):.1f}")
            print()
    
    def show_insights(self):
        """Show architectural insights"""
        insights = self.queries.get_architecture_insights()
        
        print("\n🏗️ ARCHITECTURAL INSIGHTS:")
        print("="*60)
        
        # Entry points
        if insights.get('entry_points'):
            print("\n🚪 Entry Points (few incoming, many outgoing calls):")
            for entry in insights['entry_points'][:5]:
                repo = entry.get('repository', 'unknown')
                print(f"   • {entry['filename']} ({repo})")
                print(f"     Importance: {entry.get('importance', 0):.1f}, Out: {entry.get('outgoing', 0)}")
        
        # Core files
        if insights.get('core_files'):
            print("\n🏛️ Core Files (highly important and connected):")
            for core in insights['core_files'][:5]:
                repo = core.get('repository', 'unknown')
                print(f"   • {core['filename']} ({repo})")
                print(f"     Importance: {core.get('importance', 0):.1f}, Connections: {core.get('total_connections', 0)}")
        
        # Isolated files
        if insights.get('isolated_files'):
            print("\n🏝️ Isolated Files (no call relationships):")
            for isolated in insights['isolated_files'][:5]:
                repo = isolated.get('repository', 'unknown')
                print(f"   • {isolated['filename']} ({repo})")
                print(f"     Importance: {isolated.get('importance', 0):.1f}, LOC: {isolated.get('loc', 0)}")
        
        print()


def main():
    """Main execution with comprehensive analysis"""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change to your Neo4j password
    
    print("🔍 AAPS Multi-Repository Neo4j Utilities")
    print("🗄️ Ultimate Analyzer Database Explorer")
    print("="*60)
    
    try:
        # Initialize queries
        queries = UltimateMultiRepoQueries(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Check database connectivity and content
        overview = queries.get_database_overview()
        if not overview.get('global', {}).get('total_files'):
            print("❌ Database appears to be empty or not populated.")
            print("Run the Ultimate Analyzer first: python aaps_analyzer.py")
            return
        
        print(f"✅ Connected to database with {len(queries.available_repos)} repositories")
        
        # Generate comprehensive report
        print("📊 Generating comprehensive analysis report...")
        report = queries.generate_comprehensive_report()
        
        # Save report
        with open('aaps_ultimate_database_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display key findings
        print("\n" + "="*80)
        print("📊 AAPS ULTIMATE DATABASE ANALYSIS REPORT")
        print("="*80)
        
        # Database overview
        if report['database_overview'].get('global'):
            global_stats = report['database_overview']['global']
            print(f"\n🌍 GLOBAL OVERVIEW:")
            print(f"   Total Files: {global_stats.get('total_files', 0):,}")
            print(f"   Total LOC: {global_stats.get('total_loc', 0):,}")
            print(f"   Total Functions: {global_stats.get('total_functions', 0):,}")
            print(f"   Average Importance: {global_stats.get('avg_importance', 0):.1f}")
        
        # Repository comparison
        print(f"\n📚 REPOSITORY COMPARISON:")
        for repo in report['repository_comparison']:
            print(f"   📦 {repo['repository']}:")
            print(f"      Files: {repo['file_count']:,}, LOC: {repo['total_loc']:,}")
            print(f"      Functions: {repo['total_functions']:,}")
            print(f"      Avg Importance: {repo.get('avg_importance', 0):.1f}")
            print(f"      Internal Calls: {repo.get('internal_calls', 0):,}")
        
        # Top global files
        print(f"\n⭐ TOP GLOBAL FILES (All Repositories):")
        for i, file_info in enumerate(report['most_important_files']['global'][:10], 1):
            repo = file_info.get('repository', 'unknown')
            print(f"   {i:2d}. {file_info['filename']} ({repo})")
            print(f"       Importance: {file_info.get('importance', 0):.1f}, Connections: {file_info.get('total_connections', 0)}")
        
        # Cross-repository similarities
        similarities = report['cross_repository_analysis']['similarities']
        if similarities:
            print(f"\n🔄 CROSS-REPOSITORY SIMILARITIES:")
            for sim in similarities[:5]:
                print(f"   🔗 {sim['filename']}:")
                print(f"      {sim['repo1']}: importance {sim.get('importance1', 0):.1f}")
                print(f"      {sim['repo2']}: importance {sim.get('importance2', 0):.1f}")
        
        # Architectural insights
        global_insights = report['architecture_insights']['global']
        if global_insights.get('entry_points'):
            print(f"\n🚪 KEY ENTRY POINTS:")
            for entry in global_insights['entry_points'][:5]:
                repo = entry.get('repository', 'unknown')
                print(f"   • {entry['filename']} ({repo}): {entry.get('outgoing', 0)} outgoing calls")
        
        print(f"\n📁 GENERATED FILES:")
        print("   📊 aaps_ultimate_database_report.json - Complete analysis report")
        
        print(f"\n💡 USAGE EXAMPLES:")
        print("   🔍 Interactive explorer: python neo4j_utilities.py")
        print("   🤖 RAG system: python ollama_neo4j_rag.py")
        
        print(f"\n🔍 EXAMPLE QUERIES:")
        print("   MATCH (r:Repository) RETURN r.name, r.file_count ORDER BY r.file_count DESC")
        print("   MATCH (f:File) WHERE f.importance_score > 50 RETURN f.name, f.repository, f.importance_score")
        print("   MATCH (f:File {repository: 'EN_new'}) RETURN f.name ORDER BY f.importance_score DESC LIMIT 10")
        
        print("\n" + "="*80)
        print("📊 Analysis complete!")
        
        # Start interactive explorer
        response = input("\nStart interactive database explorer? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            explorer = InteractiveExplorer(queries)
            explorer.run()
        
        queries.close()
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        print(f"❌ Analysis failed: {e}")
        print("Make sure:")
        print("1. Neo4j is running and accessible")
        print("2. Database credentials are correct")
        print("3. Ultimate Analyzer has been run to populate the database")

# Utility functions for programmatic use
def quick_search(term: str, repository: str = None, uri: str = "bolt://localhost:7687", 
                user: str = "neo4j", password: str = "password") -> List[Dict]:
    """Quick search function for external use"""
    queries = UltimateMultiRepoQueries(uri, user, password)
    results = queries.search_files_by_name(term, repository)
    queries.close()
    return results

def get_important_files(repository: str = None, limit: int = 10, uri: str = "bolt://localhost:7687",
                       user: str = "neo4j", password: str = "password") -> List[Dict]:
    """Get important files for external use"""
    queries = UltimateMultiRepoQueries(uri, user, password)
    results = queries.find_most_important_files(repository, limit)
    queries.close()
    return results

def compare_repositories(uri: str = "bolt://localhost:7687", user: str = "neo4j", 
                        password: str = "password") -> List[Dict]:
    """Compare repositories for external use"""
    queries = UltimateMultiRepoQueries(uri, user, password)
    results = queries.get_repository_comparison()
    queries.close()
    return results

def get_database_summary(uri: str = "bolt://localhost:7687", user: str = "neo4j", 
                        password: str = "password") -> Dict[str, Any]:
    """Get database summary for external use"""
    queries = UltimateMultiRepoQueries(uri, user, password)
    results = queries.get_database_overview()
    queries.close()
    return results

if __name__ == "__main__":
    main()
