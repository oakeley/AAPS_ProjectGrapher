#!/usr/bin/env python3
"""
Memory-Optimized Neo4j Query Utilities for AAPS EatingNow Debugging
Designed to work efficiently with limited memory and smaller datasets
"""

import json
from typing import Dict, List, Any
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimizedDebugQueries:
    """Memory-efficient debugging queries for AAPS project"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query_safe(self, query: str, parameters: Dict = None, limit: int = 20) -> List[Dict]:
        """Execute a query safely with memory limits"""
        try:
            with self.driver.session() as session:
                # Add LIMIT to prevent memory issues
                if "LIMIT" not in query.upper():
                    query += f" LIMIT {limit}"
                
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get basic database statistics"""
        stats = {}
        
        # Count nodes
        result = self.execute_query_safe("MATCH (f:File) RETURN count(f) as files")
        stats['files'] = result[0]['files'] if result else 0
        
        result = self.execute_query_safe("MATCH (fn:Function) RETURN count(fn) as functions")
        stats['functions'] = result[0]['functions'] if result else 0
        
        result = self.execute_query_safe("MATCH ()-[c:CALLS]->() RETURN count(c) as calls")
        stats['calls'] = result[0]['calls'] if result else 0
        
        return stats
    
    def find_most_important_files(self) -> List[Dict]:
        """Find files with highest importance scores"""
        query = """
        MATCH (f:File)
        WHERE f.importance_score IS NOT NULL
        RETURN f.name as filename, f.package as package,
               f.importance_score as importance,
               f.function_count as functions,
               f.lines_of_code as loc
        ORDER BY f.importance_score DESC
        """
        return self.execute_query_safe(query, limit=15)
    
    def find_most_connected_files(self) -> List[Dict]:
        """Find files with most connections (incoming + outgoing)"""
        query = """
        MATCH (f:File)
        OPTIONAL MATCH (f)-[out:CALLS]->()
        OPTIONAL MATCH ()-[in:CALLS]->(f)
        WITH f, count(DISTINCT out) as outgoing, count(DISTINCT in) as incoming
        WHERE outgoing > 0 OR incoming > 0
        RETURN f.name as filename, f.package as package,
               outgoing, incoming,
               (outgoing + incoming) as total_connections
        ORDER BY total_connections DESC
        """
        return self.execute_query_safe(query, limit=15)
    
    def find_central_functions(self) -> List[Dict]:
        """Find the most called functions"""
        query = """
        MATCH ()-[c:CALLS]->()
        WITH c.function as function_name, count(*) as call_count
        WHERE call_count > 1
        RETURN function_name, call_count
        ORDER BY call_count DESC
        """
        return self.execute_query_safe(query, limit=15)
    
    def find_package_overview(self) -> List[Dict]:
        """Get overview by package"""
        query = """
        MATCH (f:File)
        WHERE f.package IS NOT NULL AND f.package <> 'unknown'
        WITH f.package as package, 
             count(f) as file_count,
             sum(f.function_count) as total_functions,
             sum(f.lines_of_code) as total_loc,
             avg(f.importance_score) as avg_importance
        WHERE file_count > 1
        RETURN package, file_count, total_functions, total_loc, 
               round(avg_importance, 2) as avg_importance
        ORDER BY avg_importance DESC
        """
        return self.execute_query_safe(query, limit=20)
    
    def find_call_chains(self, max_depth: int = 3) -> List[Dict]:
        """Find call chains with limited depth"""
        query = f"""
        MATCH path = (start:File)-[:CALLS*1..{max_depth}]->(end:File)
        WHERE start <> end
        WITH path, length(path) as chain_length
        WHERE chain_length >= 2
        RETURN [node in nodes(path) | node.name] as call_chain,
               chain_length
        ORDER BY chain_length DESC
        """
        return self.execute_query_safe(query, limit=10)
    
    def find_isolated_files(self) -> List[Dict]:
        """Find files with no connections"""
        query = """
        MATCH (f:File)
        WHERE NOT (f)-[:CALLS]->() AND NOT ()-[:CALLS]->(f)
        RETURN f.name as filename, f.package as package,
               f.function_count as functions, f.lines_of_code as loc
        ORDER BY f.lines_of_code DESC
        """
        return self.execute_query_safe(query, limit=10)
    
    def search_files_by_name(self, search_term: str) -> List[Dict]:
        """Search files by name pattern"""
        query = """
        MATCH (f:File)
        WHERE toLower(f.name) CONTAINS toLower($search_term)
           OR toLower(f.package) CONTAINS toLower($search_term)
        RETURN f.name as filename, f.package as package,
               f.function_count as functions, f.lines_of_code as loc,
               f.importance_score as importance
        ORDER BY f.importance_score DESC
        """
        return self.execute_query_safe(query, {'search_term': search_term}, limit=15)
    
    def find_files_by_type(self, file_type: str) -> List[Dict]:
        """Find files by type (java/kotlin)"""
        query = """
        MATCH (f:File)
        WHERE f.file_type = $file_type
        RETURN f.name as filename, f.package as package,
               f.function_count as functions, f.lines_of_code as loc
        ORDER BY f.importance_score DESC
        """
        return self.execute_query_safe(query, {'file_type': file_type}, limit=15)
    
    def generate_compact_report(self) -> Dict[str, Any]:
        """Generate a compact debugging report"""
        logger.info("Generating compact debugging report...")
        
        report = {
            "database_stats": self.get_database_stats(),
            "most_important_files": self.find_most_important_files()[:10],
            "most_connected_files": self.find_most_connected_files()[:10],
            "central_functions": self.find_central_functions()[:10],
            "package_overview": self.find_package_overview()[:10],
            "call_chains": self.find_call_chains()[:5],
            "isolated_files": self.find_isolated_files()[:5]
        }
        
        return report

class CompactKnowledgeGraph:
    """Compact knowledge graph for efficient queries"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.debug_queries = MemoryOptimizedDebugQueries(uri, user, password)
    
    def close(self):
        self.debug_queries.close()
    
    def search_by_concept(self, concept: str) -> List[Dict]:
        """Search for files related to a concept"""
        return self.debug_queries.search_files_by_name(concept)
    
    def get_file_details(self, filename: str) -> Dict[str, Any]:
        """Get detailed information about a specific file"""
        query = """
        MATCH (f:File {name: $filename})
        OPTIONAL MATCH (f)-[out:CALLS]->(target:File)
        OPTIONAL MATCH (source:File)-[in:CALLS]->(f)
        RETURN f.name as filename, f.path as path, f.package as package,
               f.function_count as functions, f.lines_of_code as loc,
               f.importance_score as importance,
               collect(DISTINCT target.name) as calls_to,
               collect(DISTINCT source.name) as called_by
        """
        
        result = self.debug_queries.execute_query_safe(query, {"filename": filename}, limit=1)
        return result[0] if result else {}
    
    def get_package_files(self, package: str) -> List[Dict]:
        """Get all files in a specific package"""
        query = """
        MATCH (f:File)
        WHERE f.package CONTAINS $package
        RETURN f.name as filename, f.package as full_package,
               f.function_count as functions, f.lines_of_code as loc,
               f.importance_score as importance
        ORDER BY f.importance_score DESC
        """
        
        return self.debug_queries.execute_query_safe(query, {"package": package}, limit=20)

def main():
    """Main execution with memory-optimized queries"""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change to your Neo4j password
    
    try:
        # Initialize tools
        debug_queries = MemoryOptimizedDebugQueries(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        knowledge_graph = CompactKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Check database connectivity
        stats = debug_queries.get_database_stats()
        if stats['files'] == 0:
            print("⚠️  Database appears to be empty. Run the analyzer first!")
            return
        
        # Generate compact report
        print("Generating compact debugging report...")
        report = debug_queries.generate_compact_report()
        
        # Save report
        with open('aaps_compact_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display results
        print("\n" + "="*60)
        print("🩸 AAPS EatingNow - Compact Analysis Report")
        print("="*60)
        
        print(f"\n📊 DATABASE OVERVIEW:")
        print(f"  Files: {stats['files']}")
        print(f"  Functions: {stats['functions']}")
        print(f"  Function Calls: {stats['calls']}")
        
        print(f"\n🏆 MOST IMPORTANT FILES:")
        for i, file_info in enumerate(report['most_important_files'][:5], 1):
            importance = file_info.get('importance', 0)
            functions = file_info.get('functions', 0)
            print(f"  {i}. {file_info['filename']}")
            print(f"     Package: {file_info.get('package', 'unknown')}")
            print(f"     Importance: {importance:.1f}, Functions: {functions}")
        
        print(f"\n🔗 MOST CONNECTED FILES:")
        for i, file_info in enumerate(report['most_connected_files'][:5], 1):
            total = file_info.get('total_connections', 0)
            incoming = file_info.get('incoming', 0)
            outgoing = file_info.get('outgoing', 0)
            print(f"  {i}. {file_info['filename']}")
            print(f"     Connections: {total} (in: {incoming}, out: {outgoing})")
        
        print(f"\n🎯 MOST CALLED FUNCTIONS:")
        for i, func_info in enumerate(report['central_functions'][:5], 1):
            print(f"  {i}. {func_info['function_name']}: {func_info['call_count']} calls")
        
        print(f"\n📦 TOP PACKAGES:")
        for i, pkg_info in enumerate(report['package_overview'][:5], 1):
            files = pkg_info.get('file_count', 0)
            importance = pkg_info.get('avg_importance', 0)
            print(f"  {i}. {pkg_info['package']}")
            print(f"     Files: {files}, Avg Importance: {importance}")
        
        if report['call_chains']:
            print(f"\n🔄 CALL CHAINS:")
            for i, chain in enumerate(report['call_chains'][:3], 1):
                chain_str = ' → '.join(chain['call_chain'])
                print(f"  {i}. {chain_str} (length: {chain['chain_length']})")
        
        if report['isolated_files']:
            print(f"\n🏝️  ISOLATED FILES:")
            for file_info in report['isolated_files'][:3]:
                print(f"  • {file_info['filename']} ({file_info['loc']} LOC)")
        
        print(f"\n💡 USAGE EXAMPLES:")
        print(f"  Search: python -c \"from memory_optimized_utilities import *; kg = CompactKnowledgeGraph('{NEO4J_URI}', '{NEO4J_USER}', '{NEO4J_PASSWORD}'); print(kg.search_by_concept('pump'))\"")
        print(f"  File details: python -c \"from memory_optimized_utilities import *; kg = CompactKnowledgeGraph('{NEO4J_URI}', '{NEO4J_USER}', '{NEO4J_PASSWORD}'); print(kg.get_file_details('YourFile.java'))\"")
        
        print(f"\n📋 Report saved to: aaps_compact_report.json")
        print("="*60)
        
        # Interactive examples
        print(f"\n🔍 INTERACTIVE EXAMPLES:")
        
        # Example 1: Search for algorithm-related files
        algo_files = knowledge_graph.search_by_concept("algorithm")
        if algo_files:
            print(f"\nFiles related to 'algorithm':")
            for file_info in algo_files[:3]:
                print(f"  • {file_info['filename']} (importance: {file_info.get('importance', 0):.1f})")
        
        # Example 2: Search for pump-related files
        pump_files = knowledge_graph.search_by_concept("pump")
        if pump_files:
            print(f"\nFiles related to 'pump':")
            for file_info in pump_files[:3]:
                print(f"  • {file_info['filename']} (importance: {file_info.get('importance', 0):.1f})")
        
        # Example 3: Get details for most important file
        if report['most_important_files']:
            top_file = report['most_important_files'][0]['filename']
            details = knowledge_graph.get_file_details(top_file)
            if details:
                print(f"\nDetails for top file '{top_file}':")
                print(f"  Path: {details.get('path', 'unknown')}")
                print(f"  Package: {details.get('package', 'unknown')}")
                print(f"  Functions: {details.get('functions', 0)}")
                print(f"  Calls to: {len(details.get('calls_to', []))} files")
                print(f"  Called by: {len(details.get('called_by', []))} files")
        
        # Close connections
        debug_queries.close()
        knowledge_graph.close()
        
        print(f"\n✅ Analysis complete! Use the saved report for further investigation.")
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        print(f"❌ Analysis failed: {e}")
        print("Make sure Neo4j is running and the database has been populated.")

# Example usage functions for interactive use
def quick_search(term: str, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
    """Quick search function for interactive use"""
    kg = CompactKnowledgeGraph(uri, user, password)
    results = kg.search_by_concept(term)
    kg.close()
    return results

def file_info(filename: str, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
    """Get file information for interactive use"""
    kg = CompactKnowledgeGraph(uri, user, password)
    info = kg.get_file_details(filename)
    kg.close()
    return info

if __name__ == "__main__":
    main()