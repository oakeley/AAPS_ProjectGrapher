#!/usr/bin/env python3
"""
Neo4j Query Utilities for AAPS EatingNow Debugging
Provides helpful queries and analysis functions for debugging the AAPS project
"""

import json
from typing import Dict, List, Any
from neo4j import GraphDatabase
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AAPSDebugQueries:
    """Collection of useful debugging queries for AAPS project"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def find_critical_files(self) -> List[Dict]:
        """Find files that are heavily interconnected (potential bottlenecks)"""
        query = """
        MATCH (f:File)
        OPTIONAL MATCH (f)-[out:CALLS]->()
        OPTIONAL MATCH ()-[in:CALLS]->(f)
        WITH f, count(out) as outgoing_calls, count(in) as incoming_calls
        RETURN f.name as filename, f.path as path,
               outgoing_calls, incoming_calls,
               (outgoing_calls + incoming_calls) as total_connections,
               f.lines_of_code as loc, f.function_count as functions
        ORDER BY total_connections DESC
        LIMIT 20
        """
        return self.execute_query(query)
    
    def find_dependency_chains(self, start_file: str = None, max_depth: int = 5) -> List[Dict]:
        """Find long dependency chains that might indicate complex coupling"""
        if start_file:
            query = f"""
            MATCH path = (start:File {{name: $start_file}})-[:CALLS*1..{max_depth}]->(end:File)
            WHERE start <> end
            RETURN [node in nodes(path) | node.name] as dependency_chain,
                   length(path) as chain_length
            ORDER BY chain_length DESC
            LIMIT 50
            """
            return self.execute_query(query)
    
    def find_circular_dependencies(self) -> List[Dict]:
        """Find circular dependencies (potential design issues)"""
        query = """
        MATCH path = (f:File)-[:CALLS*2..10]->(f)
        RETURN [node in nodes(path) | node.name] as circular_path,
               length(path) as cycle_length
        ORDER BY cycle_length
        """
        return self.execute_query(query)
    
    def find_orphaned_files(self) -> List[Dict]:
        """Find files with no incoming or outgoing calls (potential dead code)"""
        query = """
        MATCH (f:File)
        WHERE NOT (f)-[:CALLS]->() AND NOT ()-[:CALLS]->(f)
        RETURN f.name as filename, f.path as path, f.lines_of_code as loc
        ORDER BY f.lines_of_code DESC
        """
        return self.execute_query(query)
    
    def find_function_hotspots(self) -> List[Dict]:
        """Find functions that are called from many different places"""
        query = """
        MATCH (f1:File)-[c:CALLS]->(f2:File)
        WITH c.function as function_name, collect(DISTINCT f1.name) as callers, count(*) as call_count
        WHERE call_count > 2
        RETURN function_name, callers, call_count
        ORDER BY call_count DESC
        LIMIT 20
        """
        return self.execute_query(query)
    
    def analyze_file_complexity(self) -> List[Dict]:
        """Analyze file complexity metrics"""
        query = """
        MATCH (f:File)
        OPTIONAL MATCH (f)-[out:CALLS]->()
        OPTIONAL MATCH ()-[in:CALLS]->(f)
        WITH f, count(out) as outgoing, count(in) as incoming
        RETURN f.name as filename,
               f.lines_of_code as loc,
               f.function_count as functions,
               f.class_count as classes,
               outgoing as calls_out,
               incoming as calls_in,
               CASE 
                 WHEN f.lines_of_code > 0 THEN toFloat(f.function_count) / f.lines_of_code * 1000
                 ELSE 0 
               END as function_density,
               CASE
                 WHEN f.function_count > 0 THEN toFloat(outgoing) / f.function_count
                 ELSE 0
               END as avg_calls_per_function
        ORDER BY loc DESC
        """
        return self.execute_query(query)
    
    def find_similar_files(self, threshold: float = 0.7) -> List[Dict]:
        """Find files with similar structure (potential refactoring opportunities)"""
        query = """
        MATCH (f1:File), (f2:File)
        WHERE f1 <> f2 AND f1.function_count > 0 AND f2.function_count > 0
        WITH f1, f2,
             abs(f1.function_count - f2.function_count) as func_diff,
             abs(f1.lines_of_code - f2.lines_of_code) as loc_diff,
             abs(f1.class_count - f2.class_count) as class_diff
        WHERE func_diff <= 3 AND loc_diff <= 100 AND class_diff <= 2
        RETURN f1.name as file1, f2.name as file2,
               f1.function_count as f1_functions, f2.function_count as f2_functions,
               f1.lines_of_code as f1_loc, f2.lines_of_code as f2_loc
        ORDER BY func_diff, loc_diff
        LIMIT 20
        """
        return self.execute_query(query)
    
    def trace_data_flow(self, start_function: str, max_depth: int = 6) -> List[Dict]:
        """Trace potential data flow paths starting from a specific function"""
        query = f"""
        MATCH (f:File)-[:CONTAINS]->(fn:Function {{name: $start_function}})
        MATCH path = (f)-[:CALLS*1..{max_depth}]->(target:File)
        RETURN [node in nodes(path) | node.name] as flow_path,
               length(path) as depth,
               [rel in relationships(path) | rel.function] as functions_called
        ORDER BY depth
        LIMIT 30
        """
        return self.execute_query(query, {"start_function": start_function})
    
    def find_potential_entry_points(self) -> List[Dict]:
        """Find files that are likely entry points (few incoming calls, many outgoing)"""
        query = """
        MATCH (f:File)
        OPTIONAL MATCH (f)-[out:CALLS]->()
        OPTIONAL MATCH ()-[in:CALLS]->(f)
        WITH f, count(out) as outgoing, count(in) as incoming
        WHERE incoming <= 2 AND outgoing >= 5
        RETURN f.name as filename, f.path as path,
               incoming as calls_in, outgoing as calls_out,
               f.function_count as functions
        ORDER BY outgoing DESC
        """
        return self.execute_query(query)
    
    def find_error_handling_patterns(self) -> List[Dict]:
        """Find files that might contain error handling based on naming patterns"""
        query = """
        MATCH (f:File)
        WHERE f.name =~ '.*(?i)(error|exception|handler|catch|try).*'
           OR ANY(func IN f.functions WHERE func =~ '.*(?i)(error|exception|handle|catch|try).*')
        RETURN f.name as filename, f.path as path, f.function_count as functions
        ORDER BY f.function_count DESC
        """
        return self.execute_query(query)
    
    def generate_debugging_report(self) -> Dict[str, Any]:
        """Generate a comprehensive debugging report"""
        report = {
            "critical_files": self.find_critical_files()[:10],
            "circular_dependencies": self.find_circular_dependencies(),
            "orphaned_files": self.find_orphaned_files(),
            "function_hotspots": self.find_function_hotspots()[:10],
            "complexity_analysis": self.analyze_file_complexity()[:15],
            "potential_entry_points": self.find_potential_entry_points(),
            "error_handling_files": self.find_error_handling_patterns(),
            "similar_files": self.find_similar_files()[:10]
        }
        
        # Add summary statistics
        summary_query = """
        MATCH (f:File)
        OPTIONAL MATCH ()-[c:CALLS]->()
        RETURN count(f) as total_files,
               sum(f.lines_of_code) as total_loc,
               sum(f.function_count) as total_functions,
               sum(f.class_count) as total_classes,
               count(c) as total_calls
        """
        
        summary = self.execute_query(summary_query)[0]
        report["summary"] = summary
        
        return report

class AAPSKnowledgeGraph:
    """Enhanced knowledge graph functionality for AI assistance"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.debug_queries = AAPSDebugQueries(uri, user, password)
    
    def close(self):
        self.driver.close()
        self.debug_queries.close()
    
    def semantic_search(self, concept: str, context: str = "") -> List[Dict]:
        """Search for files/functions related to a concept"""
        search_terms = concept.lower().split()
        
        query = """
        MATCH (f:File)
        WHERE ANY(term IN $search_terms WHERE 
                  toLower(f.name) CONTAINS term OR 
                  toLower(f.path) CONTAINS term)
        OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)
        WHERE ANY(term IN $search_terms WHERE toLower(fn.name) CONTAINS term)
        RETURN f.name as filename, f.path as path,
               collect(fn.name) as matching_functions,
               f.function_count as total_functions,
               f.lines_of_code as loc
        ORDER BY f.function_count DESC
        """
        
        return self.debug_queries.execute_query(query, {"search_terms": search_terms})
    
    def explain_file_purpose(self, filename: str) -> Dict[str, Any]:
        """Provide detailed information about a specific file for AI context"""
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
               collect(DISTINCT target.name) as calls_to,
               collect(DISTINCT source.name) as called_by
        """
        
        result = self.debug_queries.execute_query(query, {"filename": filename})
        if result:
            return result[0]
        return {}
    
    def find_related_functionality(self, filename: str, depth: int = 2) -> Dict[str, Any]:
        """Find files and functions related to a given file (for context building)"""
        query = f"""
        MATCH (f:File {{name: $filename}})
        OPTIONAL MATCH path1 = (f)-[:CALLS*1..{depth}]->(related:File)
        OPTIONAL MATCH path2 = (related2:File)-[:CALLS*1..{depth}]->(f)
        WITH f, collect(DISTINCT related.name) as downstream,
                collect(DISTINCT related2.name) as upstream
        RETURN f.name as central_file,
               downstream as calls_these_files,
               upstream as called_by_these_files
        """
        
        result = self.debug_queries.execute_query(query, {"filename": filename})
        return result[0] if result else {}
    
    def generate_ai_context(self, query_context: str) -> str:
        """Generate context string optimized for AI debugging assistance"""
        # Parse the query to understand what the user is looking for
        concepts = query_context.lower().split()
        
        context_parts = []
        
        # Add project overview
        summary_query = """
        MATCH (f:File)
        RETURN count(f) as total_files,
               sum(f.lines_of_code) as total_loc,
               sum(f.function_count) as total_functions
        """
        summary = self.debug_queries.execute_query(summary_query)[0]
        
        context_parts.append(f"""
PROJECT OVERVIEW:
- Total Files: {summary['total_files']}
- Total Lines of Code: {summary['total_loc']}
- Total Functions: {summary['total_functions']}
""")
        
        # Find relevant files based on concepts
        relevant_files = []
        for concept in concepts:
            results = self.semantic_search(concept)
            relevant_files.extend(results[:3])  # Top 3 matches per concept
        
        if relevant_files:
            context_parts.append("\nRELEVANT FILES:")
            for file_info in relevant_files[:10]:  # Limit to top 10
                context_parts.append(f"- {file_info['filename']}: {file_info['total_functions']} functions, {file_info['loc']} LOC")
        
        # Add critical files information
        critical_files = self.debug_queries.find_critical_files()[:5]
        if critical_files:
            context_parts.append("\nCRITICAL/CENTRAL FILES:")
            for file_info in critical_files:
                context_parts.append(f"- {file_info['filename']}: {file_info['total_connections']} connections")
        
        # Add potential issues
        circular_deps = self.debug_queries.find_circular_dependencies()
        if circular_deps:
            context_parts.append(f"\nPOTENTIAL ISSUES:")
            context_parts.append(f"- Found {len(circular_deps)} circular dependencies")
        
        orphaned = self.debug_queries.find_orphaned_files()
        if orphaned:
            context_parts.append(f"- Found {len(orphaned)} potentially orphaned files")
        
        return "\n".join(context_parts)

def setup_neo4j_indexes(uri: str, user: str, password: str):
    """Set up indexes for better query performance"""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    indexes = [
        "CREATE INDEX file_name_idx IF NOT EXISTS FOR (f:File) ON (f.name)",
        "CREATE INDEX file_path_idx IF NOT EXISTS FOR (f:File) ON (f.path)",
        "CREATE INDEX function_name_idx IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
        "CREATE INDEX class_name_idx IF NOT EXISTS FOR (c:Class) ON (c.name)",
    ]
    
    with driver.session() as session:
        for index_query in indexes:
            try:
                session.run(index_query)
                logger.info(f"Created index: {index_query}")
            except Exception as e:
                logger.warning(f"Index creation failed or already exists: {e}")
    
    driver.close()

def main():
    """Example usage of the debugging utilities"""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change to your Neo4j password
    
    try:
        # Set up indexes for performance
        setup_neo4j_indexes(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Initialize debugging tools
        debug_queries = AAPSDebugQueries(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        knowledge_graph = AAPSKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Generate comprehensive debugging report
        print("Generating debugging report...")
        report = debug_queries.generate_debugging_report()
        
        # Save report to JSON
        with open('aaps_debugging_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("=== AAPS EatingNow Debugging Report ===\n")
        
        print(f"PROJECT SUMMARY:")
        print(f"  Total Files: {report['summary']['total_files']}")
        print(f"  Total LOC: {report['summary']['total_loc']}")
        print(f"  Total Functions: {report['summary']['total_functions']}")
        print(f"  Total Classes: {report['summary']['total_classes']}")
        print(f"  Total Function Calls: {report['summary']['total_calls']}\n")
        
        print("CRITICAL FILES (Most Connected):")
        for file_info in report['critical_files']:
            print(f"  {file_info['filename']}: {file_info['total_connections']} connections")
        print()
        
        print("FUNCTION HOTSPOTS (Most Called):")
        for hotspot in report['function_hotspots']:
            callers = ', '.join(hotspot['callers'][:3])
            if len(hotspot['callers']) > 3:
                callers += f" (+{len(hotspot['callers'])-3} more)"
            print(f"  {hotspot['function_name']}: {hotspot['call_count']} calls from [{callers}]")
        print()
        
        if report['circular_dependencies']:
            print("⚠️  CIRCULAR DEPENDENCIES FOUND:")
            for cycle in report['circular_dependencies'][:5]:
                path = ' → '.join(cycle['circular_path'])
                print(f"  {path}")
            print()
        
        if report['orphaned_files']:
            print("⚠️  POTENTIALLY ORPHANED FILES:")
            for orphan in report['orphaned_files'][:5]:
                print(f"  {orphan['filename']}: {orphan['loc']} LOC")
            print()
        
        print("POTENTIAL ENTRY POINTS:")
        for entry in report['potential_entry_points']:
            print(f"  {entry['filename']}: {entry['calls_out']} outgoing calls, {entry['calls_in']} incoming")
        print()
        
        # Example of AI context generation
        print("=== AI CONTEXT EXAMPLE ===")
        ai_context = knowledge_graph.generate_ai_context("blood glucose calculation loop")
        print(ai_context[:500] + "..." if len(ai_context) > 500 else ai_context)
        
        # Example queries for different debugging scenarios
        print("\n=== EXAMPLE DEBUGGING QUERIES ===")
        
        # Find files related to "algorithm" or "calculation"
        algo_files = knowledge_graph.semantic_search("algorithm calculation")
        if algo_files:
            print("\nFiles related to algorithms/calculations:")
            for file_info in algo_files[:3]:
                print(f"  {file_info['filename']}: {file_info['total_functions']} functions")
        
        # Close connections
        debug_queries.close()
        knowledge_graph.close()
        
        print(f"\nDebugging report saved to: aaps_debugging_report.json")
        print("Use the AAPSDebugQueries and AAPSKnowledgeGraph classes for interactive debugging!")
        
    except Exception as e:
        logger.error(f"Error running debugging analysis: {e}")
        print("Make sure Neo4j is running and the project has been analyzed first.")

# Example Cypher queries for common debugging scenarios
EXAMPLE_QUERIES = {
    "find_main_loop": """
        MATCH (f:File)
        WHERE toLower(f.name) CONTAINS 'loop' OR 
              toLower(f.name) CONTAINS 'main' OR
              toLower(f.name) CONTAINS 'service'
        RETURN f.name, f.function_count, f.lines_of_code
        ORDER BY f.function_count DESC
    """,
    
    "find_pump_communication": """
        MATCH (f:File)
        WHERE toLower(f.name) CONTAINS 'pump' OR 
              toLower(f.name) CONTAINS 'comm' OR
              toLower(f.name) CONTAINS 'bluetooth'
        RETURN f.name, f.path, f.function_count
    """,
    
    "find_data_processing": """
        MATCH (f:File)
        WHERE toLower(f.name) CONTAINS 'data' OR 
              toLower(f.name) CONTAINS 'process' OR
              toLower(f.name) CONTAINS 'bg' OR
              toLower(f.name) CONTAINS 'glucose'
        RETURN f.name, f.function_count, f.lines_of_code
        ORDER BY f.lines_of_code DESC
    """,
    
    "trace_from_cgm_input": """
        MATCH (f:File)
        WHERE toLower(f.name) CONTAINS 'cgm' OR 
              toLower(f.name) CONTAINS 'dexcom' OR
              toLower(f.name) CONTAINS 'sensor'
        MATCH path = (f)-[:CALLS*1..5]->(target:File)
        RETURN [node in nodes(path) | node.name] as processing_chain,
               length(path) as steps
        ORDER BY steps
        LIMIT 20
    """
}

if __name__ == "__main__":
    main()_length DESC
            LIMIT 50
            """
            return self.execute_query(query, {"start_file": start_file})
        else:
            query = f"""
            MATCH path = (start:File)-[:CALLS*1..{max_depth}]->(end:File)
            WHERE start <> end
            RETURN [node in nodes(path) | node.name] as dependency_chain,
                   length(path) as chain_length
            ORDER BY chain
