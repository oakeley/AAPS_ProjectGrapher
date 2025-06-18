#!/usr/bin/env python3
"""
Direct Cypher Query Tool for AAPS Neo4j Database
Allows running raw Cypher queries directly against the database
"""

from neo4j import GraphDatabase
import json
import sys

class CypherQueryTool:
    """Direct Cypher query execution tool"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            print("✅ Connected to Neo4j database")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            sys.exit(1)
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query: str, parameters: dict = None, limit: int = None) -> list:
        """Execute a Cypher query and return results"""
        try:
            # Only add LIMIT for complete MATCH...RETURN queries
            query_stripped = query.strip()
            query_upper = query_stripped.upper()
            
            # Check if it's a complete query that could benefit from LIMIT
            needs_limit = (
                limit and 
                "LIMIT" not in query_upper and
                "RETURN" in query_upper and
                query_upper.startswith("MATCH") and
                not any(cmd in query_upper for cmd in ["COUNT(", "DELETE", "CREATE", "SET", "REMOVE", "MERGE"])
            )
            
            if needs_limit:
                query = query_stripped + f" LIMIT {limit}"
            else:
                query = query_stripped
            
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []
    
    def format_results(self, results: list, max_results: int = 20) -> None:
        """Format and display query results"""
        if not results:
            print("📭 No results returned")
            return
        
        print(f"📊 Found {len(results)} result(s)")
        if len(results) > max_results:
            print(f"   Showing first {max_results} results (use LIMIT in query for more control)")
            results = results[:max_results]
        
        print("=" * 80)
        
        for i, record in enumerate(results, 1):
            print(f"\n🔍 Result {i}:")
            for key, value in record.items():
                if isinstance(value, str) and len(value) > 500:
                    # Truncate very long strings (like source code)
                    print(f"   {key}: {value[:500]}...")
                    print(f"   [... truncated, total length: {len(value)} characters]")
                elif isinstance(value, (dict, list)):
                    # Pretty print complex data
                    print(f"   {key}: {json.dumps(value, indent=2)}")
                else:
                    print(f"   {key}: {value}")
        
        print("=" * 80)
    
    def interactive_mode(self):
        """Interactive Cypher query mode"""
        print("🔍 AAPS Database - Direct Cypher Query Tool")
        print("🗄️ Execute raw Cypher queries against the enhanced database")
        print("=" * 60)
        print("💡 Examples:")
        print("   MATCH (f:File) WHERE f.eating_now_score > 500 RETURN f.name, f.eating_now_score")
        print("   MATCH (f:File {name: 'BolusCalculatorPlugin.kt'}) RETURN f.source_code")
        print("   MATCH (f:File) WHERE f.has_source_code = true RETURN count(f)")
        print("\n📝 Commands:")
        print("   'quit' or 'exit' - Exit the tool")
        print("   'examples' - Show more query examples")
        print("   'schema' - Show database schema")
        print("   'stats' - Show database statistics")
        print("\n🚀 Start typing your Cypher queries!\n")
        
        query_buffer = []
        
        while True:
            try:
                if not query_buffer:
                    prompt = "cypher> "
                else:
                    prompt = "   ...> "
                
                line = input(prompt).strip()
                
                if line.lower() in ['quit', 'exit', 'q']:
                    break
                elif line.lower() == 'examples':
                    self.show_examples()
                    continue
                elif line.lower() == 'schema':
                    self.show_schema()
                    continue
                elif line.lower() == 'stats':
                    self.show_stats()
                    continue
                elif not line:
                    if query_buffer:
                        # Execute multi-line query
                        query = ' '.join(query_buffer)
                        print(f"\n🔍 Executing: {query}")
                        results = self.execute_query(query, limit=25)
                        self.format_results(results)
                        query_buffer = []
                    continue
                
                query_buffer.append(line)
                
                # Check if query looks complete (ends with semicolon or contains RETURN)
                full_query = ' '.join(query_buffer)
                if (line.endswith(';') or 
                    ('RETURN' in full_query.upper() and full_query.upper().startswith('MATCH')) or
                    (len(query_buffer) == 1 and any(full_query.upper().startswith(cmd) 
                     for cmd in ['SHOW', 'CALL'])) or
                    (not full_query.upper().startswith('MATCH') and any(full_query.upper().startswith(cmd)
                     for cmd in ['RETURN', 'CREATE', 'DELETE', 'SET']))):
                    
                    # Remove semicolon if present
                    query = full_query.rstrip(';')
                    print(f"\n🔍 Executing: {query}")
                    results = self.execute_query(query, limit=25)
                    self.format_results(results)
                    query_buffer = []
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                query_buffer = []
        
        print("👋 Goodbye!")
    
    def show_examples(self):
        """Show example queries"""
        examples = [
            ("Find top eating now files", 
             "MATCH (f:File) WHERE f.eating_now_score > 200 RETURN f.name, f.repository, f.eating_now_score ORDER BY f.eating_now_score DESC"),
            
            ("Get source code for a specific file", 
             "MATCH (f:File {name: 'BolusCalculatorPlugin.kt'}) RETURN f.name, f.repository, f.source_code"),
            
            ("Find files with 'bolus' in source code", 
             "MATCH (f:File) WHERE f.source_code IS NOT NULL AND toLower(f.source_code) CONTAINS 'bolus' RETURN f.name, f.repository, f.eating_now_score"),
            
            ("Count files by repository", 
             "MATCH (f:File) RETURN f.repository, count(f) as file_count ORDER BY file_count DESC"),
            
            ("Find files with high eating now scores in EN_new", 
             "MATCH (f:File {repository: 'EN_new'}) WHERE f.eating_now_score > 300 RETURN f.name, f.eating_now_score, f.package"),
            
            ("Show repository statistics", 
             "MATCH (r:Repository) RETURN r.name, r.file_count, r.avg_eating_now_score, r.files_with_source_code"),
            
            ("Find files that call other files", 
             "MATCH (f1:File)-[c:CALLS]->(f2:File) RETURN f1.name, f2.name, c.weight ORDER BY c.weight DESC"),
            
            ("Find eating now critical files with source code", 
             "MATCH (f:File) WHERE f.is_eating_now_critical = true AND f.has_source_code = true RETURN f.name, f.repository, f.eating_now_score"),
            
            ("Search for files containing specific functions", 
             "MATCH (f:File) WHERE 'calculateBolus' IN f.functions RETURN f.name, f.repository, f.functions"),
            
            ("Find packages with highest eating now scores", 
             "MATCH (f:File) WHERE f.package IS NOT NULL RETURN f.package, avg(f.eating_now_score) as avg_score, count(f) as files ORDER BY avg_score DESC")
        ]
        
        print("\n💡 Example Cypher Queries:")
        print("=" * 60)
        for i, (description, query) in enumerate(examples, 1):
            print(f"\n{i}. {description}:")
            print(f"   {query}")
        print("=" * 60)
    
    def show_schema(self):
        """Show database schema"""
        print("\n🗄️ Database Schema:")
        print("=" * 50)
        
        # Show node labels
        results = self.execute_query("CALL db.labels()")
        if results:
            labels = [r['label'] for r in results]
            print(f"📋 Node Labels: {', '.join(labels)}")
        
        # Show relationship types
        results = self.execute_query("CALL db.relationshipTypes()")
        if results:
            rel_types = [r['relationshipType'] for r in results]
            print(f"🔗 Relationship Types: {', '.join(rel_types)}")
        
        # Show File node properties
        print("\n📄 File Node Properties:")
        file_props = [
            "name", "repository", "package", "eating_now_score", "importance_score",
            "has_source_code", "is_eating_now_critical", "source_code", "functions",
            "classes", "imports", "lines_of_code", "function_count", "class_count"
        ]
        for prop in file_props:
            print(f"   • {prop}")
        
        # Show Repository node properties
        print("\n📦 Repository Node Properties:")
        repo_props = [
            "name", "file_count", "total_loc", "total_functions", 
            "avg_eating_now_score", "is_eating_now_repo", "files_with_source_code"
        ]
        for prop in repo_props:
            print(f"   • {prop}")
        
        print("=" * 50)
    
    def show_stats(self):
        """Show database statistics"""
        print("\n📊 Database Statistics:")
        print("=" * 40)
        
        # Node counts
        stats = [
            ("Total Files", "MATCH (f:File) RETURN count(f) as count"),
            ("Total Repositories", "MATCH (r:Repository) RETURN count(r) as count"),
            ("Files with Source Code", "MATCH (f:File) WHERE f.has_source_code = true RETURN count(f) as count"),
            ("Critical Eating Now Files", "MATCH (f:File) WHERE f.is_eating_now_critical = true RETURN count(f) as count"),
            ("Total Relationships", "MATCH ()-[r]->() RETURN count(r) as count")
        ]
        
        for name, query in stats:
            results = self.execute_query(query)
            count = results[0]['count'] if results else 0
            print(f"   {name}: {count:,}")
        
        # Repository breakdown
        print("\n📚 Repository Breakdown:")
        results = self.execute_query("""
            MATCH (r:Repository) 
            RETURN r.name, r.file_count, r.avg_eating_now_score, r.files_with_source_code
            ORDER BY r.avg_eating_now_score DESC
        """)
        for repo in results:
            name = repo.get('r.name', 'Unknown')
            files = repo.get('r.file_count', 0)
            avg_score = repo.get('r.avg_eating_now_score', 0)
            with_source = repo.get('r.files_with_source_code', 0)
            print(f"   📦 {name}: {files:,} files, avg score: {avg_score:.1f}, with source: {with_source:,}")
        
        print("=" * 40)


def main():
    """Main execution"""
    print("🚀 AAPS Database - Direct Cypher Query Tool")
    print("🗄️ Enhanced database with source code access")
    print("=" * 60)
    
    # Initialize query tool
    query_tool = CypherQueryTool()
    
    try:
        # Check if arguments provided for single query
        if len(sys.argv) > 1:
            # Execute single query from command line
            query = ' '.join(sys.argv[1:])
            print(f"🔍 Executing: {query}")
            results = query_tool.execute_query(query, limit=50)
            query_tool.format_results(results, max_results=50)
        else:
            # Interactive mode
            query_tool.interactive_mode()
    
    finally:
        query_tool.close()

if __name__ == "__main__":
    main()
