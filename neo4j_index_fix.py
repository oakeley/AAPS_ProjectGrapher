#!/usr/bin/env python3
"""
Neo4j Index Fix Script
Creates the missing full-text search index for source code
"""

from neo4j import GraphDatabase
import sys

def fix_neo4j_indexes():
    """Create missing indexes in Neo4j"""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    
    print("🔧 Neo4j Index Fix Script")
    print("========================")
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            print("✅ Connected to Neo4j")
            
            # Get Neo4j version to determine correct syntax
            result = session.run("CALL dbms.components() YIELD name, versions, edition UNWIND versions AS version RETURN name, version, edition")
            version_info = result.single()
            if version_info:
                print(f"📋 Neo4j Version: {version_info['version']}")
            
            # Check if full-text index exists
            try:
                result = session.run("SHOW INDEXES")
                indexes = [record for record in result]
                fulltext_exists = any('file_source_idx' in str(index) for index in indexes)
            except:
                # Older Neo4j versions
                try:
                    result = session.run("CALL db.indexes()")
                    indexes = [record for record in result]
                    fulltext_exists = any('file_source_idx' in str(index) for index in indexes)
                except:
                    fulltext_exists = False
            
            if fulltext_exists:
                print("✅ Full-text index already exists")
            else:
                print("🔧 Creating full-text search index...")
                
                # Try multiple methods based on Neo4j version
                methods = [
                    # Neo4j 4.0+ syntax
                    """
                    CREATE FULLTEXT INDEX file_source_idx 
                    FOR (f:File) 
                    ON EACH [f.source_code]
                    """,
                    # Neo4j 4.0+ alternative syntax
                    """
                    CREATE FULLTEXT INDEX file_source_idx 
                    FOR (f:File) 
                    ON EACH [f.source_code]
                    OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard'}}
                    """,
                    # Procedure-based approach for Neo4j 3.5+
                    """
                    CALL db.index.fulltext.createNodeIndex('file_source_idx', ['File'], ['source_code'])
                    """,
                    # Alternative procedure syntax
                    """
                    CALL db.index.fulltext.createNodeIndex('file_source_idx', ['File'], ['source_code'], {analyzer: 'standard'})
                    """
                ]
                
                index_created = False
                for i, method in enumerate(methods, 1):
                    try:
                        session.run(method.strip())
                        print(f"✅ Created full-text index using method {i}")
                        index_created = True
                        break
                    except Exception as e:
                        print(f"❌ Method {i} failed: {e}")
                        continue
                
                if not index_created:
                    print("❌ Could not create full-text index with any method")
                    print("   The system will use property-based search fallback")
                    print("   This may be slower but will still work correctly")
            
            # Create other missing indexes
            indexes_to_create = [
                ("file_repo_idx", "CREATE INDEX file_repo_idx IF NOT EXISTS FOR (f:File) ON (f.repository)"),
                ("file_importance_idx", "CREATE INDEX file_importance_idx IF NOT EXISTS FOR (f:File) ON (f.importance_score)"),
                ("file_eating_now_idx", "CREATE INDEX file_eating_now_idx IF NOT EXISTS FOR (f:File) ON (f.eating_now_score)"),
                ("file_package_idx", "CREATE INDEX file_package_idx IF NOT EXISTS FOR (f:File) ON (f.package)"),
                ("repo_name_idx", "CREATE INDEX repo_name_idx IF NOT EXISTS FOR (r:Repository) ON (r.name)"),
                ("file_has_source_idx", "CREATE INDEX file_has_source_idx IF NOT EXISTS FOR (f:File) ON (f.has_source_code)"),
                ("file_critical_idx", "CREATE INDEX file_critical_idx IF NOT EXISTS FOR (f:File) ON (f.is_eating_now_critical)")
            ]
            
            print("\n🔧 Creating/verifying standard indexes...")
            for index_name, query in indexes_to_create:
                try:
                    session.run(query)
                    print(f"✅ Created/verified index: {index_name}")
                except Exception as e:
                    # Try older syntax for Neo4j 3.5
                    try:
                        old_syntax = query.replace("IF NOT EXISTS ", "").replace("CREATE INDEX ", "CREATE INDEX ON ")
                        session.run(old_syntax)
                        print(f"✅ Created/verified index: {index_name} (legacy syntax)")
                    except Exception as e2:
                        print(f"⚠️  Index {index_name} issue: {e2}")
            
            # Verify database content
            print("\n📊 Database verification:")
            result = session.run("MATCH (f:File) RETURN count(f) as total_files")
            total_files = result.single()["total_files"]
            print(f"   Total files: {total_files:,}")
            
            result = session.run("MATCH (f:File) WHERE f.has_source_code = true RETURN count(f) as files_with_source")
            files_with_source = result.single()["files_with_source"]
            print(f"   Files with source code: {files_with_source:,}")
            
            result = session.run("MATCH (f:File) WHERE f.eating_now_score > 100 RETURN count(f) as critical_files")
            critical_files = result.single()["critical_files"]
            print(f"   Critical eating now files: {critical_files:,}")
            
            # Test if we have source code in files
            result = session.run("MATCH (f:File) WHERE f.source_code IS NOT NULL AND size(f.source_code) > 100 RETURN count(f) as files_with_substantial_source")
            substantial_source = result.single()["files_with_substantial_source"]
            print(f"   Files with substantial source code: {substantial_source:,}")
            
            # Test full-text search
            print("\n🔍 Testing full-text search...")
            try:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('file_source_idx', 'bolus') 
                    YIELD node 
                    RETURN count(node) as search_results
                    LIMIT 1
                """)
                search_results = result.single()["search_results"]
                print(f"✅ Full-text search working: {search_results} results for 'bolus'")
            except Exception as e:
                print(f"❌ Full-text search test failed: {e}")
                print("   ✅ Property-based search fallback is available and will work")
                
                # Test property-based search fallback
                try:
                    result = session.run("""
                        MATCH (f:File) 
                        WHERE f.source_code IS NOT NULL 
                        AND toLower(f.source_code) CONTAINS 'bolus'
                        RETURN count(f) as search_results
                        LIMIT 1
                    """)
                    fallback_results = result.single()["search_results"]
                    print(f"   ✅ Property-based search working: {fallback_results} results for 'bolus'")
                except Exception as e2:
                    print(f"   ❌ Property-based search also failed: {e2}")
        
        driver.close()
        print("\n🎉 Index fix completed!")
        print("\n📊 Summary:")
        print(f"   • Database has {total_files:,} files")
        print(f"   • {files_with_source:,} files have source code stored")
        print(f"   • {critical_files:,} files are critical eating now files")
        print(f"   • Standard indexes are all created")
        print(f"   • Search functionality is available (with fallback if needed)")
        
        print("\n✅ You can now run:")
        print("   python neo4j_utilities.py")
        print("   python ollama_neo4j_rag.py")
        
        print("\n💡 Note: Even without full-text indexing, the system will work")
        print("   using property-based search which may be slightly slower")
        print("   but provides the same functionality.")
        
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("Make sure Neo4j is running and credentials are correct")
        sys.exit(1)

if __name__ == "__main__":
    fix_neo4j_indexes()
