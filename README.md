# AAPS EatingNow Project Analysis Setup Guide

This comprehensive analysis system provides deep insights into the AAPS (AndroidAPS) EatingNow project across multiple repositories using maximum system performance, Neo4j graph database, and AI-powered RAG (Retrieval Augmented Generation) capabilities.

## 🚀 System Overview

The system analyzes three key repositories:
- **EN_new** - Latest EatingNow variant (EN-MASTER-NEW branch) 
- **EN_old** - Previous EatingNow variant (master branch) 
- **AAPS_source** - Main AndroidAPS source code (nightscout/AndroidAPS)

## ✨ Enhanced Features (LATEST VERSION)

### 🍽️ Eating Now Prioritization
- **Complete source code storage** for ALL files (99.99% coverage)
- **Massive scoring boosts** for eating now functionality (EN_new: +250, EN_old: +150 points)
- **Intelligent plugin templates** - Automatic discovery of reusable eating now patterns
- **Code generation and caching** - AI generates code based on existing implementations

### 💾 Complete Source Code Integration
- **Full source code storage** - ALL 9,400+ files have complete source code stored
- **Full-text search** within source code content (with property-based fallback)
- **Code snippet extraction** for functions and classes
- **Template discovery** for plugin development
- **Automatic code caching** when AI generates new code

### 🤖 Enhanced AI Capabilities
- **Eating now focused responses** with actual source code examples
- **Plugin development assistance** with working code templates from real files
- **Automatic code detection** and persistent caching
- **Repository-aware suggestions** prioritizing eating now implementations

### 🗄️ Direct Database Access
- **Interactive explorer** with eating now focused commands
- **Direct Cypher query tool** for advanced database queries
- **Property-based search fallback** when full-text indexing isn't available
- **Complete schema documentation** and examples

## 📋 Prerequisites

### 1. System Requirements
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 32GB+ RAM, 16+ CPU cores
- **Optimal**: 64GB+ RAM, 24+ CPU cores (the system scales automatically)

### 2. Python Environment
```bash
# Create virtual environment
python -m venv aaps_analysis
source aaps_analysis/bin/activate  # On Windows: aaps_analysis\Scripts\activate

# Install required packages
pip install networkx matplotlib plotly neo4j gitpython pandas scipy aiofiles requests psutil
```

### 3. Neo4j Database Setup (CRITICAL - Follow Exact Order)

#### Option A: Docker Setup (Recommended)
```bash
# Install Docker first (https://docs.docker.com/get-docker/)

# Create and start Neo4j container
docker run -d \
  --name neo4j-aaps \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_server_memory_heap_max__size=8g \
  neo4j:5.15-community

# Verify it's running
docker ps
```

#### Option B: Native Installation
Download and install Neo4j from https://neo4j.com/download/

#### CRITICAL: Database Setup Scripts (Run in Order), I know this is ugly but I had to hack the DB quite a lot to add new features

**Step 1: Create the database container**
```bash
python docker_neo4j_setup.py
```

**Step 2: Apply quick fixes**
```bash
python neo4j_quick_fix.py 
```

**Step 3: Fix indexes**
```bash
python neo4j_index_fix.py
```

### 4. Ollama (Optional - for AI RAG features)
```bash
# Install Ollama: https://ollama.ai/
# Pull a model (example):
ollama pull deepseek-r1:1.5b
```

## 🚀 Quick Start

### 1. Download All Analysis Scripts
Save these enhanced scripts:
- **`aaps_analyzer.py`** - Enhanced multi-repository analyzer with FULL source code storage
- **`neo4j_utilities.py`** - Interactive database exploration with eating now focus
- **`ollama_neo4j_rag.py`** - AI-powered question answering with code generation and caching
- **`cypher_query_tool.py`** - Direct Cypher query execution tool

### 2. Run the Complete Enhanced Analysis

#### Step 1: Populate Database with Full Source Code Storage for RAG queries
```bash
# This will:
# - Clone all 3 repositories in parallel
# - Analyze ALL files with eating now focus
# - Store COMPLETE source code for all files in Neo4j
# - Generate enhanced visualizations with eating now priority
# - Populate Neo4j database with full source code and metadata

# Delete the old downloads
rm -rf aaps_en_new
rm -rf aaps_en_old
rm -rf aaps_source

# Populate the database with the newest versions of the repositories
python aaps_analyzer.py
```

Expected output:
```
🚀 AAPS ENHANCED MULTI-REPOSITORY ANALYZER
🧠 FULL SOURCE CODE STORAGE + EATING NOW PRIORITIZATION
💾 COMPLETE FILE INDEXING FOR MAXIMUM RAG PERFORMANCE
================================================================================
🖥️  System: 377.2GB RAM, 48 CPU cores
⚡ Configuration: 48 workers, 183 files/chunk
💾 Memory Target: 330.4GB (87.6%)
🧠 Storage Strategy: Full source code for ALL files
🍽️ Enhanced Features: Eating now scoring, full source storage, enhanced indexing
================================================================================
2025-06-18 18:46:44,928 - INFO - 🔄 Cloning EN_new...
2025-06-18 18:46:44,928 - INFO - 🔄 Cloning EN_old...
2025-06-18 18:46:44,929 - INFO - 🔄 Cloning AAPS_source...
2025-06-18 18:46:49,801 - INFO - ✅ EN_old cloned successfully
2025-06-18 18:47:13,625 - INFO - ✅ EN_new cloned successfully
2025-06-18 18:47:25,264 - INFO - ✅ AAPS_source cloned successfully
2025-06-18 18:47:25,265 - INFO - ✅ Successfully cloned 3/3 repositories
2025-06-18 18:47:25,304 - INFO - 📁 EN_new: 3214 source files
2025-06-18 18:47:25,345 - INFO - 📁 EN_old: 3024 source files
2025-06-18 18:47:25,382 - INFO - 📁 AAPS_source: 3208 source files
2025-06-18 18:47:25,382 - INFO - 🎯 Total files to process: 9446
2025-06-18 18:47:26,578 - INFO - 🔥 Progress: 94.2% (8897/9446) | Memory: 2.9%
2025-06-18 18:47:27,575 - INFO - 🔥 Progress: 96.1% (9080/9446) | Memory: 2.9%
2025-06-18 18:47:27,781 - INFO - 🔥 Progress: 98.1% (9263/9446) | Memory: 3.0%
2025-06-18 18:47:27,939 - INFO - 🔥 Progress: 100.0% (9446/9446) | Memory: 3.0%
2025-06-18 18:47:27,967 - INFO - ✅ Enhanced analysis completed in 43.04 seconds
2025-06-18 18:47:27,967 - INFO - 📊 Successfully processed 9446 files
2025-06-18 18:47:27,967 - INFO - 🔗 Building function mappings and call graphs...
2025-06-18 18:47:28,393 - INFO - 📊 EN_new call graph: 3107 nodes, 306979 edges
2025-06-18 18:47:28,722 - INFO - 📊 EN_old call graph: 2945 nodes, 293318 edges
2025-06-18 18:47:29,065 - INFO - 📊 AAPS_source call graph: 3101 nodes, 304461 edges
2025-06-18 18:47:29,068 - INFO - 📈 Generating enhanced outputs...
2025-06-18 18:47:29,068 - INFO - 📄 Generating enhanced JSON report...
2025-06-18 18:47:29,080 - INFO - ✅ Saved: aaps_enhanced_analysis.json
2025-06-18 18:47:29,080 - INFO - 📊 Source code stored for 9445 files
2025-06-18 18:47:29,080 - INFO - 🗄️ Populating Neo4j with enhanced approach...
2025-06-18 18:47:30,576 - INFO - 📊 Neo4j: Processed 9446 files...
⏱️  Total Time: 4.16 seconds (0.1 minutes)
📊 Files Processed: 9,446
🍽️ Eating Now Relevant Files: 7,139
💾 Files with Source Code Stored: 9,445 (99.99% coverage)
✅ Enhanced Neo4j populated: 9,446 files with full source code
```

#### Step 2: Fix Database Indexes (if needed - shouldn't be if you did it at the start)
```bash
# Run the index fix to optimize database performance
python neo4j_index_fix.py
```

Expected output:
```
🔧 Neo4j Index Fix Script
✅ Connected to Neo4j
📊 Database verification:
   Total files: 9,446
   Files with source code: 9,445
   Critical eating now files: 7,117
✅ Property-based search working: 1,234 results for 'bolus'
🎉 Index fix completed!
```

#### Step 3: Explore the Enhanced Database

**Interactive Explorer (Eating Now Focused):**
```bash
python neo4j_utilities.py
```

Enhanced commands available:
```bash
🔍 Enhanced Explorer> eating EN_new    # Top eating now files in EN_new
🔍 Enhanced Explorer> source BolusCalculatorPlugin.kt  # Show complete source code
🔍 Enhanced Explorer> search "bolus calculation"  # Full-text search in source
🔍 Enhanced Explorer> templates bolus  # Find bolus plugin templates
🔍 Enhanced Explorer> architecture     # Show eating now architecture
🔍 Enhanced Explorer> overview        # Database overview with eating now metrics
```

**Direct Cypher Query Tool (NEW):**
```bash
python cypher_query_tool.py
```

Execute raw Cypher queries:
```cypher
cypher> MATCH (f:File) WHERE f.eating_now_score > 500 
   ...> RETURN f.name, f.repository, f.eating_now_score 
   ...> ORDER BY f.eating_now_score DESC

cypher> MATCH (f:File {name: 'BolusCalculatorPlugin.kt'}) 
   ...> RETURN f.source_code

cypher> stats    # Show database statistics
cypher> examples # Show example queries
cypher> schema   # Show database schema
```

#### Step 4: Enhanced AI-Powered Analysis with Code Generation
```bash
# Start the enhanced RAG system with automatic code caching
python ollama_neo4j_rag.py
```

New capabilities:
```bash
🍽️ Ask about AAPS eating now: Create a bolus plugin for AAPS_source using EN_new templates
🍽️ Ask about AAPS eating now: Show me carb counting source code from EN_new with examples
🍽️ Ask about AAPS eating now: Generate meal timing functions based on existing code
🍽️ Ask about AAPS eating now: source BolusCalculatorPlugin.kt  # View source directly
🍽️ Ask about AAPS eating now: cache  # See generated code from this session
```

## 📊 What You'll Get (ENHANCED)

### 🌐 Interactive Visualizations
1. **`aaps_enhanced_overview.html`** - Multi-repository overview with eating now metrics and source code indicators
2. **`aaps_enhanced_eating_now.html`** - Eating now specific analysis with source code availability
3. **Previous visualizations** - All original network analysis files still generated

### 🗄️ Enhanced Neo4j Knowledge Graph
- **9,400+ file nodes** with eating now scoring and **COMPLETE source code** (99.99% coverage)
- **Full-text search indexes** on source code content (with property-based fallback)
- **Enhanced relationship edges** showing call dependencies with eating now context
- **Repository nodes** with eating now metrics and source code statistics
- **Optimized indexes** for fast querying and search

### 📈 Enhanced Analysis Reports
- **`aaps_enhanced_analysis.json`** - Comprehensive analysis with full source code metadata
- **`aaps_enhanced_database_report.json`** - Database exploration report with eating now insights

### 🤖 Enhanced AI RAG System
- **Eating now prioritized responses** with actual working source code from database
- **Plugin development assistance** with real templates and complete source code
- **Automatic code generation** based on existing eating now implementations
- **Code caching system** - Generated code automatically saved to `./generated_code_cache/`
- **Complete source code integration** - AI responses include full file contents

### 💾 Code Generation and Caching (NEW)
- **Automatic code detection** in AI responses (```kotlin, ```java, inline functions)
- **Smart file naming** based on extracted class/function names
- **Metadata preservation** - Each cached file includes context and timestamp
- **Session tracking** - Summary of generated code at session end
- **Template-based generation** - New code based on existing eating now patterns

## 🔍 Using the Analysis Tools

### Neo4j Database Exploration

#### Interactive Explorer
```bash
python neo4j_utilities.py

# Enhanced commands available:
🔍 Enhanced Explorer> overview          # Database statistics with eating now metrics
🔍 Enhanced Explorer> eating            # Top eating now files globally
🔍 Enhanced Explorer> eating EN_new     # Top eating now files in EN_new
🔍 Enhanced Explorer> source BolusCalculatorPlugin.kt  # Show complete source code
🔍 Enhanced Explorer> search "bolus calculation"  # Full-text search in source code
🔍 Enhanced Explorer> templates eating  # Show eating now plugin templates
🔍 Enhanced Explorer> templates bolus   # Show bolus-specific templates
🔍 Enhanced Explorer> architecture      # Eating now architecture insights
🔍 Enhanced Explorer> examples "carb counting"  # Get code examples
```

#### Direct Cypher Query Tool
```bash
python cypher_query_tool.py

# Execute any Cypher query directly:
cypher> MATCH (f:File) WHERE f.has_source_code = true 
   ...> RETURN count(f) as total_files_with_source

cypher> MATCH (f:File) WHERE f.eating_now_score > 300 
   ...> RETURN f.name, f.repository, f.eating_now_score, f.package
   ...> ORDER BY f.eating_now_score DESC

cypher> stats     # Quick database statistics
cypher> examples  # Show 10 example queries
cypher> schema    # Show database schema
```

#### Neo4j Cypher Queries (Examples)

1. **Top eating now files with source code:**
```cypher
MATCH (f:File)
WHERE f.eating_now_score > 100 AND f.has_source_code = true
RETURN f.name, f.repository, f.eating_now_score, f.package
ORDER BY f.eating_now_score DESC
LIMIT 20
```

2. **Get complete source code for a file:**
```cypher
MATCH (f:File {name: 'BolusCalculatorPlugin.kt'})
RETURN f.name, f.repository, f.source_code, f.eating_now_score
```

3. **Search within source code (property-based fallback):**
```cypher
MATCH (f:File) 
WHERE f.source_code IS NOT NULL 
AND toLower(f.source_code) CONTAINS 'bolus calculation'
RETURN f.name, f.repository, f.eating_now_score
ORDER BY f.eating_now_score DESC
```

4. **Repository comparison with source code metrics:**
```cypher
MATCH (r:Repository) 
RETURN r.name, r.file_count, r.avg_eating_now_score, 
       r.files_with_source_code, r.is_eating_now_repo
ORDER BY r.avg_eating_now_score DESC
```

5. **Files calling eating now functionality:**
```cypher
MATCH (f1:File)-[c:CALLS]->(f2:File)
WHERE f2.eating_now_score > 100
RETURN f1.name, f2.name, c.weight, f2.eating_now_score
ORDER BY c.weight DESC, f2.eating_now_score DESC
```

### AI RAG System Usage

#### Interactive Mode (with Code Generation)
```bash
python ollama_neo4j_rag.py

# Enhanced example questions with full source code access:
🍽️ Create an eating now plugin for AAPS_source using EN_new as template with complete source code
🍽️ Show me how bolus calculation works in EN_new with actual implementation details
🍽️ Generate carb counting functions based on existing source code implementations
🍽️ What eating now files should I use as templates with their complete source code?
🍽️ Compare insulin dosing algorithms between EN_new and EN_old with code examples
🍽️ Write a meal timing plugin using existing eating now patterns and source code

# Enhanced commands:
🍽️ Ask about AAPS eating now: stats     # Eating now database statistics
🍽️ Ask about AAPS eating now: templates # Available plugin templates with source
🍽️ Ask about AAPS eating now: cache     # Show generated code from session
🍽️ Ask about AAPS eating now: source <filename>  # View complete source code
```

#### Command Line Mode
```bash
# Single question with eating now focus and source code context
python ollama_neo4j_rag.py --question "Create a bolus plugin based on EN_new templates with complete source code"

# Repository-specific with eating now priority
python ollama_neo4j_rag.py --question "Generate carb functions with source code" --repository EN_new

# Custom cache directory for generated code
python ollama_neo4j_rag.py --cache-dir "./my_generated_code"
```

## 💾 Database Export and Sharing

### 🎯 Sharing Your Enhanced Database

After running the complete analysis, you can export and share the enhanced database (including full source code) with collaborators.

#### Method 1: Database Dump (Recommended - Includes Full Source Code)

**Step 1: Find Your Docker Container**
```bash
docker ps | grep neo4j
```

**Step 2: Stop and Dump the Database in one Line to Stop it Restarting**
```bash
# Replace 'neo4j-aaps' with your actual container name
docker exec -it neo4j-aaps neo4j stop; docker exec -it neo4j-aaps neo4j-admin database dump neo4j --to-path=/tmp
```

**Step 3: Export Enhanced Database (it won't compress so don't try)**
```bash
# Copy the dump file (includes all source code and eating now data)
docker cp neo4j-aaps:/tmp/neo4j.dump ./aaps_enhanced_database.dump

# Check file size (typically 100MB with full source code)
ls -lh aaps_enhanced_database.dump
```

**Step 4: Restart Database (probably not needed)**
```bash
# Manually restart
docker exec -it neo4j-aaps neo4j start
```

#### Method 2: JSON Export (Source Code Included)

Create `export_enhanced_database.py`:

```python
#!/usr/bin/env python3
"""Export Enhanced AAPS Database with Complete Source Code"""

import json
from neo4j import GraphDatabase

def export_enhanced_database():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    
    export_data = {
        "metadata": {
            "version": "enhanced_v2.0", 
            "eating_now_focused": True,
            "complete_source_code": True,
            "coverage": "99.99%"
        },
        "repositories": [],
        "eating_now_files": [],
        "source_code_samples": []
    }
    
    with driver.session() as session:
        # Export eating now prioritized files WITH source code
        result = session.run("""
            MATCH (f:File) 
            WHERE f.eating_now_score > 50 AND f.has_source_code = true
            RETURN f.name, f.repository, f.package, f.eating_now_score, 
                   f.importance_score, f.source_code, f.key_snippets,
                   f.functions, f.classes, f.is_eating_now_critical
            ORDER BY f.eating_now_score DESC
            LIMIT 1000
        """)
        export_data["eating_now_files"] = [dict(record) for record in result]
        
        # Export repository summaries with source code metrics
        result = session.run("""
            MATCH (r:Repository) 
            RETURN r.name, r.file_count, r.avg_eating_now_score, 
                   r.files_with_source_code, r.is_eating_now_repo
        """)
        export_data["repositories"] = [dict(record) for record in result]
    
    # Save enhanced export with source code
    with open('aaps_enhanced_export_with_source.json', 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(export_data['eating_now_files'])} eating now files with source code")

if __name__ == "__main__":
    export_enhanced_database()
```

### 📤 What to Share with Collaborators

**Complete Package:**
1. **Enhanced database dump** (`aaps_enhanced_database.dump`) - ~100MB with full source code
2. **All enhanced analysis scripts** (5 Python files)
3. **Import instructions** (below)

**Collaborator Import Instructions:**

Create `IMPORT_INSTRUCTIONS.md`:
```markdown
# Import AAPS Database (Full Source Code)

## Prerequisites
- Docker installed
- Python environment with packages

## Quick Import Steps

1. **Download Database Dump from Swiss Transfer:**
   ```bash
   wget https://www.swisstransfer.com/d/32436766-eb63-4b9a-966e-56833b73bf3a
   ```

2. **Start fresh Neo4j container:**
   ```bash
   docker run -d --name neo4j-aaps \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     -e NEO4J_server_memory_heap_max__size=8g \
     neo4j:latest
   ```

3. **Wait for startup, then stop:**
   ```bash
   sleep 30
   docker stop neo4j-aaps
   ```

4. **Copy dump file into container:**
   ```bash
   docker cp ./aaps_enhanced_database.dump neo4j-aaps:/tmp/
   ```

5. **Import the enhanced database:**
   ```bash
   docker exec neo4j-aaps neo4j-admin database load --from-path=/tmp neo4j --overwrite-destination=true
   ```

6. **Start and optimize:**
   ```bash
   docker start neo4j-aaps
   sleep 20
   python neo4j_index_fix.py  # Optimize indexes
   ```

## Verify Enhanced Import Success
- Test interactive explorer: `python neo4j_utilities.py`
- Test direct queries: `python cypher_query_tool.py`
- Verify source code access: `source <filename>` in explorer
- Test AI system: `python ollama_neo4j_rag.py`
```

### 📊 Export Size Comparison (Updated)

| Method | File Size | Contents | Source Code | Ease of Use |
|--------|-----------|----------|-------------|-------------|
| Enhanced Database Dump | 200-400MB | Complete database with full source code | ✅ Complete | ⭐⭐⭐⭐⭐ |
| JSON Export | 50-150MB | Eating now files with source code | ✅ Selective | ⭐⭐⭐⭐ |
| Docker Image | 3-6GB | Complete environment with source | ✅ Complete | ⭐⭐ |

**Recommended: Database Dump** - Complete source code with optimized size

## 🛠️ Configuration

### Database Connection
Edit connection details in all scripts (default works out of the box):
```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j" 
NEO4J_PASSWORD = "password"
```

### Performance Tuning
The analyzer automatically detects and uses:
- **90% of available RAM** (configurable)
- **All CPU cores** (configurable)
- **Dynamic chunk sizing** based on system resources
- **Complete source code storage** for all files

To customize:
```python
# In aaps_analyzer.py
MAX_WORKERS = 16  # Limit CPU cores
MAX_MEMORY_USAGE = int(32 * 1024**3)  # Limit to 32GB RAM
```

### Repository Configuration
To analyze different repositories, modify the `REPOSITORIES` dict in `aaps_analyzer.py`:
```python
REPOSITORIES = {
    "your_repo": {
        "url": "https://github.com/user/repo.git",
        "branch": "main", 
        "local_path": "./your_repo"
    }
}
```

## 🔧 Advanced Usage

### Database Queries with Source Code

#### Using the Interactive Explorer
```python
from neo4j_utilities import EnhancedMultiRepoQueries

# Initialize enhanced queries
queries = EnhancedMultiRepoQueries("bolt://localhost:7687", "neo4j", "password")

# Find top eating now files with source code
eating_now_files = queries.find_top_eating_now_files("EN_new", limit=10)

# Get complete source code for a file
source_data = queries.get_source_code_for_file("BolusCalculatorPlugin.kt")

# Search within source code (with fallback)
search_results = queries.search_source_code("bolus calculation")

# Find plugin templates with source code
templates = queries.find_plugin_templates("eating")
```

#### Using the Direct Cypher Tool
```python
from cypher_query_tool import CypherQueryTool

# Initialize query tool
cypher = CypherQueryTool()

# Execute complex queries
results = cypher.execute_query("""
    MATCH (f:File) 
    WHERE f.eating_now_score > 200 AND f.has_source_code = true
    RETURN f.name, f.repository, f.source_code, f.eating_now_score
    ORDER BY f.eating_now_score DESC
""")

# Format and display results
cypher.format_results(results)
```

#### AI Integration with Full Source Code
```python
from ollama_neo4j_rag import EnhancedAAPSRAGSystem

# Initialize enhanced RAG system
rag = EnhancedAAPSRAGSystem("bolt://localhost:7687", "neo4j", "password")

# Ask enhanced questions with complete source code context
answer = rag.answer_question_enhanced("Create a bolus plugin based on EN_new with complete source code")

# Check for generated code
if rag.code_cache.session_cache:
    print("Generated code files:", list(rag.code_cache.session_cache.keys()))
```

### Custom Analysis Extensions (Enhanced)

#### Adding New Eating Now Patterns
```python
# In aaps_analyzer.py, extend eating now scoring:
def _init_eating_now_patterns(self):
    self.eating_now_critical = {
        'eating': 100, 'eatnow': 100, 'eatingnow': 100,
        'your_custom_pattern': 150,  # Add custom patterns
        # ... existing patterns
    }
```

#### Custom Source Code Analysis
```python
# Add to cypher_query_tool.py or neo4j_utilities.py
def analyze_code_patterns(repository: str = None) -> List[Dict]:
    """Analyze specific patterns in source code"""
    query = """
    MATCH (f:File)
    WHERE f.source_code IS NOT NULL
    AND toLower(f.source_code) CONTAINS 'your_pattern'
    RETURN f.name, f.repository, f.eating_now_score, 
           size(f.source_code) as source_length
    ORDER BY f.eating_now_score DESC
    """
    return execute_query(query)
```

## 🚨 Troubleshooting (Updated)

### Common Issues and Solutions

1. **Database Index Issues**
   ```bash
   # Solution: Run the index fix script
   python neo4j_index_fix.py
   # This handles all index creation with fallbacks
   ```

2. **Source Code Access Issues**
   ```bash
   # Check if source code was stored
   python cypher_query_tool.py
   cypher> MATCH (f:File) WHERE f.has_source_code = true RETURN count(f)
   # Should return close to 9,400+ files
   ```

3. **Query Tool Issues**
   ```bash
   # Make sure queries are complete with both MATCH and RETURN
   # Use multi-line input for complex queries
   cypher> MATCH (f:File) WHERE f.eating_now_score > 100
      ...> RETURN f.name, f.eating_now_score
      ...> ORDER BY f.eating_now_score DESC
   ```

4. **Full-Text Search Not Available**
   ```bash
   # The system automatically falls back to property-based search
   # This works just as well, slightly slower
   # No action needed - fallback is automatic
   ```

5. **Memory Issues During Analysis**
   ```bash
   # Reduce batch size if needed
   # In aaps_analyzer.py:
   NEO4J_BATCH_SIZE = 5000  # Reduce from default
   CHUNK_SIZE = 50  # Reduce chunk size
   ```

### Performance Optimization

1. **For Complete Source Code Storage:**
   ```python
   # The system now stores ALL source code by default
   # No configuration needed for optimal RAG performance
   ```

2. **For Advanced Queries:**
   ```python
   # Use the cypher_query_tool.py for complex analysis
   # Property-based search provides good performance
   ```

3. **For Code Generation:**
   ```python
   # In ollama_neo4j_rag.py, adjust context length for more source code
   max_context_length = 20000  # Increase for more complete source code context
   ```

## 📈 System Scaling

### Small System (8GB RAM, 4 cores)
```
Expected Performance with Complete Source Code:
- Analysis Time: 15-30 minutes
- Memory Usage: ~7GB
- Database Size: ~250MB with complete source code
- Files Processed: ~9,400+
- Source Code Coverage: 95%+
```

### Medium System (32GB RAM, 16 cores)  
```
Expected Performance with Complete Source Code:
- Analysis Time: 5-12 minutes
- Memory Usage: ~28GB
- Database Size: ~350MB with complete source code
- Files Processed: ~9,400+
- Source Code Coverage: 99%+
```

### Large System (64GB+ RAM, 24+ cores)
```
Expected Performance with Complete Source Code:
- Analysis Time: 2-6 minutes
- Memory Usage: ~55GB+
- Database Size: ~400MB with complete source code
- Files Processed: ~9,400+
- Source Code Coverage: 99.99%
- Full source code indexing and optimization
```

## 🎯 Best Practices

### 1. Workflow (Complete Source Code)
```bash
# Recommended sequence for eating now development:
1. python aaps_analyzer.py        # Complete analysis with full source
2. python neo4j_index_fix.py     # Optimize database indexes
3. python neo4j_utilities.py     # Explore eating now results interactively
   - Use 'eating' command to find top files
   - Use 'source <filename>' to view complete source code
   - Use 'templates' to find plugin patterns
4. python cypher_query_tool.py   # Database queries
   - Run complex Cypher queries
   - Analyze source code patterns
   - Extract specific code examples
5. python ollama_neo4j_rag.py    # Generate eating now plugins with full context
   - Ask for plugin creation with complete source code
   - Use 'cache' command to see generated code
```

### 2. Plugin Development Workflow
```bash
# For creating eating now plugins with complete source code:
1. Ask: "What eating now files should I use as templates with complete source code?"
2. Ask: "Show me bolus calculation implementation from EN_new with full source code"
3. Ask: "Create a [feature] plugin based on EN_new templates with complete implementation"
4. Check: Use 'cache' command to see generated code with metadata
5. Refine: Ask follow-up questions for specific functionality with source examples
```

### 3. Source Code Best Practices
- **Complete coverage**: 99.99% of files have source code stored and searchable
- **Use eating now score > 100** for critical template files with guaranteed source access
- **Search capabilities**: Full-text search with automatic property-based fallback
- **Export generated code** regularly for backup and version control
- **Use repository-specific queries**: `{repository: 'EN_new'}` for focused analysis

## 🤝 Contributing

To extend the enhanced analysis system:

1. **Add New Eating Now Patterns:**
   - Extend eating now scoring in `_init_eating_now_patterns()`
   - Add new critical functionality keywords
   - Enhance source code analysis patterns

2. **AI Code Generation:**
   - Add more code detection patterns in `CodeCache`
   - Improve template matching algorithms in `find_eating_now_templates()`
   - Create specialized code generation prompts

3. **Database Extensions:**
   - Add new eating now specific node types
   - Create code similarity relationships using source code analysis
   - Implement version tracking for generated code

4. **Query Tool Enhancements:**
   - Add new built-in commands to `cypher_query_tool.py`
   - Create query templates for common source code analysis tasks
   - Add export functionality for query results

## 📚 Additional Resources

- **Neo4j Documentation**: https://neo4j.com/docs/
- **Cypher Query Language**: https://neo4j.com/docs/cypher-manual/current/
- **Plotly Visualization**: https://plotly.com/python/
- **Ollama AI Models**: https://ollama.ai/library
- **AAPS Documentation**: https://wiki.aaps.app
- **NetworkX Graph Analysis**: https://networkx.org/documentation/

## 🎉 What Makes This System Unique

1. **Complete Source Code Storage**: ALL 9,400+ files have their complete source code stored and searchable
2. **Eating Now Prioritization**: Massive scoring boosts for eating now functionality with real source code
3. **Advanced Query Capabilities**: Both interactive explorer and direct Cypher query tool
4. **AI Code Generation**: Creates working plugins based on complete source code templates
5. **Automatic Code Caching**: Generated code saved with metadata and context for reuse
6. **Property-Based Search Fallback**: Works even without full-text indexing
7. **Great Performance**: Source code storage with optimized indexing and chunking
8. **Plugin Development Focus**: Specifically designed for eating now plugin creation with complete source access
9. **99.99% Coverage
