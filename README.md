# AAPS EatingNow Project Analysis Setup Guide

This comprehensive analysis system provides deep insights into the AAPS (AndroidAPS) EatingNow project across multiple repositories using maximum system performance, Neo4j graph database, and AI-powered RAG (Retrieval Augmented Generation) capabilities.

## 🚀 System Overview

The system analyzes three key repositories:
- **EN_new** - Latest EatingNow variant (EN-MASTER-NEW branch)
- **EN_old** - Previous EatingNow variant (master branch)  
- **AAPS_source** - Main AndroidAPS source code (nightscout/AndroidAPS)

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

### 3. Neo4j Database Setup

#### Option A: Docker Setup (Recommended)
```bash
# Install Docker first (https://docs.docker.com/get-docker/)

# Clone a basic Neo4j setup or create docker-compose.yml:
docker run -d \
  --name neo4j-aaps \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_server_memory_heap_max__size=8g \
  neo4j:5.15-community

# Verify it's running
docker ps
```

OK, this needs a bit of a clean-up... To make the Neo4J DB you need Docker (see above)

Assuming you have Docker setup then run
```bash
python docker_neo4j_setup.py
```
This will complain that it failed to start so run
```bash
python neo4j_quick_fix.py 
```
Sorry... Haven't merged yet

#### Option B: Native Installation
Download and install Neo4j from https://neo4j.com/download/

### 4. Ollama (Optional - for AI RAG features)
```bash
# Install Ollama: https://ollama.ai/
# Pull a model (example):
ollama pull deepseek-r1:1.5b
```

## 🚀 Quick Start

### 1. Download the Analysis Scripts
Save these three main scripts:
- **`aaps_analyzer.py`** - Ultimate high-performance multi-repository analyzer
- **`neo4j_utilities.py`** - Database exploration and debugging utilities  
- **`ollama_neo4j_rag.py`** - AI-powered question answering system

### 2. Run the Complete Analysis

#### Step 1: Ultimate Analysis (Uses ALL system resources)
```bash
# This will:
# - Clone all 3 repositories in parallel
# - Analyze files using all CPU cores
# - Generate visualizations
# - Populate Neo4j database
python aaps_analyzer.py
```

Expected output:
```
🚀 AAPS ULTIMATE HIGH-PERFORMANCE MULTI-REPOSITORY ANALYZER
💪 MAXIMUM RAM AND CPU UTILIZATION MODE
🖥️  System: 64.0GB RAM, 24 CPU cores
⚡ Configuration: 24 workers, 32 files/chunk
🔥 Progress: 100.0% (5,247/5,247) | Memory: 45.2%
✅ Neo4j populated: 5,247 files, 12,456 relationships
🎉 ULTIMATE HIGH-PERFORMANCE ANALYSIS COMPLETE!
⏱️  Total Time: 342.7 seconds (5.7 minutes)
```

#### Step 2: Explore the Database
```bash
# Interactive database exploration
python neo4j_utilities.py
```

#### Step 3: AI-Powered Analysis (Optional)
```bash
# Start the RAG system for intelligent questioning
python ollama_neo4j_rag.py
```

## 📊 What You'll Get

### 🌐 Interactive Visualizations
1. **`aaps_ultimate_overview.html`** - Multi-repository overview dashboard
2. **`aaps_EN_new_network.html`** - EN_new repository network analysis
3. **`aaps_EN_old_network.html`** - EN_old repository network analysis
4. **`aaps_AAPS_source_network.html`** - AAPS_source repository network analysis
5. **`aaps_ultimate_comparison.html`** - Side-by-side repository comparison

### 🗄️ Neo4j Knowledge Graph
- **5,000+ file nodes** with importance scoring
- **12,000+ relationship edges** showing call dependencies
- **Repository-aware structure** for cross-repository analysis
- **Optimized indexes** for fast querying

### 📈 Analysis Reports
- **`aaps_ultimate_analysis.json`** - Comprehensive analysis data
- **`aaps_ultimate_database_report.json`** - Database exploration report

### 🤖 AI RAG System
- **Intelligent question answering** about the codebase
- **Cross-repository comparisons** 
- **Architectural insights** and recommendations
- **Context-aware responses** using actual project structure

## 🔍 Using the Analysis Tools

### Neo4j Database Exploration

#### Interactive Explorer
```bash
python neo4j_utilities.py

# Commands available:
🔍 Explorer> overview          # Database statistics
🔍 Explorer> repos             # Repository details  
🔍 Explorer> important EN_new  # Top files in EN_new
🔍 Explorer> search glucose    # Search for glucose-related files
🔍 Explorer> compare           # Compare repositories
🔍 Explorer> similarities      # Cross-repository file similarities
```

#### Sample Neo4j Queries

1. **Most important files across all repositories:**
```cypher
MATCH (f:File)
WHERE f.importance_score > 50
RETURN f.name, f.repository, f.importance_score
ORDER BY f.importance_score DESC
LIMIT 20
```

2. **Compare glucose handling across repositories:**
```cypher
MATCH (f:File)
WHERE toLower(f.name) CONTAINS 'glucose' OR toLower(f.package) CONTAINS 'glucose'
RETURN f.repository, count(f) as glucose_files, avg(f.importance_score) as avg_importance
ORDER BY avg_importance DESC
```

3. **Find call relationships within a repository:**
```cypher
MATCH (f1:File {repository: 'EN_new'})-[:CALLS]->(f2:File {repository: 'EN_new'})
RETURN f1.name, f2.name, f1.importance_score, f2.importance_score
ORDER BY f1.importance_score DESC
LIMIT 15
```

4. **Cross-repository file analysis:**
```cypher
MATCH (f1:File), (f2:File)
WHERE f1.name = f2.name AND f1.repository <> f2.repository
RETURN f1.name, f1.repository, f1.importance_score, f2.repository, f2.importance_score
```

### AI RAG System Usage

#### Interactive Mode
```bash
python ollama_neo4j_rag.py

# Example questions:
❓ Compare insulin algorithms between EN_new and AAPS_source
❓ What files handle blood glucose calculations in EN_new?
❓ How does the pump communication work?
❓ What are the main differences between EN_old and EN_new?
❓ Show me the most important files for automation
```

#### Command Line Mode
```bash
# Single question
python ollama_neo4j_rag.py --question "How does the insulin dosing algorithm work?"

# Repository-specific question
python ollama_neo4j_rag.py --question "What are the core pump files?" --repository EN_new
```

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

### Programmatic API Usage

#### Database Queries
```python
from neo4j_utilities import quick_search, get_important_files, compare_repositories

# Quick file search
results = quick_search("pump", "EN_new")

# Get top important files
important = get_important_files("AAPS_source", limit=10)

# Repository comparison
comparison = compare_repositories()
```

#### AI Integration
```python
from ollama_neo4j_rag import UltimateAAPSRAGSystem

# Initialize RAG system
rag = UltimateAAPSRAGSystem("bolt://localhost:7687", "neo4j", "password")

# Ask questions programmatically
answer = rag.answer_question("How does bolus calculation work?")
print(answer)
```

### Custom Analysis Extensions

#### Adding New Metrics
```python
# In aaps_analyzer.py, extend the FileData class:
@dataclass  
class FileData:
    # ... existing fields ...
    security_score: float = 0.0  # Your custom metric
    
# Add custom scoring in _calculate_importance_ultimate()
def _calculate_importance_ultimate(self, ...):
    # ... existing logic ...
    if 'security' in name_lower:
        score += 30  # Boost security-related files
```

#### Custom Neo4j Queries
```python
# Add to neo4j_utilities.py
def find_security_vulnerabilities(self) -> List[Dict]:
    """Find potential security issues"""
    query = """
    MATCH (f:File)
    WHERE toLower(f.name) CONTAINS 'password' 
       OR toLower(f.name) CONTAINS 'token'
       OR toLower(f.name) CONTAINS 'auth'
    RETURN f.name, f.repository, f.importance_score
    ORDER BY f.importance_score DESC
    """
    return self.execute_query_safe(query)
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Solution: Reduce workers or memory usage
   # In aaps_analyzer.py:
   MAX_WORKERS = 8  # Reduce from default
   ```

2. **Neo4j Connection Failed**
   ```bash
   # Check if Neo4j is running
   docker ps | grep neo4j
   
   # Or check native installation
   systemctl status neo4j  # Linux
   ```

3. **Git Clone Issues**
   ```bash
   # Manual clone if needed
   git clone https://github.com/dicko72/AAPS-EatingNow.git aaps_en_new
   git clone https://github.com/nightscout/AndroidAPS.git aaps_source
   ```

4. **Ollama Model Issues**
   ```bash
   # Check available models
   ollama list
   
   # Pull recommended model
   ollama pull deepseek-r1:1.5b
   ```

### Performance Optimization

1. **For Large Systems (64GB+ RAM):**
   - The analyzer automatically scales to use maximum resources
   - Monitor with `htop` to see full CPU utilization

2. **For Smaller Systems (8GB-32GB):**
   ```python
   # Reduce memory usage in aaps_analyzer.py
   MAX_WORKERS = 4
   CHUNK_SIZE = 10
   NEO4J_BATCH_SIZE = 1000
   ```

3. **Neo4j Performance:**
   - Increase heap size: `-e NEO4J_server_memory_heap_max__size=16g`
   - Add more memory: `-e NEO4J_server_memory_pagecache_size=8g`

## 📈 System Scaling

### Small System (8GB RAM, 4 cores)
```
Expected Performance:
- Analysis Time: 15-30 minutes
- Memory Usage: ~6GB
- Files Processed: ~2,000-5,000
```

### Medium System (32GB RAM, 16 cores)  
```
Expected Performance:
- Analysis Time: 5-10 minutes
- Memory Usage: ~25GB
- Files Processed: ~5,000+
```

### Large System (64GB+ RAM, 24+ cores)
```
Expected Performance: 
- Analysis Time: 2-5 minutes
- Memory Usage: ~50GB+
- Files Processed: ~5,000+
- Full utilization of all resources
```

## 🎯 Best Practices

### 1. Workflow
```bash
# Recommended sequence:
1. python aaps_analyzer.py        # Complete analysis
2. python neo4j_utilities.py     # Explore results  
3. python ollama_neo4j_rag.py    # AI-powered insights
```

### 2. Regular Updates
```bash
# Re-run analysis when repositories change
python aaps_analyzer.py  # Will pull latest changes automatically
```

### 3. Query Optimization
- Use specific repository filters: `{repository: 'EN_new'}`
- Add LIMIT clauses for large result sets
- Use importance scores to filter: `WHERE f.importance_score > 25`

## 🤝 Contributing

To extend the analysis system:

1. **Add New Analysis Types:**
   - Extend `UltimateFileAnalyzer` class
   - Add new importance scoring patterns
   - Create custom visualization types

2. **Enhance AI Capabilities:**
   - Add domain-specific knowledge to RAG system
   - Create specialized query templates
   - Improve context retrieval algorithms

3. **Database Extensions:**
   - Add new node types (e.g., `:Bug`, `:Feature`)
   - Create additional relationship types
   - Implement temporal analysis capabilities

## 📚 Additional Resources

- **Neo4j Documentation**: https://neo4j.com/docs/
- **Plotly Visualization**: https://plotly.com/python/
- **Ollama AI Models**: https://ollama.ai/library
- **AAPS Documentation**: https://wiki.aaps.app
- **NetworkX Graph Analysis**: https://networkx.org/documentation/

## 🎉 What Makes This System Unique

1. **Maximum Performance**: Uses ALL available system resources automatically
2. **Multi-Repository Intelligence**: Analyzes relationships across 3 different codebases
3. **AI-Powered**: Includes RAG system for intelligent code exploration
4. **Scalable**: Works on everything from laptops to high-end workstations
5. **Production Ready**: Includes comprehensive error handling and monitoring

The system is designed to give you unprecedented insights into the AAPS codebase structure, helping with debugging, onboarding, and architectural decisions!
