# AAPS EatingNow Project Analysis Setup Guide

This guide will help you set up the complete analysis environment for creating mind maps and populating a Neo4j RAG database for the AAPS EatingNow project.

## Prerequisites

### 1. Python Environment
```bash
# Create a virtual environment
python -m venv aaps_analysis
source aaps_analysis/bin/activate  # On Windows: aaps_analysis\Scripts\activate

# Install required packages
pip install networkx matplotlib plotly neo4j gitpython pandas scipy aiofiles requests
```

### 2. Neo4j Database
OK, this needs a bit of a clean-up... To make the Neo4J DB you need Docker (go Google how to do that)
Assuming you have Docker setup then run
```bash
python docker_neo4j_setup.py
```
This will complain that it failed to start so run
```bash
python neo4j_quick_fix.py 
```
Sorry... Haven't merged yet

## Quick Start

### 1. Download the Analysis Scripts
Save the provided Python scripts:
- `aaps_analyzer.py` - Main analysis script
- `neo4j_utilities.py` - Debugging and query utilities

### 2. Configure Connection
Edit the connection details in both scripts:
(it will all work fine if you just leave the password as "password" so this is optional)
```python
NEO4J_URI = "bolt://localhost:7687"  # Your Neo4j URI
NEO4J_USER = "neo4j"                 # Your username
NEO4J_PASSWORD = "your_password"     # Your password
```

### 3. Run the Analysis
```bash
# First, run the main analyzer
python aaps_analyzer.py

# Then run the debugging utilities for additional analysis
python neo4j_utilities.py
```

## What You'll Get

### Mind Maps (HTML Files)
1. **File Interaction Map** (`aaps_file_interactions.html`)
   - Shows how files connect to each other
   - Bubble size indicates complexity
   - Lines show function call relationships

2. **Internal Structure Maps** (`aaps_internal_*.html`)
   - One map per file showing internal functions and classes
   - Shows how subroutines connect within each file

3. **Data Flow Map** (`aaps_data_flow.html`)
   - High-level view of how data progresses through the app
   - Shows the logical flow from CGM input to pump output

### Neo4j Knowledge Graph
- Complete project structure stored as a graph database
- Optimized for AI querying and debugging assistance
- Rich metadata about files, functions, and relationships

### Analysis Reports
- **JSON Report** (`aaps_project_analysis.json`) - Raw analysis data
- **Debugging Report** (`aaps_debugging_report.json`) - Issues and insights

## Using the Neo4j Database for Debugging

### Sample Queries

1. **Find the most complex files:**
```cypher
MATCH (f:File)
RETURN f.name, f.function_count, f.lines_of_code
ORDER BY f.function_count DESC
LIMIT 10
```

2. **Find circular dependencies:**
```cypher
MATCH path = (f:File)-[:CALLS*2..6]->(f)
RETURN [node in nodes(path) | node.name] as cycle
```

3. **Find files that call a specific function:**
```cypher
MATCH (f1:File)-[c:CALLS]->(f2:File)
WHERE c.function = "calculateIOB"
RETURN f1.name, f2.name, c.line
```

### Python API Usage

```python
from neo4j_utilities import AAPSDebugQueries, AAPSKnowledgeGraph

# Initialize
debug = AAPSDebugQueries("bolt://localhost:7687", "neo4j", "password")
kg = AAPSKnowledgeGraph("bolt://localhost:7687", "neo4j", "password")

# Find critical files
critical = debug.find_critical_files()

# Search for concept
results = kg.semantic_search("blood glucose algorithm")

# Generate AI context
context = kg.generate_ai_context("debugging pump communication issues")
```

## Customizing for Your Needs

### Adding New Analysis Types
You can extend the `AAPSProjectAnalyzer` class:

```python
def analyze_ui_components(self) -> Dict:
    """Custom analysis for UI components"""
    ui_files = [f for f in self.project.files.values() 
                if 'ui' in f.path.lower() or 'activity' in f.name.lower()]
    return {"ui_files": ui_files}
```

### Custom Neo4j Queries
Add your own debugging queries to the `AAPSDebugQueries` class:

```python
def find_battery_optimization_code(self):
    """Find code related to battery optimization"""
    query = """
    MATCH (f:File)
    WHERE toLower(f.name) CONTAINS 'battery' OR 
          toLower(f.name) CONTAINS 'power' OR
          toLower(f.name) CONTAINS 'wakelock'
    RETURN f.name, f.path, f.function_count
    """
    return self.execute_query(query)
```

### Visualization Customization
Modify the plotting functions to change colors, layouts, or add annotations:

```python
# In create_file_interaction_map()
node_trace = go.Scatter(
    marker=dict(
        size=20, 
        color='red',  # Change color
        colorscale='Viridis',  # Add color scale
        showscale=True
    )
)
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Check if Neo4j is running
   - Verify connection string and credentials
   - Check firewall settings

2. **Git Clone Issues**
   - Ensure you have git installed
   - Check internet connection
   - Verify repository URL and branch name

3. **Memory Issues with Large Projects**
   - Increase Python memory limits
   - Process files in batches
   - Use Neo4j pagination for large queries

4. **Missing Dependencies**
   - Run `pip install -r requirements.txt` if provided
   - Install packages individually as shown in prerequisites

### Performance Optimization

1. **For Large Projects:**
   ```python
   # Process files in batches
   batch_size = 100
   for i in range(0, len(all_files), batch_size):
       batch = all_files[i:i+batch_size]
       process_batch(batch)
   ```

2. **Neo4j Performance:**
   - Create indexes (automatically done by the script)
   - Use LIMIT clauses in queries
   - Consider using APOC procedures for complex operations

## Advanced Usage

### Integration with AI Tools
The Neo4j database is designed to work well with AI coding assistants:

```python
# Generate context for AI
context = kg.generate_ai_context("fix insulin calculation bug")

# Use with your favorite AI tool
prompt = f"""
Based on this AAPS project context:
{context}

Help me debug an issue where insulin calculations are incorrect.
The problem seems to occur during meal bolus calculations.
"""
```

### Continuous Analysis
Set up the analyzer to run periodically:

```python
import schedule
import time

def analyze_project():
    analyzer = AAPSProjectAnalyzer(REPO_URL)
    analyzer.analyze_project()
    # Update Neo4j database

schedule.every().day.at("02:00").do(analyze_project)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

## Contributing

To extend this analysis system:

1. Fork the repository
2. Add new analysis methods
3. Create additional visualization types
4. Add more specialized Neo4j queries
5. Submit pull requests

## Support

For issues with:
- **Neo4j**: Check Neo4j documentation at https://neo4j.com/docs/
- **Plotting**: Refer to Plotly documentation at https://plotly.com/python/
- **AAPS Specific**: Consult AAPS documentation at https://wiki.aaps.app

The analysis system is designed to be a starting point - feel free to customize it for your specific debugging and onboarding needs!
