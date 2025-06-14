#!/usr/bin/env python3
"""
AAPS EatingNow Project Analyzer
Creates graphical mind maps and populates Neo4j RAG database for debugging assistance
"""

import os
import re
import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Required imports (install with pip)
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    from neo4j import GraphDatabase
    import git
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install networkx matplotlib plotly neo4j gitpython")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FunctionCall:
    """Represents a function call relationship"""
    caller: str
    callee: str
    function_name: str
    line_number: int
    context: str = ""

@dataclass
class FileInfo:
    """Represents information about a source file"""
    path: str
    name: str
    functions: List[str]
    classes: List[str]
    imports: List[str]
    calls_made: List[FunctionCall]
    calls_received: List[str]
    lines_of_code: int
    file_type: str

@dataclass
class ProjectStructure:
    """Represents the overall project structure"""
    files: Dict[str, FileInfo]
    dependencies: Dict[str, List[str]]
    call_graph: Dict[str, List[FunctionCall]]

class AAPSProjectAnalyzer:
    """Main analyzer class for AAPS EatingNow project"""
    
    def __init__(self, repo_url: str, local_path: str = "./aaps_eating_now"):
        self.repo_url = repo_url
        self.local_path = Path(local_path)
        self.project = ProjectStructure({}, {}, {})
        
    def clone_repository(self) -> bool:
        """Clone the repository if it doesn't exist"""
        try:
            if not self.local_path.exists():
                logger.info(f"Cloning repository to {self.local_path}")
                git.Repo.clone_from(self.repo_url, self.local_path, branch="EN-MASTER-NEW")
            else:
                logger.info(f"Repository already exists at {self.local_path}")
                # Try to pull latest changes
                repo = git.Repo(self.local_path)
                repo.remotes.origin.pull()
            return True
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            return False
    
    def analyze_java_file(self, file_path: Path) -> FileInfo:
        """Analyze a Java source file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract functions (methods)
            functions = []
            method_pattern = r'(?:public|private|protected|static|\s)*\s+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
            functions = [match.group(1) for match in re.finditer(method_pattern, content)]
            
            # Extract classes
            class_pattern = r'(?:public|private|protected)?\s*class\s+(\w+)'
            classes = [match.group(1) for match in re.finditer(class_pattern, content)]
            
            # Extract imports
            import_pattern = r'import\s+([\w.]+);'
            imports = [match.group(1) for match in re.finditer(import_pattern, content)]
            
            # Extract function calls
            calls_made = self._extract_java_function_calls(content, str(file_path))
            
            return FileInfo(
                path=str(file_path),
                name=file_path.name,
                functions=functions,
                classes=classes,
                imports=imports,
                calls_made=calls_made,
                calls_received=[],
                lines_of_code=len(content.splitlines()),
                file_type="java"
            )
        except Exception as e:
            logger.error(f"Error analyzing Java file {file_path}: {e}")
            return FileInfo(str(file_path), file_path.name, [], [], [], [], [], 0, "java")
    
    def analyze_kotlin_file(self, file_path: Path) -> FileInfo:
        """Analyze a Kotlin source file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract functions
            function_pattern = r'(?:fun|suspend\s+fun)\s+(\w+)\s*\([^)]*\)'
            functions = [match.group(1) for match in re.finditer(function_pattern, content)]
            
            # Extract classes
            class_pattern = r'(?:class|interface|object)\s+(\w+)'
            classes = [match.group(1) for match in re.finditer(class_pattern, content)]
            
            # Extract imports
            import_pattern = r'import\s+([\w.]+)'
            imports = [match.group(1) for match in re.finditer(import_pattern, content)]
            
            # Extract function calls
            calls_made = self._extract_kotlin_function_calls(content, str(file_path))
            
            return FileInfo(
                path=str(file_path),
                name=file_path.name,
                functions=functions,
                classes=classes,
                imports=imports,
                calls_made=calls_made,
                calls_received=[],
                lines_of_code=len(content.splitlines()),
                file_type="kotlin"
            )
        except Exception as e:
            logger.error(f"Error analyzing Kotlin file {file_path}: {e}")
            return FileInfo(str(file_path), file_path.name, [], [], [], [], [], 0, "kotlin")
    
    def _extract_java_function_calls(self, content: str, file_path: str) -> List[FunctionCall]:
        """Extract function calls from Java content"""
        calls = []
        lines = content.splitlines()
        
        # Simple pattern for method calls
        call_pattern = r'(\w+)\.(\w+)\s*\('
        
        for i, line in enumerate(lines):
            matches = re.finditer(call_pattern, line)
            for match in matches:
                object_name = match.group(1)
                method_name = match.group(2)
                
                calls.append(FunctionCall(
                    caller=file_path,
                    callee=f"{object_name}.{method_name}",
                    function_name=method_name,
                    line_number=i + 1,
                    context=line.strip()
                ))
        
        return calls
    
    def _extract_kotlin_function_calls(self, content: str, file_path: str) -> List[FunctionCall]:
        """Extract function calls from Kotlin content"""
        calls = []
        lines = content.splitlines()
        
        # Pattern for function calls in Kotlin
        call_pattern = r'(\w+)\.(\w+)\s*\('
        
        for i, line in enumerate(lines):
            matches = re.finditer(call_pattern, line)
            for match in matches:
                object_name = match.group(1)
                method_name = match.group(2)
                
                calls.append(FunctionCall(
                    caller=file_path,
                    callee=f"{object_name}.{method_name}",
                    function_name=method_name,
                    line_number=i + 1,
                    context=line.strip()
                ))
        
        return calls
    
    def analyze_project(self) -> bool:
        """Analyze the entire project structure"""
        if not self.clone_repository():
            return False
        
        logger.info("Analyzing project structure...")
        
        # Find all source files
        java_files = list(self.local_path.rglob("*.java"))
        kotlin_files = list(self.local_path.rglob("*.kt"))
        
        logger.info(f"Found {len(java_files)} Java files and {len(kotlin_files)} Kotlin files")
        
        # For very large projects, we might want to limit analysis to core files
        # or process in batches to avoid memory issues
        total_files = len(java_files) + len(kotlin_files)
        
        if total_files > 1000:
            logger.warning(f"Large project detected ({total_files} files). Consider filtering to core directories.")
            # You can add filtering logic here, e.g.:
            # java_files = [f for f in java_files if 'main' in str(f) or 'core' in str(f)]
            # kotlin_files = [f for f in kotlin_files if 'main' in str(f) or 'core' in str(f)]
        
        # Analyze each file with progress tracking
        processed = 0
        batch_size = 100
        
        # Process Java files
        for i in range(0, len(java_files), batch_size):
            batch = java_files[i:i+batch_size]
            for java_file in batch:
                try:
                    file_info = self.analyze_java_file(java_file)
                    self.project.files[str(java_file)] = file_info
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to analyze {java_file}: {e}")
                
            if processed % batch_size == 0:
                logger.info(f"Processed {processed}/{total_files} files...")
        
        # Process Kotlin files
        for i in range(0, len(kotlin_files), batch_size):
            batch = kotlin_files[i:i+batch_size]
            for kotlin_file in batch:
                try:
                    file_info = self.analyze_kotlin_file(kotlin_file)
                    self.project.files[str(kotlin_file)] = file_info
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to analyze {kotlin_file}: {e}")
                
            if processed % batch_size == 0:
                logger.info(f"Processed {processed}/{total_files} files...")
        
        logger.info(f"Analysis complete. Processed {processed} files successfully.")
        
        # Build call graph
        logger.info("Building call graph...")
        self._build_call_graph()
        
        return True
    
    def _build_call_graph(self):
        """Build the call graph from analyzed files"""
        for file_path, file_info in self.project.files.items():
            self.project.call_graph[file_path] = file_info.calls_made
            
            # Update calls_received for target files
            for call in file_info.calls_made:
                target_files = [f for f in self.project.files.keys() 
                              if call.callee in str(f) or any(call.function_name in func 
                                  for func in self.project.files[f].functions)]
                for target_file in target_files:
                    self.project.files[target_file].calls_received.append(call.function_name)
    
    def create_file_interaction_map(self) -> go.Figure:
        """Create Map 1: File interactions with inputs/outputs"""
        G = nx.DiGraph()
        
        # Add nodes for each file
        for file_path, file_info in self.project.files.items():
            node_name = file_info.name
            G.add_node(node_name, 
                      full_path=file_path,
                      functions=len(file_info.functions),
                      classes=len(file_info.classes),
                      loc=file_info.lines_of_code)
        
        # Add edges for function calls between files
        for file_path, calls in self.project.call_graph.items():
            source_name = Path(file_path).name
            for call in calls:
                # Find target file
                for target_path, target_info in self.project.files.items():
                    if call.function_name in target_info.functions:
                        target_name = target_info.name
                        G.add_edge(source_name, target_name, 
                                 function=call.function_name,
                                 line=call.line_number)
        
        # Create interactive plot using Plotly
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create edge traces
        edge_x, edge_y, edge_info = [], [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge[0]} → {edge[1]}<br>Function: {edge[2].get('function', 'Unknown')}")
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                               line=dict(width=2, color='rgba(50,50,50,0.5)'),
                               hoverinfo='none')
        
        # Create node traces
        node_x, node_y, node_text, node_info = [], [], [], []
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[0])
            
            info = f"File: {node[0]}<br>"
            info += f"Functions: {node[1].get('functions', 0)}<br>"
            info += f"Classes: {node[1].get('classes', 0)}<br>"
            info += f"Lines of Code: {node[1].get('loc', 0)}"
            node_info.append(info)
        
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                               marker=dict(size=20, color='lightblue',
                                         line=dict(width=2, color='darkblue')),
                               text=node_text, textposition="middle center",
                               hovertext=node_info, hoverinfo='text')
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title='AAPS EatingNow - File Interaction Map',
                                       showlegend=False,
                                       hovermode='closest',
                                       margin=dict(b=20,l=5,r=5,t=40),
                                       annotations=[ dict(
                                           text="File interactions and function calls",
                                           showarrow=False,
                                           xref="paper", yref="paper",
                                           x=0.005, y=-0.002,
                                           xanchor='left', yanchor='bottom',
                                           font=dict(color='black', size=12)
                                       )],
                                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        return fig
    
    def create_file_internal_maps(self) -> Dict[str, go.Figure]:
        """Create Map 2: Internal structure for each file"""
        file_maps = {}
        
        for file_path, file_info in self.project.files.items():
            if not file_info.functions and not file_info.classes:
                continue
                
            G = nx.DiGraph()
            
            # Add function nodes
            for func in file_info.functions:
                G.add_node(f"func_{func}", type='function', name=func)
            
            # Add class nodes
            for cls in file_info.classes:
                G.add_node(f"class_{cls}", type='class', name=cls)
            
            # Add edges for internal calls (simplified approach)
            for i, func1 in enumerate(file_info.functions):
                for j, func2 in enumerate(file_info.functions):
                    if i != j and any(call.function_name == func2 for call in file_info.calls_made):
                        G.add_edge(f"func_{func1}", f"func_{func2}")
            
            if G.nodes():
                try:
                    pos = nx.spring_layout(G, k=1, iterations=50)
                except:
                    # Fallback to circular layout if spring layout fails
                    pos = nx.circular_layout(G)
                
                # Create traces
                edge_x, edge_y = [], []
                for edge in G.edges():
                    if edge[0] in pos and edge[1] in pos:
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                       line=dict(width=1, color='gray'),
                                       hoverinfo='none')
                
                node_x, node_y, node_text, node_colors = [], [], [], []
                for node_id in G.nodes():
                    if node_id in pos:
                        x, y = pos[node_id]
                        node_x.append(x)
                        node_y.append(y)
                        
                        # Extract the actual name from node_id
                        if node_id.startswith('func_'):
                            name = node_id[5:]  # Remove 'func_' prefix
                            color = 'lightcoral'
                        elif node_id.startswith('class_'):
                            name = node_id[6:]  # Remove 'class_' prefix
                            color = 'lightgreen'
                        else:
                            name = node_id
                            color = 'lightblue'
                        
                        node_text.append(name)
                        node_colors.append(color)
                
                node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                       marker=dict(size=15, color=node_colors,
                                                 line=dict(width=1, color='black')),
                                       text=node_text, textposition="middle center",
                                       hoverinfo='text')
                
                fig = go.Figure(data=[edge_trace, node_trace],
                               layout=go.Layout(
                                   title=f'Internal Structure - {file_info.name}',
                                   showlegend=False,
                                   hovermode='closest',
                                   margin=dict(b=20,l=5,r=5,t=40),
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                               ))
                
                file_maps[file_info.name] = fig
        
        return file_maps
    
    def create_data_flow_map(self) -> go.Figure:
        """Create Map 3: Data and logic flow through the app"""
        # This is a high-level conceptual map based on AAPS architecture
        # You would customize this based on actual analysis
        
        flow_steps = [
            "CGM Data Input",
            "Blood Glucose Processing",
            "IOB Calculation", 
            "COB Calculation",
            "Algorithm Decision",
            "Basal/Bolus Adjustment",
            "Pump Communication",
            "Loop Execution",
            "Data Logging",
            "UI Update"
        ]
        
        G = nx.DiGraph()
        
        # Add sequential flow
        for i in range(len(flow_steps) - 1):
            G.add_edge(flow_steps[i], flow_steps[i + 1])
        
        # Add some feedback loops
        G.add_edge("Data Logging", "Blood Glucose Processing")
        G.add_edge("Loop Execution", "CGM Data Input")
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                               line=dict(width=3, color='blue'))
        
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                               marker=dict(size=30, color='orange',
                                         line=dict(width=2, color='darkorange')),
                               text=node_text, textposition="middle center")
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title='AAPS EatingNow - Data Flow Map',
                                       showlegend=False))
        
        return fig

class Neo4jRAGDatabase:
    """Neo4j database for storing project knowledge"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def populate_from_project(self, project: ProjectStructure):
        """Populate Neo4j with project structure"""
        with self.driver.session() as session:
            # Create file nodes
            for file_path, file_info in project.files.items():
                session.run("""
                    CREATE (f:File {
                        name: $name,
                        path: $path,
                        file_type: $file_type,
                        lines_of_code: $loc,
                        function_count: $func_count,
                        class_count: $class_count
                    })
                """, name=file_info.name, path=file_info.path, 
                    file_type=file_info.file_type, loc=file_info.lines_of_code,
                    func_count=len(file_info.functions), class_count=len(file_info.classes))
                
                # Create function nodes
                for func_name in file_info.functions:
                    session.run("""
                        MATCH (f:File {path: $file_path})
                        CREATE (fn:Function {name: $func_name})
                        CREATE (f)-[:CONTAINS]->(fn)
                    """, file_path=file_info.path, func_name=func_name)
                
                # Create class nodes
                for class_name in file_info.classes:
                    session.run("""
                        MATCH (f:File {path: $file_path})
                        CREATE (c:Class {name: $class_name})
                        CREATE (f)-[:CONTAINS]->(c)
                    """, file_path=file_info.path, class_name=class_name)
            
            # Create call relationships
            for file_path, calls in project.call_graph.items():
                for call in calls:
                    session.run("""
                        MATCH (f1:File {path: $caller_path})
                        MATCH (f2:File) WHERE ANY(func IN f2.functions WHERE func = $callee_func)
                        CREATE (f1)-[:CALLS {function: $function_name, line: $line_num, context: $context}]->(f2)
                    """, caller_path=call.caller, callee_func=call.function_name,
                        function_name=call.function_name, line_num=call.line_number, context=call.context)
    
    def query_project_info(self, query: str) -> List[Dict]:
        """Query project information using Cypher"""
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

def main():
    """Main execution function"""
    # Configuration
    REPO_URL = "https://github.com/dicko72/AAPS-EatingNow.git"
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this to your Neo4j password
    
    # Initialize analyzer
    analyzer = AAPSProjectAnalyzer(REPO_URL)
    
    # Analyze project
    logger.info("Starting project analysis...")
    if not analyzer.analyze_project():
        logger.error("Failed to analyze project")
        return
    
    # Create mind maps (limit internal maps for large projects)
    logger.info("Creating mind maps...")
    
    # Map 1: File interactions
    try:
        file_map = analyzer.create_file_interaction_map()
        pyo.plot(file_map, filename='aaps_file_interactions.html', auto_open=False)
        logger.info("Created file interaction map: aaps_file_interactions.html")
    except Exception as e:
        logger.error(f"Failed to create file interaction map: {e}")
    
    # Map 2: Internal file structures (limit to manageable number)
    try:
        internal_maps = analyzer.create_file_internal_maps()
        
        # If too many files, only create maps for the largest ones
        if len(internal_maps) > 50:
            logger.info(f"Too many files ({len(internal_maps)}), creating maps for top 50 largest files only")
            # Sort by complexity and take top 50
            sorted_files = sorted(analyzer.project.files.items(), 
                                key=lambda x: len(x[1].functions) + len(x[1].classes), 
                                reverse=True)[:50]
            internal_maps = {info.name: internal_maps[info.name] 
                           for _, info in sorted_files 
                           if info.name in internal_maps}
        
        for file_name, fig in internal_maps.items():
            try:
                filename = f"aaps_internal_{file_name.replace('.', '_').replace('/', '_')}.html"
                pyo.plot(fig, filename=filename, auto_open=False)
            except Exception as e:
                logger.error(f"Failed to create internal map for {file_name}: {e}")
                
        logger.info(f"Created {len(internal_maps)} internal structure maps")
    except Exception as e:
        logger.error(f"Failed to create internal maps: {e}")
    
    # Map 3: Data flow
    try:
        flow_map = analyzer.create_data_flow_map()
        pyo.plot(flow_map, filename='aaps_data_flow.html', auto_open=False)
        logger.info("Created data flow map: aaps_data_flow.html")
    except Exception as e:
        logger.error(f"Failed to create data flow map: {e}")
    
    # Populate Neo4j database
    try:
        logger.info("Populating Neo4j database...")
        db = Neo4jRAGDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        db.clear_database()
        db.populate_from_project(analyzer.project)
        
        # Example queries
        logger.info("Running sample queries...")
        
        # Find files with most functions
        result = db.query_project_info("""
            MATCH (f:File)
            RETURN f.name as filename, f.function_count as functions
            ORDER BY f.function_count DESC
            LIMIT 10
        """)
        
        print("\nFiles with most functions:")
        for record in result:
            print(f"  {record['filename']}: {record['functions']} functions")
        
        # Find most called functions
        result = db.query_project_info("""
            MATCH (f1:File)-[c:CALLS]->(f2:File)
            RETURN c.function as function_name, count(*) as call_count
            ORDER BY call_count DESC
            LIMIT 10
        """)
        
        print("\nMost called functions:")
        for record in result:
            print(f"  {record['function_name']}: {record['call_count']} calls")
        
        db.close()
        logger.info("Neo4j database populated successfully")
        
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        logger.info("Make sure Neo4j is running and credentials are correct")
        logger.info("You can skip Neo4j and still use the HTML visualizations")
    
    # Save project structure as JSON for further analysis
    try:
        # Limit the JSON output to avoid huge files
        limited_files = {}
        for path, info in list(analyzer.project.files.items())[:500]:  # Limit to first 500 files
            limited_files[path] = asdict(info)
        
        project_data = {
            'files': limited_files,
            'summary': {
                'total_files': len(analyzer.project.files),
                'total_functions': sum(len(info.functions) for info in analyzer.project.files.values()),
                'total_classes': sum(len(info.classes) for info in analyzer.project.files.values()),
                'total_loc': sum(info.lines_of_code for info in analyzer.project.files.values()),
                'files_in_export': len(limited_files)
            }
        }
        
        with open('aaps_project_analysis.json', 'w') as f:
            json.dump(project_data, f, indent=2, default=str)
        
        logger.info("Analysis complete! Generated files:")
        logger.info("  - aaps_file_interactions.html (Map 1)")
        logger.info(f"  - aaps_internal_*.html (Map 2 - {len(internal_maps) if 'internal_maps' in locals() else 0} files)")
        logger.info("  - aaps_data_flow.html (Map 3)")
        logger.info("  - aaps_project_analysis.json (Raw data)")
        if 'db' in locals():
            logger.info("  - Neo4j database populated (if connected)")
        
        # Print summary statistics
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Total files analyzed: {project_data['summary']['total_files']}")
        print(f"Total functions: {project_data['summary']['total_functions']}")
        print(f"Total classes: {project_data['summary']['total_classes']}")
        print(f"Total lines of code: {project_data['summary']['total_loc']}")
        
    except Exception as e:
        logger.error(f"Failed to save project analysis: {e}")

if __name__ == "__main__":
    main()
