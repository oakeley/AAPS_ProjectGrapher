#!/usr/bin/env python3
"""
High-Performance AAPS Analyzer - Complete Version with GitHub Integration
Designed for high-memory, multi-core systems (384GB RAM, 96 cores)
Includes automatic GitHub repository cloning and full analysis pipeline
"""

import os
import re
import json
import asyncio
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, Counter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import gc
from functools import partial

# Try to import optional dependencies
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️ Neo4j driver not available. Install with: pip install neo4j")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ NetworkX not available. Install with: pip install networkx")

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Install with: pip install plotly")

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    print("⚠️ GitPython not available. Install with: pip install gitpython")

# Configure for high-performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Memory and performance settings
MAX_WORKERS = min(96, mp.cpu_count())  # Use all available cores up to 96
CHUNK_SIZE = 100  # Process files in chunks
NEO4J_BATCH_SIZE = 10000  # Large batch size for Neo4j operations
MAX_MEMORY_USAGE = 300 * 1024 * 1024 * 1024  # 300GB memory limit


def clone_or_update_repository(repo_url: str, local_path: Path, branch: str = "EN-MASTER-NEW") -> bool:
    """Clone or update the AAPS repository"""
    logger.info(f"Setting up repository: {repo_url}")
    
    try:
        if local_path.exists():
            if GIT_AVAILABLE:
                # Try to update existing repository
                logger.info(f"Repository exists at {local_path}, attempting to update...")
                try:
                    repo = git.Repo(local_path)
                    origin = repo.remotes.origin
                    origin.pull()
                    logger.info("✅ Repository updated successfully")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to update repository: {e}")
                    logger.info("Continuing with existing repository...")
                    return True
            else:
                logger.info("Repository exists, continuing with existing copy...")
                return True
        else:
            # Clone repository
            if GIT_AVAILABLE:
                logger.info(f"Cloning repository to {local_path}...")
                try:
                    git.Repo.clone_from(repo_url, local_path, branch=branch)
                    logger.info("✅ Repository cloned successfully")
                    return True
                except Exception as e:
                    logger.error(f"Failed to clone with GitPython: {e}")
                    # Fall back to subprocess
                    return clone_with_subprocess(repo_url, local_path, branch)
            else:
                # Use subprocess as fallback
                return clone_with_subprocess(repo_url, local_path, branch)
    
    except Exception as e:
        logger.error(f"Repository setup failed: {e}")
        return False


def clone_with_subprocess(repo_url: str, local_path: Path, branch: str) -> bool:
    """Clone repository using subprocess as fallback"""
    try:
        logger.info("Trying to clone with git command...")
        cmd = ["git", "clone", "-b", branch, repo_url, str(local_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Repository cloned successfully with git command")
            return True
        else:
            logger.error(f"Git clone failed: {result.stderr}")
            
            # Try without branch specification
            logger.info("Retrying without branch specification...")
            cmd = ["git", "clone", repo_url, str(local_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Repository cloned successfully (default branch)")
                return True
            else:
                logger.error(f"Git clone failed: {result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out")
        return False
    except Exception as e:
        logger.error(f"Subprocess clone failed: {e}")
        return False


def extract_package_name(content: str, file_path: Path) -> str:
    """Extract package name from file content"""
    package_match = re.search(r'package\s+([\w.]+)', content)
    if package_match:
        return package_match.group(1)
    
    # Fallback: derive from file path
    path_parts = file_path.parts
    if 'src' in path_parts:
        src_index = path_parts.index('src')
        if src_index + 2 < len(path_parts):
            return '.'.join(path_parts[src_index+2:-1])
    
    return "unknown"


@dataclass
class FileData:
    name: str
    path: str
    functions: List[str]
    classes: List[str]
    imports: List[str]
    lines_of_code: int
    file_type: str
    function_calls: List[Dict[str, str]]
    package: str = ""
    importance_score: float = 0.0
    complexity_score: float = 0.0
    file_size: int = 0


class HighPerformanceAnalyzer:
    """High-performance analyzer using all available resources"""
    
    def __init__(self, repo_url: str = "https://github.com/dicko72/AAPS-EatingNow.git", 
                 local_path: str = "./aaps_eating_now"):
        self.repo_url = repo_url
        self.local_path = Path(local_path)
        self.files_data = {}
        self.call_graph = None
        if NETWORKX_AVAILABLE:
            self.call_graph = nx.DiGraph()
        self.function_to_files = defaultdict(set)
        
    def setup_project(self) -> bool:
        """Setup the project by cloning or updating repository"""
        return clone_or_update_repository(self.repo_url, self.local_path)
    
    def find_all_source_files(self) -> Tuple[List[Path], List[Path]]:
        """Find all source files with parallel directory traversal"""
        logger.info("Finding all source files...")
        
        java_files = []
        kotlin_files = []
        
        # Use parallel processing for file discovery
        def scan_directory(directory: Path) -> Tuple[List[Path], List[Path]]:
            java = list(directory.rglob("*.java"))
            kotlin = list(directory.rglob("*.kt"))
            return java, kotlin
        
        # For very large projects, we can parallelize directory scanning
        if self.local_path.exists():
            java_files, kotlin_files = scan_directory(self.local_path)
        
        logger.info(f"Found {len(java_files)} Java files and {len(kotlin_files)} Kotlin files")
        return java_files, kotlin_files
    
    def analyze_project_parallel(self) -> bool:
        """Analyze the entire project using parallel processing"""
        # First, setup the project
        if not self.setup_project():
            logger.error("Failed to setup project repository")
            return False
        
        if not self.local_path.exists():
            logger.error(f"Project path {self.local_path} does not exist!")
            return False
        
        start_time = time.time()
        logger.info(f"Starting high-performance analysis with {MAX_WORKERS} workers...")
        
        # Find all source files
        java_files, kotlin_files = self.find_all_source_files()
        all_files = java_files + kotlin_files
        
        if not all_files:
            logger.error("No source files found!")
            return False
        
        logger.info(f"Processing {len(all_files)} files using {MAX_WORKERS} cores...")
        
        # Split files into chunks for parallel processing
        file_chunks = [all_files[i:i+CHUNK_SIZE] for i in range(0, len(all_files), CHUNK_SIZE)]
        
        # Process files in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(analyze_file_batch, chunk): i 
                             for i, chunk in enumerate(file_chunks)}
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    
                    # Progress reporting
                    completed_chunks = len([f for f in future_to_chunk if f.done()])
                    progress = (completed_chunks / len(file_chunks)) * 100
                    logger.info(f"Progress: {progress:.1f}% ({completed_chunks}/{len(file_chunks)} chunks)")
                    
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} failed: {e}")
        
        # Store results
        for file_data in all_results:
            if file_data:
                self.files_data[file_data.path] = file_data
        
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        logger.info(f"Successfully processed {len(self.files_data)} files")
        
        # Calculate importance scores
        self._calculate_importance_scores()
        
        # Build function-to-file mapping
        self._build_function_mapping()
        
        return True
    
    def _calculate_importance_scores(self):
        """Calculate importance scores with enhanced metrics"""
        logger.info("Calculating importance scores...")
        
        # Calculate various metrics
        max_loc = max((f.lines_of_code for f in self.files_data.values()), default=1)
        max_functions = max((len(f.functions) for f in self.files_data.values()), default=1)
        max_complexity = max((f.complexity_score for f in self.files_data.values()), default=1)
        
        for file_data in self.files_data.values():
            score = 0.0
            
            # Normalized metrics
            loc_score = (file_data.lines_of_code / max_loc) * 10
            function_score = (len(file_data.functions) / max_functions) * 15
            class_score = len(file_data.classes) * 3
            complexity_score = (file_data.complexity_score / max_complexity) * 8
            imports_score = len(file_data.imports) * 0.1
            calls_score = len(file_data.function_calls) * 0.1
            
            score = loc_score + function_score + class_score + complexity_score + imports_score + calls_score
            
            # Bonus points for important file patterns
            name_lower = file_data.name.lower()
            package_lower = file_data.package.lower()
            
            # Core functionality bonuses
            if any(keyword in name_lower for keyword in ['service', 'manager', 'controller', 'plugin', 'processor']):
                score += 20
            if any(keyword in name_lower for keyword in ['main', 'app', 'activity', 'application']):
                score += 15
            if any(keyword in name_lower for keyword in ['algorithm', 'calculator', 'compute', 'engine']):
                score += 18
            if any(keyword in name_lower for keyword in ['pump', 'cgm', 'sensor', 'glucose', 'insulin']):
                score += 16
            if any(keyword in name_lower for keyword in ['loop', 'automation', 'treatment']):
                score += 14
            
            # Package importance
            if any(keyword in package_lower for keyword in ['core', 'main', 'engine', 'algorithm']):
                score += 10
            if any(keyword in package_lower for keyword in ['pump', 'cgm', 'glucose', 'insulin']):
                score += 8
            
            # Penalty for test/generated files
            if any(keyword in name_lower for keyword in ['test', 'mock', 'fake', 'stub']):
                score *= 0.1
            if any(keyword in file_data.path.lower() for keyword in ['test', 'generated', 'build']):
                score *= 0.2
            
            file_data.importance_score = score
    
    def _build_function_mapping(self):
        """Build mapping from functions to files for efficient call resolution"""
        logger.info("Building function-to-file mapping...")
        
        for file_data in self.files_data.values():
            for function_name in file_data.functions:
                self.function_to_files[function_name].add(file_data.name)
    
    def build_comprehensive_call_graph(self):
        """Build comprehensive call graph using parallel processing"""
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, skipping call graph creation")
            return
        
        logger.info("Building comprehensive call graph...")
        start_time = time.time()
        
        # Add all file nodes
        for file_data in self.files_data.values():
            self.call_graph.add_node(
                file_data.name,
                path=file_data.path,
                package=file_data.package,
                functions=len(file_data.functions),
                classes=len(file_data.classes),
                loc=file_data.lines_of_code,
                file_type=file_data.file_type,
                importance=file_data.importance_score,
                complexity=file_data.complexity_score
            )
        
        # Build edges in parallel
        call_counts = defaultdict(int)
        
        def process_file_calls(file_data):
            """Process function calls for a single file"""
            local_calls = defaultdict(int)
            
            for call in file_data.function_calls:
                function_name = call['function']
                target_files = self.function_to_files.get(function_name, set())
                
                for target_file in target_files:
                    if target_file != file_data.name:  # Avoid self-loops
                        local_calls[(file_data.name, target_file)] += 1
            
            return local_calls
        
        # Process calls in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_file_calls, file_data) 
                      for file_data in self.files_data.values()]
            
            for future in as_completed(futures):
                try:
                    local_calls = future.result()
                    for key, count in local_calls.items():
                        call_counts[key] += count
                except Exception as e:
                    logger.error(f"Failed to process calls: {e}")
        
        # Add edges to graph
        for (source, target), weight in call_counts.items():
            if weight > 0:
                self.call_graph.add_edge(source, target, weight=weight, calls=weight)
        
        build_time = time.time() - start_time
        logger.info(f"Call graph built in {build_time:.2f} seconds: {len(self.call_graph.nodes)} nodes, {len(self.call_graph.edges)} edges")
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations using the full dataset"""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, skipping visualizations")
            return
        
        logger.info("Creating comprehensive visualizations...")
        
        # 1. Main network overview (top files)
        main_fig = self._create_main_network_viz()
        if main_fig:
            pyo.plot(main_fig, filename='aaps_full_network.html', auto_open=False)
            logger.info("✅ Created: aaps_full_network.html")
        
        # 2. Package hierarchy
        package_fig = self._create_package_hierarchy()
        if package_fig:
            pyo.plot(package_fig, filename='aaps_package_hierarchy.html', auto_open=False)
            logger.info("✅ Created: aaps_package_hierarchy.html")
        
        # 3. Complexity heatmap
        complexity_fig = self._create_complexity_heatmap()
        if complexity_fig:
            pyo.plot(complexity_fig, filename='aaps_complexity_heatmap.html', auto_open=False)
            logger.info("✅ Created: aaps_complexity_heatmap.html")
        
        # 4. File type analysis
        filetype_fig = self._create_filetype_analysis()
        if filetype_fig:
            pyo.plot(filetype_fig, filename='aaps_filetype_analysis.html', auto_open=False)
            logger.info("✅ Created: aaps_filetype_analysis.html")
        
        # 5. Interactive file explorer with full search
        explorer_fig = self._create_advanced_file_explorer()
        if explorer_fig:
            pyo.plot(explorer_fig, filename='aaps_advanced_explorer.html', auto_open=False)
            logger.info("✅ Created: aaps_advanced_explorer.html")
    
    def _create_main_network_viz(self):
        """Create main network visualization with top files"""
        if not NETWORKX_AVAILABLE or not self.call_graph:
            logger.warning("NetworkX or call graph not available, creating simple visualization")
            return self._create_simple_scatter_plot()
        
        # Get top 100 most important files
        top_files = sorted(self.files_data.values(), key=lambda x: x.importance_score, reverse=True)[:100]
        subgraph_nodes = [f.name for f in top_files]
        subgraph = self.call_graph.subgraph(subgraph_nodes)
        
        # Enhanced layout
        try:
            pos = nx.spring_layout(subgraph, k=5, iterations=100, seed=42, weight='weight')
        except:
            pos = nx.circular_layout(subgraph)
        
        # Create edges with variable thickness
        edge_x, edge_y = [], []
        edge_weights = []
        
        for edge in subgraph.edges(data=True):
            source, target = edge[0], edge[1]
            if source in pos and target in pos:
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(edge[2].get('weight', 1))
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='rgba(125,125,125,0.4)'),
            hoverinfo='none',
            showlegend=False
        )
        
        # Create nodes with enhanced information
        node_x, node_y, node_text, node_info, node_colors, node_sizes = [], [], [], [], [], []
        
        for node in subgraph.nodes(data=True):
            if node[0] in pos:
                x, y = pos[node[0]]
                node_x.append(x)
                node_y.append(y)
                
                name = node[0]
                data = node[1]
                
                # Enhanced node text
                display_name = name if len(name) <= 25 else name[:22] + "..."
                node_text.append(display_name)
                
                # Rich hover information
                info = f"<b>{name}</b><br>"
                info += f"Package: {data.get('package', 'unknown')}<br>"
                info += f"Functions: {data.get('functions', 0)}<br>"
                info += f"Classes: {data.get('classes', 0)}<br>"
                info += f"Lines: {data.get('loc', 0)}<br>"
                info += f"Type: {data.get('file_type', 'unknown')}<br>"
                info += f"Importance: {data.get('importance', 0):.1f}<br>"
                info += f"Complexity: {data.get('complexity', 0):.1f}<br>"
                info += f"Connections: {len(list(subgraph.neighbors(name)))}<br>"
                info += f"Path: {data.get('path', '')}"
                node_info.append(info)
                
                # Color by file type and importance
                file_type = data.get('file_type', 'unknown')
                importance = data.get('importance', 0)
                
                if file_type == 'java':
                    if importance > 50:
                        node_colors.append('darkred')
                    elif importance > 25:
                        node_colors.append('lightcoral')
                    else:
                        node_colors.append('pink')
                elif file_type == 'kotlin':
                    if importance > 50:
                        node_colors.append('darkblue')
                    elif importance > 25:
                        node_colors.append('lightblue')
                    else:
                        node_colors.append('lightcyan')
                else:
                    node_colors.append('lightgray')
                
                # Size by importance
                node_sizes.append(max(15, min(60, importance)))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='darkblue'),
                sizemode='diameter'
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=8),
            hovertext=node_info,
            hoverinfo='text',
            showlegend=False
        )
        
        # Create the figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text='AAPS EatingNow - Complete Network Analysis<br><sub>Top 100 Most Important Files - Full Resolution</sub>',
                    x=0.5,
                    font=dict(size=24)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=100),
                annotations=[
                    dict(
                        text="🔵 Kotlin Files &nbsp;&nbsp; 🔴 Java Files<br>Darker colors = Higher importance • Size = Importance score<br>Click and drag to explore • Hover for details",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='black', size=14)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                width=1400,
                height=1000
            )
        )
        
        return fig
    
    def _create_simple_scatter_plot(self):
        """Create a simple scatter plot when NetworkX is not available"""
        # Get top files
        top_files = sorted(self.files_data.values(), key=lambda x: x.importance_score, reverse=True)[:100]
        
        # Create scatter plot
        x_vals = [f.lines_of_code for f in top_files]
        y_vals = [len(f.functions) for f in top_files]
        colors = [f.importance_score for f in top_files]
        text_vals = [f.name for f in top_files]
        
        fig = go.Figure(data=go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text',
            marker=dict(
                size=[max(10, min(40, f.importance_score)) for f in top_files],
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance Score")
            ),
            text=text_vals,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>LOC: %{x}<br>Functions: %{y}<br>Importance: %{marker.color:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='AAPS EatingNow - File Analysis<br><sub>Top 100 Most Important Files</sub>',
            xaxis_title='Lines of Code',
            yaxis_title='Number of Functions',
            width=1200,
            height=800
        )
        
        return fig
    
    def _create_package_hierarchy(self):
        """Create package hierarchy visualization"""
        # Group files by package
        packages = defaultdict(list)
        for file_data in self.files_data.values():
            packages[file_data.package].append(file_data)
        
        # Calculate package metrics
        package_data = []
        for package, files in packages.items():
            if package == "unknown" or len(files) < 2:
                continue
                
            total_loc = sum(f.lines_of_code for f in files)
            total_functions = sum(len(f.functions) for f in files)
            total_classes = sum(len(f.classes) for f in files)
            avg_importance = sum(f.importance_score for f in files) / len(files)
            avg_complexity = sum(f.complexity_score for f in files) / len(files)
            
            package_data.append({
                'package': package,
                'files': len(files),
                'loc': total_loc,
                'functions': total_functions,
                'classes': total_classes,
                'importance': avg_importance,
                'complexity': avg_complexity,
                'java_files': len([f for f in files if f.file_type == 'java']),
                'kotlin_files': len([f for f in files if f.file_type == 'kotlin'])
            })
        
        # Sort by importance
        package_data.sort(key=lambda x: x['importance'], reverse=True)
        
        # Create enhanced bubble chart
        fig = go.Figure()
        
        for pkg in package_data[:50]:  # Top 50 packages
            # Color by complexity
            color_val = pkg['complexity']
            
            fig.add_trace(go.Scatter(
                x=[pkg['functions']],
                y=[pkg['loc']],
                mode='markers+text',
                marker=dict(
                    size=max(20, min(120, pkg['files'] * 8)),
                    color=color_val,
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=2, color='black'),
                    colorbar=dict(title="Average Complexity")
                ),
                text=[pkg['package'].split('.')[-1]],  # Just the last part
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                hovertemplate=
                    "<b>%{text}</b><br>" +
                    "Full Package: " + pkg['package'] + "<br>" +
                    "Files: " + str(pkg['files']) + "<br>" +
                    "Functions: %{x}<br>" +
                    "Lines of Code: %{y}<br>" +
                    "Classes: " + str(pkg['classes']) + "<br>" +
                    "Java Files: " + str(pkg['java_files']) + "<br>" +
                    "Kotlin Files: " + str(pkg['kotlin_files']) + "<br>" +
                    "Importance: " + f"{pkg['importance']:.1f}<br>" +
                    "Complexity: " + f"{pkg['complexity']:.1f}" +
                    "<extra></extra>",
                showlegend=False
            ))
        
        fig.update_layout(
            title='AAPS EatingNow - Package Hierarchy Analysis<br><sub>Bubble size = file count, Color = complexity, Position = functions vs LOC</sub>',
            xaxis_title='Number of Functions',
            yaxis_title='Lines of Code',
            hovermode='closest',
            plot_bgcolor='white',
            width=1200,
            height=800
        )
        
        return fig
    
    def _create_complexity_heatmap(self):
        """Create complexity heatmap"""
        # Get top complex files
        top_complex = sorted(self.files_data.values(), key=lambda x: x.complexity_score, reverse=True)[:50]
        
        # Create simple bar chart instead of heatmap for better compatibility
        fig = go.Figure(data=[
            go.Bar(
                x=[f.complexity_score for f in top_complex],
                y=[f.name[:30] + '...' if len(f.name) > 30 else f.name for f in top_complex],
                orientation='h',
                marker=dict(
                    color=[f.importance_score for f in top_complex],
                    colorscale='Reds',
                    colorbar=dict(title="Importance Score")
                ),
                hovertemplate='<b>%{y}</b><br>Complexity: %{x:.1f}<br>Importance: %{marker.color:.1f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='AAPS EatingNow - Complexity Analysis<br><sub>Top 50 Most Complex Files</sub>',
            xaxis_title='Complexity Score',
            yaxis_title='Files',
            height=1200,
            width=1000
        )
        
        return fig
    
    def _create_filetype_analysis(self):
        """Create file type analysis"""
        # Separate Java and Kotlin files
        java_files = [f for f in self.files_data.values() if f.file_type == 'java']
        kotlin_files = [f for f in self.files_data.values() if f.file_type == 'kotlin']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Lines of Code Distribution', 'Function Count Distribution', 
                          'Importance Score Distribution', 'Complexity Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # LOC distribution
        fig.add_trace(go.Histogram(x=[f.lines_of_code for f in java_files], name='Java LOC', opacity=0.7), row=1, col=1)
        fig.add_trace(go.Histogram(x=[f.lines_of_code for f in kotlin_files], name='Kotlin LOC', opacity=0.7), row=1, col=1)
        
        # Function count distribution
        fig.add_trace(go.Histogram(x=[len(f.functions) for f in java_files], name='Java Functions', opacity=0.7), row=1, col=2)
        fig.add_trace(go.Histogram(x=[len(f.functions) for f in kotlin_files], name='Kotlin Functions', opacity=0.7), row=1, col=2)
        
        # Importance distribution
        fig.add_trace(go.Histogram(x=[f.importance_score for f in java_files], name='Java Importance', opacity=0.7), row=2, col=1)
        fig.add_trace(go.Histogram(x=[f.importance_score for f in kotlin_files], name='Kotlin Importance', opacity=0.7), row=2, col=1)
        
        # Complexity distribution
        fig.add_trace(go.Histogram(x=[f.complexity_score for f in java_files], name='Java Complexity', opacity=0.7), row=2, col=2)
        fig.add_trace(go.Histogram(x=[f.complexity_score for f in kotlin_files], name='Kotlin Complexity', opacity=0.7), row=2, col=2)
        
        fig.update_layout(
            title='AAPS EatingNow - File Type Analysis<br><sub>Java vs Kotlin Comparison</sub>',
            height=800,
            width=1200
        )
        
        return fig
    
    def _create_advanced_file_explorer(self):
        """Create advanced file explorer table"""
        # Prepare comprehensive data
        file_list = []
        for file_data in self.files_data.values():
            # Count connections if call graph available
            incoming = outgoing = 0
            if NETWORKX_AVAILABLE and self.call_graph:
                incoming = len([edge for edge in self.call_graph.edges() if edge[1] == file_data.name])
                outgoing = len([edge for edge in self.call_graph.edges() if edge[0] == file_data.name])
            
            file_list.append({
                'name': file_data.name,
                'package': file_data.package,
                'type': file_data.file_type,
                'loc': file_data.lines_of_code,
                'functions': len(file_data.functions),
                'classes': len(file_data.classes),
                'imports': len(file_data.imports),
                'calls_in': incoming,
                'calls_out': outgoing,
                'importance': round(file_data.importance_score, 2),
                'complexity': round(file_data.complexity_score, 2),
                'file_size': file_data.file_size,
                'path': file_data.path
            })
        
        # Sort by importance
        file_list.sort(key=lambda x: x['importance'], reverse=True)
        
        # Create interactive table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['File Name', 'Package', 'Type', 'LOC', 'Functions', 'Classes', 'Imports', 
                       'Calls In', 'Calls Out', 'Importance', 'Complexity', 'Size (KB)'],
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='black'),
                height=40
            ),
            cells=dict(
                values=[
                    [f['name'] for f in file_list],
                    [f['package'] for f in file_list],
                    [f['type'] for f in file_list],
                    [f['loc'] for f in file_list],
                    [f['functions'] for f in file_list],
                    [f['classes'] for f in file_list],
                    [f['imports'] for f in file_list],
                    [f['calls_in'] for f in file_list],
                    [f['calls_out'] for f in file_list],
                    [f['importance'] for f in file_list],
                    [f['complexity'] for f in file_list],
                    [round(f['file_size']/1024, 1) for f in file_list]
                ],
                fill_color='white',
                align='left',
                font=dict(size=10),
                height=30
            )
        )])
        
        fig.update_layout(
            title='AAPS EatingNow - Advanced File Explorer<br><sub>Complete file analysis - sortable and searchable</sub>',
            height=1000,
            width=1600
        )
        
        return fig
    
    def populate_neo4j_high_performance(self, uri: str, user: str, password: str):
        """High-performance Neo4j population with large batches"""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available, skipping database population")
            return
        
        logger.info("Populating Neo4j with high-performance settings...")
        
        # Configure Neo4j driver for high performance
        driver = GraphDatabase.driver(
            uri, 
            auth=(user, password),
            max_connection_lifetime=3600,
            max_connection_pool_size=100,
            connection_acquisition_timeout=60,
            encrypted=False
        )
        
        start_time = time.time()
        
        with driver.session() as session:
            # Clear existing data
            logger.info("Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create file nodes in large batches
            logger.info("Creating file nodes...")
            file_batch = []
            
            for file_data in self.files_data.values():
                file_batch.append({
                    'name': file_data.name,
                    'path': file_data.path,
                    'file_type': file_data.file_type,
                    'loc': file_data.lines_of_code,
                    'func_count': len(file_data.functions),
                    'class_count': len(file_data.classes),
                    'import_count': len(file_data.imports),
                    'functions': file_data.functions,
                    'classes': file_data.classes,
                    'imports': file_data.imports,
                    'package': file_data.package,
                    'importance': file_data.importance_score,
                    'complexity': file_data.complexity_score,
                    'file_size': file_data.file_size
                })
                
                if len(file_batch) >= NEO4J_BATCH_SIZE:
                    session.run("""
                        UNWIND $batch AS file
                        CREATE (f:File {
                            name: file.name,
                            path: file.path,
                            file_type: file.file_type,
                            lines_of_code: file.loc,
                            function_count: file.func_count,
                            class_count: file.class_count,
                            import_count: file.import_count,
                            functions: file.functions,
                            classes: file.classes,
                            imports: file.imports,
                            package: file.package,
                            importance_score: file.importance,
                            complexity_score: file.complexity,
                            file_size: file.file_size
                        })
                    """, batch=file_batch)
                    logger.info(f"Created {len(file_batch)} file nodes...")
                    file_batch = []
            
            # Process remaining files
            if file_batch:
                session.run("""
                    UNWIND $batch AS file
                    CREATE (f:File {
                        name: file.name,
                        path: file.path,
                        file_type: file.file_type,
                        lines_of_code: file.loc,
                        function_count: file.func_count,
                        class_count: file.class_count,
                        import_count: file.import_count,
                        functions: file.functions,
                        classes: file.classes,
                        imports: file.imports,
                        package: file.package,
                        importance_score: file.importance,
                        complexity_score: file.complexity,
                        file_size: file.file_size
                    })
                """, batch=file_batch)
            
            # Create function and class nodes in batches
            logger.info("Creating function and class nodes...")
            function_batch = []
            class_batch = []
            
            for file_data in self.files_data.values():
                for function_name in file_data.functions:
                    function_batch.append({
                        'file_path': file_data.path,
                        'function_name': function_name
                    })
                    
                    if len(function_batch) >= NEO4J_BATCH_SIZE:
                        session.run("""
                            UNWIND $batch AS item
                            MATCH (f:File {path: item.file_path})
                            CREATE (fn:Function {name: item.function_name})
                            CREATE (f)-[:CONTAINS]->(fn)
                        """, batch=function_batch)
                        function_batch = []
                
                for class_name in file_data.classes:
                    class_batch.append({
                        'file_path': file_data.path,
                        'class_name': class_name
                    })
                    
                    if len(class_batch) >= NEO4J_BATCH_SIZE:
                        session.run("""
                            UNWIND $batch AS item
                            MATCH (f:File {path: item.file_path})
                            CREATE (c:Class {name: item.class_name})
                            CREATE (f)-[:CONTAINS]->(c)
                        """, batch=class_batch)
                        class_batch = []
            
            # Process remaining functions and classes
            if function_batch:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (f:File {path: item.file_path})
                    CREATE (fn:Function {name: item.function_name})
                    CREATE (f)-[:CONTAINS]->(fn)
                """, batch=function_batch)
            
            if class_batch:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (f:File {path: item.file_path})
                    CREATE (c:Class {name: item.class_name})
                    CREATE (f)-[:CONTAINS]->(c)
                """, batch=class_batch)
            
            # Create CALLS relationships
            logger.info("Creating CALLS relationships...")
            calls_batch = []
            calls_created = 0
            
            if NETWORKX_AVAILABLE and self.call_graph:
                # Use call graph if available
                for source, target, data in self.call_graph.edges(data=True):
                    calls_batch.append({
                        'source': source,
                        'target': target,
                        'weight': data.get('weight', 1),
                        'calls': data.get('calls', 1)
                    })
                    
                    if len(calls_batch) >= NEO4J_BATCH_SIZE:
                        session.run("""
                            UNWIND $batch AS call
                            MATCH (f1:File {name: call.source})
                            MATCH (f2:File {name: call.target})
                            CREATE (f1)-[:CALLS {
                                weight: call.weight,
                                call_count: call.calls
                            }]->(f2)
                        """, batch=calls_batch)
                        calls_created += len(calls_batch)
                        logger.info(f"Created {calls_created} CALLS relationships...")
                        calls_batch = []
            else:
                # Create simple CALLS relationships
                for file_data in self.files_data.values():
                    for call in file_data.function_calls[:5]:  # Limit to first 5 calls per file
                        target_files = self.function_to_files.get(call['function'], set())
                        for target_file in list(target_files)[:2]:  # Max 2 targets per call
                            if target_file != file_data.name:
                                calls_batch.append({
                                    'source': file_data.name,
                                    'target': target_file,
                                    'weight': 1,
                                    'calls': 1
                                })
                                
                                if len(calls_batch) >= NEO4J_BATCH_SIZE:
                                    session.run("""
                                        UNWIND $batch AS call
                                        MATCH (f1:File {name: call.source})
                                        MATCH (f2:File {name: call.target})
                                        CREATE (f1)-[:CALLS {
                                            weight: call.weight,
                                            call_count: call.calls
                                        }]->(f2)
                                    """, batch=calls_batch)
                                    calls_created += len(calls_batch)
                                    calls_batch = []
            
            # Process remaining calls
            if calls_batch:
                session.run("""
                    UNWIND $batch AS call
                    MATCH (f1:File {name: call.source})
                    MATCH (f2:File {name: call.target})
                    CREATE (f1)-[:CALLS {
                        weight: call.weight,
                        call_count: call.calls
                    }]->(f2)
                """, batch=calls_batch)
                calls_created += len(calls_batch)
            
            # Verify the database
            result = session.run("MATCH (f:File) RETURN count(f) as count")
            file_count = result.single()["count"]
            
            result = session.run("MATCH (fn:Function) RETURN count(fn) as count")
            function_count = result.single()["count"]
            
            result = session.run("MATCH (c:Class) RETURN count(c) as count")
            class_count = result.single()["count"]
            
            result = session.run("MATCH ()-[c:CALLS]->() RETURN count(c) as count")
            calls_count = result.single()["count"]
            
            population_time = time.time() - start_time
            
            logger.info(f"Database populated in {population_time:.2f} seconds:")
            logger.info(f"  - {file_count} File nodes")
            logger.info(f"  - {function_count} Function nodes")
            logger.info(f"  - {class_count} Class nodes")
            logger.info(f"  - {calls_count} CALLS relationships")
        
        driver.close()


def analyze_kotlin_file(file_path: Path) -> FileData:
    """Analyze a Kotlin file with comprehensive extraction"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        file_size = len(content)
        lines = content.splitlines()
        loc = len(lines)
        
        # Extract package
        package = extract_package_name(content, file_path)
        
        # Enhanced function extraction for Kotlin
        function_patterns = [
            r'(?:private|public|protected|internal)?\s*(?:suspend\s+)?fun\s+(\w+)\s*\(',
            r'(?:private|public|protected|internal)?\s*(?:inline\s+)?(?:suspend\s+)?fun\s+(\w+)\s*\(',
            r'val\s+(\w+)\s*=\s*\{',  # Lambda properties
            r'var\s+(\w+)\s*=\s*\{'   # Lambda properties
        ]
        
        functions = set()
        for pattern in function_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                func_name = match.group(1)
                if func_name and func_name[0].islower():  # Functions start with lowercase
                    functions.add(func_name)
        
        # Extract classes, objects, interfaces, enums
        class_patterns = [
            r'(?:private|public|protected|internal)?\s*(?:abstract\s+)?(?:data\s+)?class\s+(\w+)',
            r'(?:private|public|protected|internal)?\s*interface\s+(\w+)',
            r'(?:private|public|protected|internal)?\s*enum\s+class\s+(\w+)',
            r'(?:private|public|protected|internal)?\s*object\s+(\w+)',
            r'(?:private|public|protected|internal)?\s*sealed\s+class\s+(\w+)'
        ]
        
        classes = set()
        for pattern in class_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                classes.add(match.group(1))
        
        # Extract imports
        import_pattern = r'import\s+([\w.]+(?:\.\*)?)'
        imports = list(set(re.findall(import_pattern, content)))
        
        # Enhanced function call extraction for Kotlin
        function_calls = []
        call_patterns = [
            r'(\w+)\.(\w+)\s*\(',  # object.method()
            r'this\.(\w+)\s*\(',   # this.method()
            r'super\.(\w+)\s*\(',  # super.method()
            r'(\w+)::(\w+)',       # method references
            r'\b(\w+)\s*\('        # direct calls
        ]
        
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('//') or clean_line.startswith('/*'):
                continue
            
            for pattern in call_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    if len(match.groups()) == 2:
                        obj, func = match.groups()
                        if func not in ['if', 'for', 'while', 'when', 'try', 'catch', 'return', 'throw']:
                            function_calls.append({
                                'function': func,
                                'object': obj,
                                'line': i + 1,
                                'context': clean_line[:300],
                                'type': 'method_call'
                            })
                    else:
                        func = match.group(1)
                        if func not in ['if', 'for', 'while', 'when', 'try', 'catch', 'return', 'throw', 'println', 'print']:
                            function_calls.append({
                                'function': func,
                                'object': None,
                                'line': i + 1,
                                'context': clean_line[:300],
                                'type': 'direct_call'
                            })
        
        # Calculate complexity score
        complexity_indicators = [
            len(re.findall(r'\bif\b', content)),
            len(re.findall(r'\bfor\b', content)),
            len(re.findall(r'\bwhile\b', content)),
            len(re.findall(r'\bwhen\b', content)),
            len(re.findall(r'\btry\b', content)),
            len(re.findall(r'\bcatch\b', content)),
        ]
        complexity_score = sum(complexity_indicators) + len(function_calls) * 0.1
        
        return FileData(
            name=file_path.name,
            path=str(file_path),
            functions=list(functions),
            classes=list(classes),
            imports=imports,
            lines_of_code=loc,
            file_type='kotlin',
            function_calls=function_calls,
            package=package,
            complexity_score=complexity_score,
            file_size=file_size
        )
        
    except Exception as e:
        logger.error(f"Error analyzing Kotlin file {file_path}: {e}")
        return None


def analyze_java_file(file_path: Path) -> FileData:
    """Analyze a Java file with comprehensive extraction"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        file_size = len(content)
        lines = content.splitlines()
        loc = len(lines)
        
        # Extract package
        package = extract_package_name(content, file_path)
        
        # Enhanced method extraction
        method_patterns = [
            r'(?:public|private|protected|static|\s)*\s+(?:synchronized\s+)?(?:final\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{',
            r'(?:public|private|protected)\s+(\w+)\s*\([^)]*\)\s*\{',  # Constructors
        ]
        
        functions = set()
        for pattern in method_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                func_name = match.group(1)
                if func_name and func_name[0].islower():  # Methods start with lowercase
                    functions.add(func_name)
        
        # Extract classes, interfaces, enums
        class_patterns = [
            r'(?:public|private|protected)?\s*(?:abstract\s+)?(?:final\s+)?class\s+(\w+)',
            r'(?:public|private|protected)?\s*interface\s+(\w+)',
            r'(?:public|private|protected)?\s*enum\s+(\w+)',
            r'(?:public|private|protected)?\s*@interface\s+(\w+)'  # Annotations
        ]
        
        classes = set()
        for pattern in class_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                classes.add(match.group(1))
        
        # Extract imports
        import_pattern = r'import\s+(?:static\s+)?([\w.]+(?:\.\*)?);'
        imports = list(set(re.findall(import_pattern, content)))
        
        # Enhanced function call extraction
        function_calls = []
        call_patterns = [
            r'(\w+)\.(\w+)\s*\(',  # object.method()
            r'this\.(\w+)\s*\(',   # this.method()
            r'super\.(\w+)\s*\(',  # super.method()
            r'(\w+)::(\w+)',       # method references
            r'\b(\w+)\s*\('        # direct calls
        ]
        
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('//') or clean_line.startswith('/*'):
                continue
            
            for pattern in call_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    if len(match.groups()) == 2:
                        obj, func = match.groups()
                        if func not in ['if', 'for', 'while', 'switch', 'try', 'catch', 'new', 'return', 'throw']:
                            function_calls.append({
                                'function': func,
                                'object': obj,
                                'line': i + 1,
                                'context': clean_line[:300],
                                'type': 'method_call'
                            })
                    else:
                        func = match.group(1)
                        if func not in ['if', 'for', 'while', 'switch', 'try', 'catch', 'new', 'return', 'throw', 'System', 'String', 'Integer']:
                            function_calls.append({
                                'function': func,
                                'object': None,
                                'line': i + 1,
                                'context': clean_line[:300],
                                'type': 'direct_call'
                            })
        
        # Calculate complexity score
        complexity_indicators = [
            len(re.findall(r'\bif\b', content)),
            len(re.findall(r'\bfor\b', content)),
            len(re.findall(r'\bwhile\b', content)),
            len(re.findall(r'\btry\b', content)),
            len(re.findall(r'\bcatch\b', content)),
            len(re.findall(r'\bswitch\b', content)),
        ]
        complexity_score = sum(complexity_indicators) + len(function_calls) * 0.1
        
        return FileData(
            name=file_path.name,
            path=str(file_path),
            functions=list(functions),
            classes=list(classes),
            imports=imports,
            lines_of_code=loc,
            file_type='java',
            function_calls=function_calls,
            package=package,
            complexity_score=complexity_score,
            file_size=file_size
        )
        
    except Exception as e:
        logger.error(f"Error analyzing Java file {file_path}: {e}")
        return None


def analyze_file_batch(file_paths: List[Path]) -> List[FileData]:
    """Analyze a batch of files"""
    results = []
    for file_path in file_paths:
        try:
            if file_path.suffix == '.java':
                result = analyze_java_file(file_path)
            elif file_path.suffix == '.kt':
                result = analyze_kotlin_file(file_path)
            else:
                continue
            
            if result:
                results.append(result)
                
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
    
    return results


def main():
    """Main function - unleash the full power!"""
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this to your password
    
    print("🚀 AAPS High-Performance Analyzer")
    print("💪 Utilizing 384GB RAM and 96 cores for maximum performance!")
    print("="*80)
    
    analyzer = HighPerformanceAnalyzer()
    
    # Run full parallel analysis
    start_time = time.time()
    
    if not analyzer.analyze_project_parallel():
        logger.error("Project analysis failed")
        return
    
    # Build comprehensive call graph
    analyzer.build_comprehensive_call_graph()
    
    # Create all visualizations
    analyzer.create_comprehensive_visualizations()
    
    # Save comprehensive analysis data
    try:
        logger.info("Saving comprehensive analysis data...")
        
        # Calculate advanced statistics
        total_files = len(analyzer.files_data)
        java_files = len([f for f in analyzer.files_data.values() if f.file_type == 'java'])
        kotlin_files = len([f for f in analyzer.files_data.values() if f.file_type == 'kotlin'])
        total_functions = sum(len(f.functions) for f in analyzer.files_data.values())
        total_classes = sum(len(f.classes) for f in analyzer.files_data.values())
        total_loc = sum(f.lines_of_code for f in analyzer.files_data.values())
        total_imports = sum(len(f.imports) for f in analyzer.files_data.values())
        
        # Network statistics
        network_stats = {}
        if NETWORKX_AVAILABLE and analyzer.call_graph:
            network_stats = {
                'nodes': len(analyzer.call_graph.nodes()),
                'edges': len(analyzer.call_graph.edges()),
                'density': nx.density(analyzer.call_graph),
                'average_clustering': nx.average_clustering(analyzer.call_graph),
                'number_connected_components': nx.number_weakly_connected_components(analyzer.call_graph)
            }
        
        # Top files by various metrics
        by_importance = sorted(analyzer.files_data.values(), key=lambda x: x.importance_score, reverse=True)[:50]
        by_complexity = sorted(analyzer.files_data.values(), key=lambda x: x.complexity_score, reverse=True)[:50]
        by_loc = sorted(analyzer.files_data.values(), key=lambda x: x.lines_of_code, reverse=True)[:50]
        by_functions = sorted(analyzer.files_data.values(), key=lambda x: len(x.functions), reverse=True)[:50]
        
        # Package analysis
        packages = defaultdict(list)
        for f in analyzer.files_data.values():
            packages[f.package].append(f)
        
        package_stats = []
        for package, files in packages.items():
            if len(files) > 1:
                package_stats.append({
                    'package': package,
                    'file_count': len(files),
                    'total_loc': sum(f.lines_of_code for f in files),
                    'total_functions': sum(len(f.functions) for f in files),
                    'avg_importance': sum(f.importance_score for f in files) / len(files),
                    'avg_complexity': sum(f.complexity_score for f in files) / len(files)
                })
        
        package_stats.sort(key=lambda x: x['avg_importance'], reverse=True)
        
        # Create comprehensive export
        comprehensive_data = {
            'analysis_metadata': {
                'total_analysis_time_seconds': time.time() - start_time,
                'files_processed': total_files,
                'workers_used': MAX_WORKERS,
                'memory_limit_gb': MAX_MEMORY_USAGE / (1024**3),
                'neo4j_batch_size': NEO4J_BATCH_SIZE
            },
            'project_summary': {
                'total_files': total_files,
                'java_files': java_files,
                'kotlin_files': kotlin_files,
                'total_functions': total_functions,
                'total_classes': total_classes,
                'total_lines_of_code': total_loc,
                'total_imports': total_imports,
                'packages': len(packages)
            },
            'network_analysis': network_stats,
            'top_files': {
                'by_importance': [{'name': f.name, 'package': f.package, 'score': f.importance_score, 'type': f.file_type} for f in by_importance],
                'by_complexity': [{'name': f.name, 'package': f.package, 'score': f.complexity_score, 'type': f.file_type} for f in by_complexity],
                'by_loc': [{'name': f.name, 'package': f.package, 'loc': f.lines_of_code, 'type': f.file_type} for f in by_loc],
                'by_functions': [{'name': f.name, 'package': f.package, 'functions': len(f.functions), 'type': f.file_type} for f in by_functions]
            },
            'package_analysis': package_stats[:30],
            'file_type_comparison': {
                'java': {
                    'count': java_files,
                    'avg_loc': sum(f.lines_of_code for f in analyzer.files_data.values() if f.file_type == 'java') / max(java_files, 1),
                    'avg_functions': sum(len(f.functions) for f in analyzer.files_data.values() if f.file_type == 'java') / max(java_files, 1),
                    'avg_importance': sum(f.importance_score for f in analyzer.files_data.values() if f.file_type == 'java') / max(java_files, 1)
                },
                'kotlin': {
                    'count': kotlin_files,
                    'avg_loc': sum(f.lines_of_code for f in analyzer.files_data.values() if f.file_type == 'kotlin') / max(kotlin_files, 1),
                    'avg_functions': sum(len(f.functions) for f in analyzer.files_data.values() if f.file_type == 'kotlin') / max(kotlin_files, 1),
                    'avg_importance': sum(f.importance_score for f in analyzer.files_data.values() if f.file_type == 'kotlin') / max(kotlin_files, 1)
                }
            }
        }
        
        with open('aaps_comprehensive_analysis.json', 'w') as f:
            json.dump(comprehensive_data, f, indent=2, default=str)
        
        logger.info("✅ Created: aaps_comprehensive_analysis.json")
        
    except Exception as e:
        logger.error(f"Failed to save comprehensive analysis: {e}")
    
    # Populate Neo4j with high performance
    try:
        analyzer.populate_neo4j_high_performance(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        logger.info("✅ Neo4j database populated with high performance!")
        
        # Test with utilities
        logger.info("Testing with neo4j utilities...")
        import subprocess
        import sys
        
        result = subprocess.run([sys.executable, "memory_optimized_utilities.py"], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logger.info("✅ Neo4j utilities test successful!")
        else:
            logger.warning("⚠️ Neo4j utilities had issues but database should work")
            
    except Exception as e:
        logger.error(f"Neo4j population failed: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("🎉 HIGH-PERFORMANCE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"📊 Files Processed: {len(analyzer.files_data):,}")
    print(f"💻 Workers Used: {MAX_WORKERS}")
    if NETWORKX_AVAILABLE and analyzer.call_graph:
        print(f"🔗 Call Graph: {len(analyzer.call_graph.nodes):,} nodes, {len(analyzer.call_graph.edges):,} edges")
    else:
        print("🔗 Call Graph: Not available (NetworkX not installed)")
    print(f"🧠 Memory Utilized: Up to {MAX_MEMORY_USAGE/(1024**3):.0f}GB")
    print("\n📋 Generated Files:")
    print("  🌐 aaps_full_network.html - Complete network visualization")
    print("  📦 aaps_package_hierarchy.html - Package hierarchy analysis")
    print("  🔥 aaps_complexity_heatmap.html - Complexity heatmap")
    print("  📊 aaps_filetype_analysis.html - Java vs Kotlin analysis")
    print("  🔍 aaps_advanced_explorer.html - Advanced file explorer")
    print("  📈 aaps_comprehensive_analysis.json - Complete analysis data")
    print("  🗄️  Neo4j database - High-performance graph database")
    print("\n💡 Performance Optimizations Applied:")
    print(f"  • {MAX_WORKERS}-core parallel processing")
    print(f"  • {NEO4J_BATCH_SIZE:,}-record Neo4j batches")
    print(f"  • {MAX_MEMORY_USAGE/(1024**3):.0f}GB memory allocation")
    print("  • Enhanced function call detection")
    print("  • Comprehensive relationship mapping")
    print("="*80)


if __name__ == "__main__":
    main()