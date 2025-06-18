#!/usr/bin/env python3
"""
Ultimate High-Performance AAPS Multi-Repository Analyzer
Uses maximum available RAM and all CPU cores for blazing fast analysis
Single script - just run: python aaps_analyzer.py
"""

import os
import re
import json
import gc
import time
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Iterator
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, Counter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import threading
import queue
from functools import partial

# Optional imports
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

# Configure for MAXIMUM performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AGGRESSIVE performance settings - use ALL available resources
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
AVAILABLE_RAM_GB = psutil.virtual_memory().available / (1024**3)
CPU_CORES = psutil.cpu_count(logical=True)  # Use ALL cores including hyperthreading

# Use 90% of available RAM and ALL cores
MAX_WORKERS = min(CPU_CORES, 128)  # Use all cores, cap at 128 for stability
CHUNK_SIZE = max(10, min(200, int(AVAILABLE_RAM_GB / 2)))  # Dynamic chunk size based on RAM
MAX_MEMORY_USAGE = int(AVAILABLE_RAM_GB * 0.9 * 1024**3)  # Use 90% of available RAM
NEO4J_BATCH_SIZE = min(50000, int(AVAILABLE_RAM_GB * 100))  # Large batches with lots of RAM

logger.info(f"🚀 ULTIMATE PERFORMANCE MODE ACTIVATED!")
logger.info(f"💾 Total RAM: {TOTAL_RAM_GB:.1f}GB, Available: {AVAILABLE_RAM_GB:.1f}GB")
logger.info(f"⚡ CPU Cores: {CPU_CORES}, Workers: {MAX_WORKERS}")
logger.info(f"📦 Chunk Size: {CHUNK_SIZE}, Batch Size: {NEO4J_BATCH_SIZE}")

# Repository configuration
REPOSITORIES = {
    "EN_new": {
        "url": "https://github.com/dicko72/AAPS-EatingNow.git",
        "branch": "EN-MASTER-NEW", 
        "local_path": "./aaps_en_new"
    },
    "EN_old": {
        "url": "https://github.com/dicko72/AAPS-EatingNow.git",
        "branch": "master",
        "local_path": "./aaps_en_old"
    },
    "AAPS_source": {
        "url": "https://github.com/nightscout/AndroidAPS.git",
        "branch": "master",
        "local_path": "./aaps_source"
    }
}

@dataclass
class FileData:
    """Enhanced file data structure"""
    name: str
    path: str
    repository: str
    file_type: str
    lines_of_code: int
    function_count: int
    class_count: int
    import_count: int
    package: str
    complexity_score: float
    importance_score: float
    file_size: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    function_calls: List[Dict[str, str]]

class MemoryManager:
    """Aggressive memory management for maximum utilization"""
    
    def __init__(self):
        self.total_memory = psutil.virtual_memory().total
        self.target_usage = self.total_memory * 0.9  # Use 90% of total RAM
        
    def get_current_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if psutil.virtual_memory().percent > 95:
            gc.collect()
            logger.info(f"🧹 Memory cleanup: {psutil.virtual_memory().percent:.1f}% usage")

class UltimateRepositoryManager:
    """Ultimate repository management with maximum parallelization"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        
    def clone_all_repositories_parallel(self) -> bool:
        """Clone all repositories in parallel for maximum speed"""
        logger.info("🚀 Cloning all repositories in parallel...")
        
        def clone_single_repo(repo_item):
            repo_name, repo_config = repo_item
            return self._clone_repository_aggressive(repo_name, repo_config)
        
        # Clone all repositories simultaneously
        with ThreadPoolExecutor(max_workers=len(REPOSITORIES)) as executor:
            futures = {executor.submit(clone_single_repo, item): item[0] 
                      for item in REPOSITORIES.items()}
            
            results = {}
            for future in as_completed(futures):
                repo_name = futures[future]
                try:
                    results[repo_name] = future.result()
                except Exception as e:
                    logger.error(f"Failed to clone {repo_name}: {e}")
                    results[repo_name] = False
        
        success_count = sum(results.values())
        logger.info(f"✅ Successfully cloned {success_count}/{len(REPOSITORIES)} repositories")
        return success_count > 0
    
    def _clone_repository_aggressive(self, repo_name: str, repo_config: Dict) -> bool:
        """Aggressively clone repository with optimizations"""
        repo_url = repo_config["url"]
        branch = repo_config["branch"]
        local_path = Path(repo_config["local_path"])
        
        logger.info(f"🔄 Cloning {repo_name}...")
        
        try:
            if local_path.exists():
                logger.info(f"📁 {repo_name} already exists, skipping")
                return True
            
            # Use shallow clone for speed
            commands_to_try = [
                ["git", "clone", "--depth", "1", "--single-branch", "-b", branch, repo_url, str(local_path)],
                ["git", "clone", "--depth", "1", repo_url, str(local_path)]
            ]
            
            for cmd in commands_to_try:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        logger.info(f"✅ {repo_name} cloned successfully")
                        return True
                except subprocess.TimeoutExpired:
                    logger.warning(f"⏰ Clone timeout for {repo_name}, trying next method")
                    continue
            
            logger.error(f"❌ Failed to clone {repo_name}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error cloning {repo_name}: {e}")
            return False

class UltimateFileAnalyzer:
    """Ultimate file analyzer using all available resources"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        # Pre-compile regex patterns for maximum speed
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Pre-compile all regex patterns for speed"""
        # Java patterns
        self.java_method_pattern = re.compile(
            r'(?:public|private|protected|static|\s)*\s+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
        )
        self.java_class_pattern = re.compile(
            r'(?:public|private|protected)?\s*(?:abstract\s+)?(?:final\s+)?(?:class|interface|enum)\s+(\w+)'
        )
        self.java_import_pattern = re.compile(r'import\s+(?:static\s+)?([\w.]+(?:\.\*)?);')
        
        # Kotlin patterns
        self.kotlin_function_pattern = re.compile(
            r'(?:private|public|protected|internal)?\s*(?:suspend\s+)?fun\s+(\w+)\s*\('
        )
        self.kotlin_class_pattern = re.compile(
            r'(?:private|public|protected|internal)?\s*(?:abstract\s+)?(?:data\s+)?(?:class|interface|object|enum class)\s+(\w+)'
        )
        self.kotlin_import_pattern = re.compile(r'import\s+([\w.]+(?:\.\*)?)')
        
        # Package pattern
        self.package_pattern = re.compile(r'package\s+([\w.]+)')
        
        # Function call patterns
        self.call_patterns = [
            re.compile(r'(\w+)\.(\w+)\s*\('),  # object.method()
            re.compile(r'this\.(\w+)\s*\('),   # this.method()
            re.compile(r'super\.(\w+)\s*\('),  # super.method()
            re.compile(r'\b(\w+)\s*\(')        # direct calls
        ]
    
    def analyze_file_ultimate(self, file_path: Path, repository: str) -> FileData:
        """Ultimate file analysis with maximum efficiency"""
        try:
            # Read file efficiently
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.splitlines()
            loc = len(lines)
            file_size = len(content.encode('utf-8'))
            
            # Extract package
            package_match = self.package_pattern.search(content)
            package = package_match.group(1) if package_match else "unknown"
            
            # File type specific extraction
            if file_path.suffix == '.java':
                functions, classes, imports, function_calls, complexity = self._analyze_java_ultimate(content, lines)
            elif file_path.suffix == '.kt':
                functions, classes, imports, function_calls, complexity = self._analyze_kotlin_ultimate(content, lines)
            else:
                functions, classes, imports, function_calls, complexity = [], [], [], [], 0
            
            # Calculate importance score
            importance = self._calculate_importance_ultimate(
                file_path.name, package, loc, len(functions), len(classes), repository, complexity
            )
            
            return FileData(
                name=file_path.name,
                path=str(file_path),
                repository=repository,
                file_type=file_path.suffix[1:] if file_path.suffix else 'unknown',
                lines_of_code=loc,
                function_count=len(functions),
                class_count=len(classes),
                import_count=len(imports),
                package=package,
                complexity_score=complexity,
                importance_score=importance,
                file_size=file_size,
                functions=functions,
                classes=classes,
                imports=imports,
                function_calls=function_calls
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def _analyze_java_ultimate(self, content: str, lines: List[str]) -> Tuple[List[str], List[str], List[str], List[Dict], float]:
        """Ultimate Java analysis"""
        # Extract functions
        functions = [m.group(1) for m in self.java_method_pattern.finditer(content) 
                    if m.group(1) and m.group(1)[0].islower()]
        
        # Extract classes
        classes = [m.group(1) for m in self.java_class_pattern.finditer(content)]
        
        # Extract imports
        imports = [m.group(1) for m in self.java_import_pattern.finditer(content)]
        
        # Extract function calls
        function_calls = self._extract_function_calls_ultimate(lines)
        
        # Calculate complexity
        complexity_keywords = ['if ', 'for ', 'while ', 'try ', 'catch ', 'switch ']
        complexity = sum(content.count(keyword) for keyword in complexity_keywords)
        
        return functions, classes, imports, function_calls, complexity
    
    def _analyze_kotlin_ultimate(self, content: str, lines: List[str]) -> Tuple[List[str], List[str], List[str], List[Dict], float]:
        """Ultimate Kotlin analysis"""
        # Extract functions
        functions = [m.group(1) for m in self.kotlin_function_pattern.finditer(content) 
                    if m.group(1) and m.group(1)[0].islower()]
        
        # Extract classes
        classes = [m.group(1) for m in self.kotlin_class_pattern.finditer(content)]
        
        # Extract imports
        imports = [m.group(1) for m in self.kotlin_import_pattern.finditer(content)]
        
        # Extract function calls
        function_calls = self._extract_function_calls_ultimate(lines)
        
        # Calculate complexity
        complexity_keywords = ['if ', 'for ', 'while ', 'when ', 'try ', 'catch ']
        complexity = sum(content.count(keyword) for keyword in complexity_keywords)
        
        return functions, classes, imports, function_calls, complexity
    
    def _extract_function_calls_ultimate(self, lines: List[str]) -> List[Dict[str, str]]:
        """Ultimate function call extraction"""
        function_calls = []
        skip_keywords = {'if', 'for', 'while', 'when', 'try', 'catch', 'return', 'throw', 'new'}
        
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line or clean_line.startswith(('//','/*')):
                continue
            
            for pattern in self.call_patterns:
                for match in pattern.finditer(line):
                    if len(match.groups()) == 2:
                        obj, func = match.groups()
                        if func not in skip_keywords:
                            function_calls.append({
                                'function': func,
                                'object': obj,
                                'line': i + 1,
                                'type': 'method_call'
                            })
                    else:
                        func = match.group(1)
                        if func not in skip_keywords:
                            function_calls.append({
                                'function': func,
                                'object': None,
                                'line': i + 1,
                                'type': 'direct_call'
                            })
        
        return function_calls
    
    def _calculate_importance_ultimate(self, filename: str, package: str, loc: int, 
                                     func_count: int, class_count: int, repository: str, complexity: float) -> float:
        """Ultimate importance calculation with enhanced scoring"""
        score = 0.0
        
        # Base metrics (logarithmic scaling for large codebases)
        score += min(20, loc / 50)  # LOC contribution
        score += min(25, func_count * 2)  # Function contribution
        score += min(15, class_count * 4)  # Class contribution
        score += min(10, complexity * 0.5)  # Complexity contribution
        
        # Advanced name-based scoring
        name_lower = filename.lower()
        package_lower = package.lower()
        
        # Critical functionality patterns
        critical_patterns = {
            'core': 25, 'main': 20, 'manager': 18, 'service': 16, 'controller': 16,
            'algorithm': 22, 'engine': 20, 'processor': 15, 'handler': 12,
            'pump': 20, 'cgm': 18, 'glucose': 18, 'insulin': 20, 'blood': 15,
            'loop': 18, 'automation': 16, 'treatment': 14, 'dose': 16,
            'plugin': 14, 'driver': 12, 'interface': 10, 'api': 12
        }
        
        for pattern, bonus in critical_patterns.items():
            if pattern in name_lower:
                score += bonus
            if pattern in package_lower:
                score += bonus * 0.6
        
        # Repository-specific bonuses
        repo_bonuses = {
            'AAPS_source': 8,  # Main source gets bonus
            'EN_new': 5,       # New version gets bonus
            'EN_old': 2        # Old version gets small bonus
        }
        score += repo_bonuses.get(repository, 0)
        
        # Architecture importance
        if any(arch in package_lower for arch in ['core', 'main', 'base', 'foundation']):
            score += 15
        
        # Penalties for less important files
        if any(penalty in name_lower for penalty in ['test', 'mock', 'stub', 'example']):
            score *= 0.1
        if any(penalty in filename.lower() for penalty in ['generated', 'build', 'temp']):
            score *= 0.2
        
        return round(score, 2)

class UltimateMultiRepoAnalyzer:
    """Ultimate multi-repository analyzer using ALL system resources"""
    
    def __init__(self):
        self.repositories = REPOSITORIES
        self.memory_manager = MemoryManager()
        self.repo_manager = UltimateRepositoryManager()
        self.file_analyzer = UltimateFileAnalyzer()
        
        # In-memory storage for maximum speed
        self.files_data = {}
        self.call_graphs = {}
        self.function_mappings = defaultdict(lambda: defaultdict(set))
        
        # Initialize call graphs
        if NETWORKX_AVAILABLE:
            for repo_name in self.repositories.keys():
                self.call_graphs[repo_name] = nx.DiGraph()
    
    def analyze_all_repositories_ultimate(self) -> bool:
        """Ultimate repository analysis using all available resources"""
        start_time = time.time()
        
        logger.info(f"🚀 STARTING ULTIMATE ANALYSIS")
        logger.info(f"💪 Using {MAX_WORKERS} workers with {CHUNK_SIZE} files per chunk")
        
        # Step 1: Clone all repositories in parallel
        if not self.repo_manager.clone_all_repositories_parallel():
            logger.error("❌ Failed to setup repositories")
            return False
        
        # Step 2: Find all source files across all repositories
        all_repo_files = {}
        total_files = 0
        
        for repo_name, repo_config in self.repositories.items():
            repo_path = Path(repo_config["local_path"])
            if repo_path.exists():
                source_files = self._find_source_files_ultimate(repo_path)
                all_repo_files[repo_name] = source_files
                total_files += len(source_files)
                logger.info(f"📁 {repo_name}: {len(source_files)} source files")
        
        if total_files == 0:
            logger.error("❌ No source files found!")
            return False
        
        logger.info(f"🎯 Total files to process: {total_files}")
        
        # Step 3: Process all files across all repositories in parallel
        logger.info(f"🔥 Processing files with MAXIMUM parallelization...")
        
        # Create work items (file_path, repository) for all files
        all_work_items = []
        for repo_name, files in all_repo_files.items():
            for file_path in files:
                all_work_items.append((file_path, repo_name))
        
        # Split into chunks for parallel processing
        chunks = [all_work_items[i:i+CHUNK_SIZE] for i in range(0, len(all_work_items), CHUNK_SIZE)]
        
        # Process all chunks in parallel using ALL workers
        processed_files = 0
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(analyze_file_batch_ultimate, chunk): i 
                      for i, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    
                    # Store results in memory for maximum speed
                    for file_data in batch_results:
                        if file_data:
                            key = f"{file_data.repository}:{file_data.path}"
                            self.files_data[key] = file_data
                            processed_files += 1
                    
                    # Progress reporting
                    progress = (processed_files / total_files) * 100
                    memory_usage = psutil.virtual_memory().percent
                    logger.info(f"🔥 Progress: {progress:.1f}% ({processed_files}/{total_files}) | Memory: {memory_usage:.1f}%")
                    
                    # Memory optimization
                    self.memory_manager.optimize_memory()
                    
                except Exception as e:
                    logger.error(f"❌ Batch processing failed: {e}")
        
        analysis_time = time.time() - start_time
        logger.info(f"✅ File analysis completed in {analysis_time:.2f} seconds")
        logger.info(f"📊 Successfully processed {len(self.files_data)} files")
        
        # Step 4: Build function mappings and call graphs
        self._build_ultimate_mappings()
        
        # Step 5: Generate all outputs
        self._generate_ultimate_outputs()
        
        return True
    
    def _find_source_files_ultimate(self, repo_path: Path) -> List[Path]:
        """Ultimate source file discovery"""
        source_files = []
        skip_dirs = {'.git', 'build', 'target', '.gradle', 'node_modules', '.idea', 'bin', 'out'}
        
        for root, dirs, files in os.walk(repo_path):
            # Skip unnecessary directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith(('.java', '.kt')):
                    source_files.append(Path(root) / file)
        
        return source_files
    
    def _build_ultimate_mappings(self):
        """Build ultimate function mappings and call graphs"""
        logger.info("🔗 Building function mappings and call graphs...")
        
        # Build function mappings
        for file_key, file_data in self.files_data.items():
            repo_name = file_data.repository
            for function_name in file_data.functions:
                self.function_mappings[repo_name][function_name].add(file_data.name)
        
        # Build call graphs if NetworkX is available
        if NETWORKX_AVAILABLE:
            for repo_name in self.repositories.keys():
                self._build_call_graph_for_repo(repo_name)
    
    def _build_call_graph_for_repo(self, repo_name: str):
        """Build call graph for a specific repository"""
        repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
        call_graph = self.call_graphs[repo_name]
        
        # Add nodes
        for file_data in repo_files:
            call_graph.add_node(
                file_data.name,
                path=file_data.path,
                importance=file_data.importance_score,
                complexity=file_data.complexity_score,
                loc=file_data.lines_of_code
            )
        
        # Add edges based on function calls
        call_counts = defaultdict(int)
        
        for file_data in repo_files:
            for call in file_data.function_calls:
                function_name = call['function']
                target_files = self.function_mappings[repo_name].get(function_name, set())
                
                for target_file in target_files:
                    if target_file != file_data.name:
                        call_counts[(file_data.name, target_file)] += 1
        
        # Add edges
        for (source, target), weight in call_counts.items():
            if weight > 0:
                call_graph.add_edge(source, target, weight=weight)
        
        logger.info(f"📊 {repo_name} call graph: {len(call_graph.nodes)} nodes, {len(call_graph.edges)} edges")
    
    def _generate_ultimate_outputs(self):
        """Generate all ultimate outputs"""
        logger.info("📈 Generating ultimate outputs...")
        
        # 1. Generate comprehensive JSON report
        self._generate_ultimate_json_report()
        
        # 2. Populate Neo4j if available
        if NEO4J_AVAILABLE:
            try:
                self._populate_neo4j_ultimate()
            except Exception as e:
                logger.warning(f"Neo4j population failed: {e}")
        
        # 3. Generate visualizations if available
        if PLOTLY_AVAILABLE:
            try:
                self._generate_ultimate_visualizations()
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
        
        logger.info("🎉 All outputs generated successfully!")
    
    def _generate_ultimate_json_report(self):
        """Generate ultimate JSON report"""
        logger.info("📄 Generating comprehensive JSON report...")
        
        report = {
            'analysis_metadata': {
                'timestamp': time.time(),
                'total_ram_gb': TOTAL_RAM_GB,
                'available_ram_gb': AVAILABLE_RAM_GB,
                'cpu_cores': CPU_CORES,
                'max_workers': MAX_WORKERS,
                'chunk_size': CHUNK_SIZE,
                'repositories_analyzed': len(self.repositories),
                'total_files_processed': len(self.files_data)
            },
            'repository_summaries': {},
            'global_summary': {
                'total_files': len(self.files_data),
                'total_loc': sum(f.lines_of_code for f in self.files_data.values()),
                'total_functions': sum(f.function_count for f in self.files_data.values()),
                'total_classes': sum(f.class_count for f in self.files_data.values())
            },
            'top_files_global': []
        }
        
        # Per-repository summaries
        for repo_name in self.repositories.keys():
            repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
            
            if repo_files:
                # Calculate repository metrics
                total_loc = sum(f.lines_of_code for f in repo_files)
                total_functions = sum(f.function_count for f in repo_files)
                total_classes = sum(f.class_count for f in repo_files)
                avg_importance = sum(f.importance_score for f in repo_files) / len(repo_files)
                
                # Top files by importance
                top_files = sorted(repo_files, key=lambda x: x.importance_score, reverse=True)[:20]
                
                report['repository_summaries'][repo_name] = {
                    'file_count': len(repo_files),
                    'total_loc': total_loc,
                    'total_functions': total_functions,
                    'total_classes': total_classes,
                    'avg_importance': round(avg_importance, 2),
                    'top_files': [
                        {
                            'name': f.name,
                            'importance': f.importance_score,
                            'loc': f.lines_of_code,
                            'functions': f.function_count,
                            'package': f.package
                        }
                        for f in top_files
                    ]
                }
        
        # Global top files
        all_files = list(self.files_data.values())
        top_global = sorted(all_files, key=lambda x: x.importance_score, reverse=True)[:50]
        
        report['top_files_global'] = [
            {
                'name': f.name,
                'repository': f.repository,
                'importance': f.importance_score,
                'loc': f.lines_of_code,
                'functions': f.function_count,
                'package': f.package
            }
            for f in top_global
        ]
        
        # Save report
        with open('aaps_ultimate_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("✅ Saved: aaps_ultimate_analysis.json")
    
    def _populate_neo4j_ultimate(self):
        """Ultimate Neo4j population using maximum performance"""
        logger.info("🗄️ Populating Neo4j with ultimate performance...")
        
        # Neo4j connection with maximum performance settings
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password"),
            max_connection_lifetime=3600,
            max_connection_pool_size=100
        )
        
        try:
            with driver.session() as session:
                # Clear database
                session.run("MATCH (n) DETACH DELETE n")
                
                # Create repository nodes
                repo_data = []
                for repo_name in self.repositories.keys():
                    repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
                    if repo_files:
                        repo_data.append({
                            'name': repo_name,
                            'file_count': len(repo_files),
                            'total_loc': sum(f.lines_of_code for f in repo_files),
                            'total_functions': sum(f.function_count for f in repo_files)
                        })
                
                if repo_data:
                    session.run("""
                        UNWIND $repos AS repo
                        CREATE (r:Repository {
                            name: repo.name,
                            file_count: repo.file_count,
                            total_loc: repo.total_loc,
                            total_functions: repo.total_functions
                        })
                    """, repos=repo_data)
                
                # Create file nodes in large batches
                file_batches = []
                current_batch = []
                
                for file_data in self.files_data.values():
                    current_batch.append({
                        'name': file_data.name,
                        'path': file_data.path,
                        'repository': file_data.repository,
                        'file_type': file_data.file_type,
                        'lines_of_code': file_data.lines_of_code,
                        'function_count': file_data.function_count,
                        'class_count': file_data.class_count,
                        'package': file_data.package,
                        'importance_score': file_data.importance_score,
                        'complexity_score': file_data.complexity_score
                    })
                    
                    if len(current_batch) >= NEO4J_BATCH_SIZE:
                        file_batches.append(current_batch)
                        current_batch = []
                
                if current_batch:
                    file_batches.append(current_batch)
                
                # Process file batches
                for i, batch in enumerate(file_batches):
                    session.run("""
                        UNWIND $batch AS file
                        MATCH (r:Repository {name: file.repository})
                        CREATE (f:File {
                            name: file.name,
                            path: file.path,
                            repository: file.repository,
                            file_type: file.file_type,
                            lines_of_code: file.lines_of_code,
                            function_count: file.function_count,
                            class_count: file.class_count,
                            package: file.package,
                            importance_score: file.importance_score,
                            complexity_score: file.complexity_score
                        })
                        CREATE (r)-[:CONTAINS]->(f)
                    """, batch=batch)
                    
                    logger.info(f"📊 Neo4j: Processed {(i+1)*len(batch)} files...")
                
                # Create call relationships from call graphs
                if NETWORKX_AVAILABLE:
                    for repo_name, call_graph in self.call_graphs.items():
                        call_batch = []
                        
                        for source, target, data in call_graph.edges(data=True):
                            call_batch.append({
                                'source': source,
                                'target': target,
                                'weight': data.get('weight', 1),
                                'repository': repo_name
                            })
                            
                            if len(call_batch) >= NEO4J_BATCH_SIZE:
                                session.run("""
                                    UNWIND $batch AS call
                                    MATCH (f1:File {name: call.source, repository: call.repository})
                                    MATCH (f2:File {name: call.target, repository: call.repository})
                                    CREATE (f1)-[:CALLS {
                                        weight: call.weight,
                                        repository: call.repository
                                    }]->(f2)
                                """, batch=call_batch)
                                call_batch = []
                        
                        if call_batch:
                            session.run("""
                                UNWIND $batch AS call
                                MATCH (f1:File {name: call.source, repository: call.repository})
                                MATCH (f2:File {name: call.target, repository: call.repository})
                                CREATE (f1)-[:CALLS {
                                    weight: call.weight,
                                    repository: call.repository
                                }]->(f2)
                            """, batch=call_batch)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX file_repo_idx IF NOT EXISTS FOR (f:File) ON (f.repository)",
                    "CREATE INDEX file_importance_idx IF NOT EXISTS FOR (f:File) ON (f.importance_score)",
                    "CREATE INDEX repo_name_idx IF NOT EXISTS FOR (r:Repository) ON (r.name)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except:
                        pass
                
                # Verify population
                result = session.run("MATCH (f:File) RETURN count(f) as files")
                file_count = result.single()["files"]
                
                result = session.run("MATCH ()-[c:CALLS]->() RETURN count(c) as calls")
                calls_count = result.single()["calls"]
                
                logger.info(f"✅ Neo4j populated: {file_count} files, {calls_count} relationships")
        
        finally:
            driver.close()
    
    def _generate_ultimate_visualizations(self):
        """Generate ultimate visualizations"""
        logger.info("📊 Generating ultimate visualizations...")
        
        # 1. Multi-repository overview
        self._create_multi_repo_overview()
        
        # 2. Individual repository networks
        for repo_name in self.repositories.keys():
            if NETWORKX_AVAILABLE and repo_name in self.call_graphs:
                self._create_repo_network_viz(repo_name)
        
        # 3. Comprehensive comparison
        self._create_comprehensive_comparison()
        
        logger.info("✅ All visualizations generated")
    
    def _create_multi_repo_overview(self):
        """Create multi-repository overview visualization"""
        # Prepare data
        repo_data = []
        for repo_name in self.repositories.keys():
            repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
            if repo_files:
                repo_data.append({
                    'repository': repo_name,
                    'files': len(repo_files),
                    'loc': sum(f.lines_of_code for f in repo_files),
                    'functions': sum(f.function_count for f in repo_files),
                    'avg_importance': sum(f.importance_score for f in repo_files) / len(repo_files)
                })
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Files by Repository', 'Lines of Code', 'Functions', 'Average Importance'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )
        
        repos = [d['repository'] for d in repo_data]
        
        fig.add_trace(go.Bar(x=repos, y=[d['files'] for d in repo_data], name='Files'), row=1, col=1)
        fig.add_trace(go.Bar(x=repos, y=[d['loc'] for d in repo_data], name='LOC'), row=1, col=2)
        fig.add_trace(go.Bar(x=repos, y=[d['functions'] for d in repo_data], name='Functions'), row=2, col=1)
        fig.add_trace(go.Bar(x=repos, y=[d['avg_importance'] for d in repo_data], name='Avg Importance'), row=2, col=2)
        
        fig.update_layout(
            title='AAPS Ultimate Multi-Repository Analysis Overview',
            height=800,
            showlegend=False
        )
        
        pyo.plot(fig, filename='aaps_ultimate_overview.html', auto_open=False)
        logger.info("✅ Created: aaps_ultimate_overview.html")
    
    def _create_repo_network_viz(self, repo_name: str):
        """Create network visualization for a repository"""
        if repo_name not in self.call_graphs:
            return
        
        call_graph = self.call_graphs[repo_name]
        if len(call_graph.nodes) == 0:
            return
        
        # Get top 100 nodes by importance
        repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
        top_files = sorted(repo_files, key=lambda x: x.importance_score, reverse=True)[:100]
        top_nodes = [f.name for f in top_files]
        
        subgraph = call_graph.subgraph(top_nodes)
        
        # Create layout
        try:
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
        except:
            pos = nx.circular_layout(subgraph)
        
        # Create edges
        edge_x, edge_y = [], []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            showlegend=False
        )
        
        # Create nodes
        node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []
        
        for node in subgraph.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Find file data
                file_data = next((f for f in top_files if f.name == node), None)
                if file_data:
                    node_text.append(f"{node}<br>Importance: {file_data.importance_score:.1f}")
                    node_colors.append(file_data.importance_score)
                    node_sizes.append(max(10, min(50, file_data.importance_score)))
                else:
                    node_text.append(node)
                    node_colors.append(0)
                    node_sizes.append(10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance Score")
            ),
            text=node_text,
            hoverinfo='text',
            showlegend=False
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f'AAPS {repo_name} - Network Analysis (Top 100 Files)',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1200,
            height=800
        )
        
        pyo.plot(fig, filename=f'aaps_{repo_name}_network.html', auto_open=False)
        logger.info(f"✅ Created: aaps_{repo_name}_network.html")
    
    def _create_comprehensive_comparison(self):
        """Create comprehensive repository comparison"""
        # Prepare comparison data
        comparison_data = []
        
        for repo_name in self.repositories.keys():
            repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
            if repo_files:
                java_files = [f for f in repo_files if f.file_type == 'java']
                kotlin_files = [f for f in repo_files if f.file_type == 'kt']
                
                comparison_data.append({
                    'repository': repo_name,
                    'total_files': len(repo_files),
                    'java_files': len(java_files),
                    'kotlin_files': len(kotlin_files),
                    'total_loc': sum(f.lines_of_code for f in repo_files),
                    'total_functions': sum(f.function_count for f in repo_files),
                    'avg_importance': sum(f.importance_score for f in repo_files) / len(repo_files),
                    'top_file': max(repo_files, key=lambda x: x.importance_score).name,
                    'top_importance': max(f.importance_score for f in repo_files)
                })
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Repository', 'Files', 'Java', 'Kotlin', 'LOC', 'Functions', 
                       'Avg Importance', 'Top File', 'Max Importance'],
                fill_color='lightblue',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[
                    [d['repository'] for d in comparison_data],
                    [d['total_files'] for d in comparison_data],
                    [d['java_files'] for d in comparison_data],
                    [d['kotlin_files'] for d in comparison_data],
                    [f"{d['total_loc']:,}" for d in comparison_data],
                    [f"{d['total_functions']:,}" for d in comparison_data],
                    [f"{d['avg_importance']:.1f}" for d in comparison_data],
                    [d['top_file'] for d in comparison_data],
                    [f"{d['top_importance']:.1f}" for d in comparison_data]
                ],
                fill_color='white',
                align='left',
                font=dict(size=10)
            )
        )])
        
        fig.update_layout(
            title='AAPS Ultimate Repository Comparison',
            height=400
        )
        
        pyo.plot(fig, filename='aaps_ultimate_comparison.html', auto_open=False)
        logger.info("✅ Created: aaps_ultimate_comparison.html")

def analyze_file_batch_ultimate(work_items: List[Tuple[Path, str]]) -> List[FileData]:
    """Ultimate file batch analysis (used in multiprocessing)"""
    results = []
    analyzer = UltimateFileAnalyzer()
    
    for file_path, repository in work_items:
        try:
            file_data = analyzer.analyze_file_ultimate(file_path, repository)
            if file_data:
                results.append(file_data)
        except Exception as e:
            # Continue processing other files even if one fails
            pass
    
    return results

def print_ultimate_summary(analyzer: UltimateMultiRepoAnalyzer, total_time: float):
    """Print ultimate analysis summary"""
    total_files = len(analyzer.files_data)
    
    print("\n" + "="*80)
    print("🎉 ULTIMATE HIGH-PERFORMANCE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"💾 RAM Used: {TOTAL_RAM_GB:.1f}GB total, {AVAILABLE_RAM_GB:.1f}GB available")
    print(f"⚡ CPU Cores: {CPU_CORES}, Workers: {MAX_WORKERS}")
    print(f"📊 Files Processed: {total_files:,}")
    print(f"🚀 Processing Speed: {total_files/total_time:.1f} files/second")
    
    # Repository breakdown
    print(f"\n📚 REPOSITORY BREAKDOWN:")
    for repo_name in REPOSITORIES.keys():
        repo_files = [f for f in analyzer.files_data.values() if f.repository == repo_name]
        if repo_files:
            total_loc = sum(f.lines_of_code for f in repo_files)
            total_funcs = sum(f.function_count for f in repo_files)
            avg_importance = sum(f.importance_score for f in repo_files) / len(repo_files)
            top_file = max(repo_files, key=lambda x: x.importance_score)
            
            print(f"  📦 {repo_name}:")
            print(f"     Files: {len(repo_files):,}")
            print(f"     Lines of Code: {total_loc:,}")
            print(f"     Functions: {total_funcs:,}")
            print(f"     Avg Importance: {avg_importance:.2f}")
            print(f"     Top File: {top_file.name} (importance: {top_file.importance_score:.1f})")
    
    # Call graph statistics
    if NETWORKX_AVAILABLE:
        total_nodes = sum(len(g.nodes) for g in analyzer.call_graphs.values())
        total_edges = sum(len(g.edges) for g in analyzer.call_graphs.values())
        print(f"\n🔗 CALL GRAPH STATISTICS:")
        print(f"   Total Nodes: {total_nodes:,}")
        print(f"   Total Edges: {total_edges:,}")
        for repo_name, graph in analyzer.call_graphs.items():
            print(f"   {repo_name}: {len(graph.nodes):,} nodes, {len(graph.edges):,} edges")
    
    # Generated files
    print(f"\n📁 GENERATED FILES:")
    print("  📊 aaps_ultimate_analysis.json - Comprehensive analysis data")
    print("  🌐 aaps_ultimate_overview.html - Multi-repository overview")
    
    for repo_name in REPOSITORIES.keys():
        print(f"  📈 aaps_{repo_name}_network.html - {repo_name} network visualization")
    
    print("  📋 aaps_ultimate_comparison.html - Repository comparison")
    
    if NEO4J_AVAILABLE:
        print("  🗄️  Neo4j database - High-performance graph database")
    
    # Performance metrics
    memory_final = psutil.virtual_memory()
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   Peak Memory Usage: {memory_final.percent:.1f}%")
    print(f"   Files per Core: {total_files/CPU_CORES:.1f}")
    print(f"   Efficiency: {(total_files*100)/(total_time*CPU_CORES):.1f} files/core/second")
    
    # Neo4j queries
    if NEO4J_AVAILABLE:
        print(f"\n🔍 EXAMPLE NEO4J QUERIES:")
        print("  MATCH (r:Repository) RETURN r.name, r.file_count ORDER BY r.file_count DESC")
        print("  MATCH (f:File {repository: 'EN_new'}) RETURN f.name, f.importance_score ORDER BY f.importance_score DESC LIMIT 10")
        print("  MATCH (f1:File)-[:CALLS]->(f2:File) WHERE f1.repository = 'AAPS_source' RETURN f1.name, f2.name LIMIT 10")
        print("  MATCH (f:File) WHERE f.importance_score > 50 RETURN f.name, f.repository, f.importance_score")
    
    print("\n💡 NEXT STEPS:")
    print("  🔍 Explore data: python neo4j_utilities.py")
    print("  🤖 Start RAG system: python ollama_neo4j_rag.py")
    print("  📊 Open visualizations in browser")
    print("="*80)

def main():
    """Main execution - ULTIMATE PERFORMANCE MODE"""
    print("🚀 AAPS ULTIMATE HIGH-PERFORMANCE MULTI-REPOSITORY ANALYZER")
    print("💪 MAXIMUM RAM AND CPU UTILIZATION MODE")
    print("="*80)
    print(f"🖥️  System: {TOTAL_RAM_GB:.1f}GB RAM, {CPU_CORES} CPU cores")
    print(f"⚡ Configuration: {MAX_WORKERS} workers, {CHUNK_SIZE} files/chunk")
    print(f"💾 Memory Target: {MAX_MEMORY_USAGE/(1024**3):.1f}GB ({(MAX_MEMORY_USAGE/psutil.virtual_memory().total)*100:.1f}%)")
    print("="*80)
    
    # Initialize ultimate analyzer
    analyzer = UltimateMultiRepoAnalyzer()
    
    try:
        # Run ultimate analysis
        start_time = time.time()
        
        success = analyzer.analyze_all_repositories_ultimate()
        
        total_time = time.time() - start_time
        
        if success:
            print_ultimate_summary(analyzer, total_time)
        else:
            print("\n❌ ULTIMATE ANALYSIS FAILED!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Ultimate analysis failed: {e}")
        print(f"\n❌ ANALYSIS FAILED: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
