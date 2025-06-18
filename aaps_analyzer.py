#!/usr/bin/env python3
"""
Enhanced AAPS Multi-Repository Analyzer
Complete source code storage and eating-now-focused importance scoring
Stores actual source code in Neo4j for better RAG performance
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

# Optional imports (same as original)
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
AVAILABLE_RAM_GB = psutil.virtual_memory().available / (1024**3)
CPU_CORES = psutil.cpu_count(logical=True)

# Enhanced performance settings
MAX_WORKERS = min(CPU_CORES, 64)  # Increased worker limit
CHUNK_SIZE = max(10, min(200, int(AVAILABLE_RAM_GB / 2)))
MAX_MEMORY_USAGE = int(AVAILABLE_RAM_GB * 0.9 * 1024**3)
NEO4J_BATCH_SIZE = min(10000, int(AVAILABLE_RAM_GB * 50))  # Reduced for stability

logger.info(f"🚀 ENHANCED PERFORMANCE MODE!")
logger.info(f"💾 Total RAM: {TOTAL_RAM_GB:.1f}GB, Available: {AVAILABLE_RAM_GB:.1f}GB")
logger.info(f"⚡ CPU Cores: {CPU_CORES}, Workers: {MAX_WORKERS}")
logger.info(f"📦 Chunk Size: {CHUNK_SIZE}, Batch Size: {NEO4J_BATCH_SIZE}")
logger.info(f"🔧 Strategy: Store ALL source code for enhanced RAG performance")

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
class EnhancedFileData:
    """Enhanced file data - stores ALL source code for RAG"""
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
    eating_now_score: float
    file_size: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    function_calls: List[Dict[str, str]]
    source_code: str = ""  # Store ALL source code
    key_snippets: Dict[str, str] = None
    has_source_code: bool = False
    is_eating_now_critical: bool = False

class MemoryManager:
    """Enhanced memory management"""
    
    def __init__(self):
        self.total_memory = psutil.virtual_memory().total
        self.target_usage = self.total_memory * 0.9
        
    def get_current_usage(self) -> float:
        return psutil.virtual_memory().percent
    
    def optimize_memory(self):
        if psutil.virtual_memory().percent > 95:
            gc.collect()
            logger.info(f"🧹 Memory cleanup: {psutil.virtual_memory().percent:.1f}% usage")

class UltimateRepositoryManager:
    """Enhanced repository management"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        
    def clone_all_repositories_parallel(self) -> bool:
        logger.info("🚀 Cloning all repositories in parallel...")
        
        def clone_single_repo(repo_item):
            repo_name, repo_config = repo_item
            return self._clone_repository_aggressive(repo_name, repo_config)
        
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
        """Enhanced cloning with better error handling"""
        repo_url = repo_config["url"]
        branch = repo_config["branch"]
        local_path = Path(repo_config["local_path"])
        
        logger.info(f"🔄 Cloning {repo_name}...")
        
        try:
            if local_path.exists():
                logger.info(f"📁 {repo_name} already exists, skipping")
                return True
            
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

class EnhancedFileAnalyzer:
    """Enhanced analyzer - stores ALL source code for better RAG"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self._compile_patterns()
        self._init_eating_now_patterns()
        
    def _compile_patterns(self):
        """Pattern compilation for analysis"""
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
            re.compile(r'(\w+)\.(\w+)\s*\('),
            re.compile(r'this\.(\w+)\s*\('),
            re.compile(r'super\.(\w+)\s*\('),
            re.compile(r'\b(\w+)\s*\(')
        ]
    
    def _init_eating_now_patterns(self):
        """Enhanced eating now scoring patterns"""
        self.eating_now_critical = {
            'eating': 100, 'eatnow': 100, 'eatingnow': 100, 'eat_now': 100,
            'bolus': 80, 'carb': 80, 'carbs': 80, 'carbohydrate': 80,
            'meal': 70, 'food': 70, 'nutrition': 70, 'insulin': 80,
            'dose': 60, 'dosing': 60, 'calculation': 50, 'calculator': 50,
            'treatment': 60, 'therapy': 50, 'algorithm': 60
        }
        
        self.eating_now_packages = {
            'eating': 150, 'eatnow': 150, 'eatingnow': 150,
            'bolus': 100, 'carb': 100, 'meal': 100, 'food': 100,
            'insulin': 90, 'dose': 80, 'calculation': 70, 'treatment': 80
        }
        
        self.eating_now_filenames = {
            'eating': 120, 'eatnow': 120, 'eatingnow': 120,
            'bolus': 100, 'carb': 100, 'meal': 90, 'food': 90,
            'insulin': 90, 'dose': 80, 'calc': 70, 'treatment': 80,
            'wizard': 80, 'assistant': 70, 'helper': 60
        }
    
    def analyze_file_enhanced(self, file_path: Path, repository: str) -> EnhancedFileData:
        """Enhanced analysis - stores ALL source code"""
        try:
            # Read file for analysis
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
            
            # Calculate eating now score
            eating_now_score = self._calculate_eating_now_score(
                file_path.name, package, content, functions, classes, repository
            )
            
            # Calculate importance score
            importance = self._calculate_importance_ultimate(
                file_path.name, package, loc, len(functions), len(classes), 
                repository, complexity, eating_now_score
            )
            
            # STORE ALL SOURCE CODE (no selective storage)
            stored_source_code = content
            has_source = len(content.strip()) > 0
            is_critical = eating_now_score > 100
            
            # Extract key snippets for large files
            key_snippets = {}
            if len(content) > 50000:  # For very large files, extract key snippets
                key_snippets = self._extract_key_snippets(content, functions, classes)
            
            return EnhancedFileData(
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
                eating_now_score=eating_now_score,
                file_size=file_size,
                functions=functions,
                classes=classes,
                imports=imports,
                function_calls=function_calls,
                source_code=stored_source_code,
                key_snippets=key_snippets,
                has_source_code=has_source,
                is_eating_now_critical=is_critical
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def _extract_key_snippets(self, content: str, functions: List[str], classes: List[str]) -> Dict[str, str]:
        """Extract key snippets for large files"""
        snippets = {}
        
        # Extract eating now related functions
        eating_functions = [f for f in functions if any(term in f.lower() for term in ['eating', 'bolus', 'carb', 'meal', 'insulin'])]
        
        for func_name in eating_functions[:5]:  # Max 5 functions
            func_pattern = re.compile(
                rf'(?:public|private|protected|static|\s)*\s+[\w<>\[\]]+\s+{re.escape(func_name)}\s*\([^)]*\).*?\{{.*?\}}',
                re.MULTILINE | re.DOTALL
            )
            match = func_pattern.search(content)
            if match:
                snippet = match.group(0)
                if len(snippet) > 2000:  # Limit snippet size
                    snippet = snippet[:2000] + "..."
                snippets[f'eating_function_{func_name}'] = snippet
        
        return snippets
    
    def _calculate_eating_now_score(self, filename: str, package: str, source_code: str, 
                                   functions: List[str], classes: List[str], repository: str) -> float:
        """Calculate eating now specific relevance score"""
        score = 0.0
        
        filename_lower = filename.lower()
        package_lower = package.lower()
        source_lower = source_code[:50000].lower() if len(source_code) > 50000 else source_code.lower()
        
        # Repository bonuses - EN repositories get massive boosts
        if repository in ['EN_new', 'EN_old']:
            score += 200
            if repository == 'EN_new':
                score += 50
        
        # Filename scoring
        for pattern, weight in self.eating_now_filenames.items():
            if pattern in filename_lower:
                score += weight * 2
        
        # Package scoring
        for pattern, weight in self.eating_now_packages.items():
            if pattern in package_lower:
                score += weight
        
        # Source code content scoring
        for pattern, weight in self.eating_now_critical.items():
            count = source_lower.count(pattern)
            if count > 0:
                score += min(weight * count * 0.1, weight)
        
        # Function and class name scoring
        all_names = functions + classes
        for name in all_names[:20]:  # Analyze more names
            name_lower = name.lower()
            for pattern, weight in self.eating_now_critical.items():
                if pattern in name_lower:
                    score += weight * 0.5
        
        # Special boost for core eating now functionality
        core_terms = ['eatnow', 'eatingnow', 'eating_now', 'eat_now']
        for term in core_terms:
            if term in filename_lower or term in package_lower:
                score += 300
        
        # Bolus and carb specific boosts
        if 'bolus' in source_lower or 'carb' in source_lower:
            score += 150
        
        return round(score, 2)
    
    def _analyze_java_ultimate(self, content: str, lines: List[str]) -> Tuple[List[str], List[str], List[str], List[Dict], float]:
        """Java analysis"""
        functions = [m.group(1) for m in self.java_method_pattern.finditer(content) 
                    if m.group(1) and m.group(1)[0].islower()]
        classes = [m.group(1) for m in self.java_class_pattern.finditer(content)]
        imports = [m.group(1) for m in self.java_import_pattern.finditer(content)]
        function_calls = self._extract_function_calls_ultimate(lines)
        
        complexity_keywords = ['if ', 'for ', 'while ', 'try ', 'catch ', 'switch ']
        complexity = sum(content.count(keyword) for keyword in complexity_keywords)
        
        return functions, classes, imports, function_calls, complexity
    
    def _analyze_kotlin_ultimate(self, content: str, lines: List[str]) -> Tuple[List[str], List[str], List[str], List[Dict], float]:
        """Kotlin analysis"""
        functions = [m.group(1) for m in self.kotlin_function_pattern.finditer(content) 
                    if m.group(1) and m.group(1)[0].islower()]
        classes = [m.group(1) for m in self.kotlin_class_pattern.finditer(content)]
        imports = [m.group(1) for m in self.kotlin_import_pattern.finditer(content)]
        function_calls = self._extract_function_calls_ultimate(lines)
        
        complexity_keywords = ['if ', 'for ', 'while ', 'when ', 'try ', 'catch ']
        complexity = sum(content.count(keyword) for keyword in complexity_keywords)
        
        return functions, classes, imports, function_calls, complexity
    
    def _extract_function_calls_ultimate(self, lines: List[str]) -> List[Dict[str, str]]:
        """Function call extraction"""
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
                                     func_count: int, class_count: int, repository: str, 
                                     complexity: float, eating_now_score: float) -> float:
        """Enhanced importance calculation with eating now priority"""
        score = 0.0
        
        # Base metrics
        score += min(20, loc / 50)
        score += min(25, func_count * 2)
        score += min(15, class_count * 4)
        score += min(10, complexity * 0.5)
        
        # MASSIVE eating now boost
        score += eating_now_score * 2
        
        # Name-based scoring
        name_lower = filename.lower()
        package_lower = package.lower()
        
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
        
        # Repository-specific bonuses with eating now priority
        repo_bonuses = {
            'EN_new': 25,        # Massive bonus for latest eating now
            'EN_old': 15,        # Good bonus for old eating now  
            'AAPS_source': 8     # Base source gets standard bonus
        }
        score += repo_bonuses.get(repository, 0)
        
        # Architecture importance
        if any(arch in package_lower for arch in ['core', 'main', 'base', 'foundation']):
            score += 15
        
        # Penalties
        if any(penalty in name_lower for penalty in ['test', 'mock', 'stub', 'example']):
            score *= 0.1
        if any(penalty in filename.lower() for penalty in ['generated', 'build', 'temp']):
            score *= 0.2
        
        return round(score, 2)

class EnhancedMultiRepoAnalyzer:
    """Enhanced analyzer with full source code storage"""
    
    def __init__(self):
        self.repositories = REPOSITORIES
        self.memory_manager = MemoryManager()
        self.repo_manager = UltimateRepositoryManager()
        self.file_analyzer = EnhancedFileAnalyzer()
        
        # In-memory storage
        self.files_data = {}
        self.call_graphs = {}
        self.function_mappings = defaultdict(lambda: defaultdict(set))
        
        if NETWORKX_AVAILABLE:
            for repo_name in self.repositories.keys():
                self.call_graphs[repo_name] = nx.DiGraph()
    
    def analyze_all_repositories_enhanced(self) -> bool:
        """Enhanced analysis with full source code storage"""
        start_time = time.time()
        
        logger.info(f"🚀 STARTING ENHANCED ANALYSIS WITH FULL SOURCE CODE STORAGE")
        logger.info(f"💪 Storing ALL source code for maximum RAG performance")
        logger.info(f"🧠 Enhanced eating now scoring and complete file indexing")
        
        # Step 1: Clone repositories
        if not self.repo_manager.clone_all_repositories_parallel():
            logger.error("❌ Failed to setup repositories")
            return False
        
        # Step 2: File discovery
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
        
        # Step 3: Enhanced parallel processing with ALL source code storage
        logger.info(f"🔥 Processing with enhanced method + full source code storage...")
        
        all_work_items = []
        for repo_name, files in all_repo_files.items():
            for file_path in files:
                all_work_items.append((file_path, repo_name))
        
        # Process in chunks
        chunks = [all_work_items[i:i+CHUNK_SIZE] for i in range(0, len(all_work_items), CHUNK_SIZE)]
        
        processed_files = 0
        # Enhanced parallel processing
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(analyze_file_batch_enhanced, chunk): i 
                      for i, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    
                    for file_data in batch_results:
                        if file_data:
                            key = f"{file_data.repository}:{file_data.path}"
                            self.files_data[key] = file_data
                            processed_files += 1
                    
                    progress = (processed_files / total_files) * 100
                    memory_usage = psutil.virtual_memory().percent
                    logger.info(f"🔥 Progress: {progress:.1f}% ({processed_files}/{total_files}) | Memory: {memory_usage:.1f}%")
                    
                    # Memory optimization
                    self.memory_manager.optimize_memory()
                    
                except Exception as e:
                    logger.error(f"❌ Batch processing failed: {e}")
        
        analysis_time = time.time() - start_time
        logger.info(f"✅ Enhanced analysis completed in {analysis_time:.2f} seconds")
        logger.info(f"📊 Successfully processed {len(self.files_data)} files")
        
        # Step 4: Build mappings
        self._build_ultimate_mappings()
        
        # Step 5: Generate outputs
        self._generate_enhanced_outputs()
        
        return True
    
    def _find_source_files_ultimate(self, repo_path: Path) -> List[Path]:
        """Source file discovery"""
        source_files = []
        skip_dirs = {'.git', 'build', 'target', '.gradle', 'node_modules', '.idea', 'bin', 'out'}
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith(('.java', '.kt')):
                    source_files.append(Path(root) / file)
        
        return source_files
    
    def _build_ultimate_mappings(self):
        """Build function mappings and call graphs"""
        logger.info("🔗 Building function mappings and call graphs...")
        
        for file_key, file_data in self.files_data.items():
            repo_name = file_data.repository
            for function_name in file_data.functions:
                self.function_mappings[repo_name][function_name].add(file_data.name)
        
        if NETWORKX_AVAILABLE:
            for repo_name in self.repositories.keys():
                self._build_call_graph_for_repo(repo_name)
    
    def _build_call_graph_for_repo(self, repo_name: str):
        """Build call graph for repository"""
        repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
        call_graph = self.call_graphs[repo_name]
        
        for file_data in repo_files:
            call_graph.add_node(
                file_data.name,
                path=file_data.path,
                importance=file_data.importance_score,
                eating_now_score=file_data.eating_now_score,
                complexity=file_data.complexity_score,
                loc=file_data.lines_of_code
            )
        
        call_counts = defaultdict(int)
        
        for file_data in repo_files:
            for call in file_data.function_calls:
                function_name = call['function']
                target_files = self.function_mappings[repo_name].get(function_name, set())
                
                for target_file in target_files:
                    if target_file != file_data.name:
                        call_counts[(file_data.name, target_file)] += 1
        
        for (source, target), weight in call_counts.items():
            if weight > 0:
                call_graph.add_edge(source, target, weight=weight)
        
        logger.info(f"📊 {repo_name} call graph: {len(call_graph.nodes)} nodes, {len(call_graph.edges)} edges")
    
    def _generate_enhanced_outputs(self):
        """Generate enhanced outputs"""
        logger.info("📈 Generating enhanced outputs...")
        
        try:
            self._generate_enhanced_json_report()
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
        
        try:
            if NEO4J_AVAILABLE:
                self._populate_neo4j_enhanced()
        except Exception as e:
            logger.warning(f"Neo4j population failed: {e}")
        
        try:
            if PLOTLY_AVAILABLE:
                self._generate_eating_now_visualizations()
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        logger.info("🎉 Enhanced outputs generated!")
    
    def _generate_enhanced_json_report(self):
        """Generate enhanced JSON report with full source code"""
        logger.info("📄 Generating enhanced JSON report...")
        
        # Count files with source code stored
        files_with_source = len([f for f in self.files_data.values() if f.has_source_code])
        
        report = {
            'analysis_metadata': {
                'timestamp': time.time(),
                'total_ram_gb': TOTAL_RAM_GB,
                'available_ram_gb': AVAILABLE_RAM_GB,
                'cpu_cores': CPU_CORES,
                'max_workers': MAX_WORKERS,
                'chunk_size': CHUNK_SIZE,
                'repositories_analyzed': len(self.repositories),
                'total_files_processed': len(self.files_data),
                'files_with_source_code': files_with_source,
                'storage_strategy': 'full_source_code_storage',
                'enhanced_features': [
                    'eating_now_prioritization',
                    'full_source_code_storage',
                    'enhanced_performance_processing',
                    'comprehensive_importance_scoring'
                ]
            },
            'repository_summaries': {},
            'global_summary': {
                'total_files': len(self.files_data),
                'total_loc': sum(f.lines_of_code for f in self.files_data.values()),
                'total_functions': sum(f.function_count for f in self.files_data.values()),
                'total_classes': sum(f.class_count for f in self.files_data.values()),
                'avg_eating_now_score': sum(f.eating_now_score for f in self.files_data.values()) / len(self.files_data) if self.files_data else 0,
                'critical_eating_now_files': len([f for f in self.files_data.values() if f.is_eating_now_critical])
            },
            'top_eating_now_files': [],
            'top_files_global': []
        }
        
        # Per-repository summaries with eating now focus
        for repo_name in self.repositories.keys():
            repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
            
            if repo_files:
                total_loc = sum(f.lines_of_code for f in repo_files)
                total_functions = sum(f.function_count for f in repo_files)
                total_classes = sum(f.class_count for f in repo_files)
                avg_importance = sum(f.importance_score for f in repo_files) / len(repo_files)
                avg_eating_now = sum(f.eating_now_score for f in repo_files) / len(repo_files)
                
                # Top files by eating now relevance
                top_eating_now = sorted(repo_files, key=lambda x: x.eating_now_score, reverse=True)[:10]
                top_files = sorted(repo_files, key=lambda x: x.importance_score, reverse=True)[:15]
                
                report['repository_summaries'][repo_name] = {
                    'file_count': len(repo_files),
                    'total_loc': total_loc,
                    'total_functions': total_functions,
                    'total_classes': total_classes,
                    'avg_importance': round(avg_importance, 2),
                    'avg_eating_now_score': round(avg_eating_now, 2),
                    'files_with_source': len([f for f in repo_files if f.has_source_code]),
                    'top_eating_now_files': [
                        {
                            'name': f.name,
                            'eating_now_score': f.eating_now_score,
                            'importance': f.importance_score,
                            'loc': f.lines_of_code,
                            'functions': f.function_count,
                            'package': f.package,
                            'has_source_code': f.has_source_code,
                            'key_snippets': len(f.key_snippets) if f.key_snippets else 0
                        }
                        for f in top_eating_now
                    ],
                    'top_files': [
                        {
                            'name': f.name,
                            'importance': f.importance_score,
                            'eating_now_score': f.eating_now_score,
                            'loc': f.lines_of_code,
                            'functions': f.function_count,
                            'package': f.package
                        }
                        for f in top_files
                    ]
                }
        
        # Global top eating now files (most critical for plugin development)
        all_files = list(self.files_data.values())
        top_eating_now_global = sorted(all_files, key=lambda x: x.eating_now_score, reverse=True)[:25]
        
        report['top_eating_now_files'] = [
            {
                'name': f.name,
                'repository': f.repository,
                'eating_now_score': f.eating_now_score,
                'importance': f.importance_score,
                'loc': f.lines_of_code,
                'functions': f.function_count,
                'package': f.package,
                'has_source_code': f.has_source_code,
                'source_preview': f.source_code[:300] + "..." if len(f.source_code) > 300 else f.source_code,
                'key_snippets': f.key_snippets if f.key_snippets else {}
            }
            for f in top_eating_now_global
        ]
        
        # Global top files by importance
        top_global = sorted(all_files, key=lambda x: x.importance_score, reverse=True)[:30]
        report['top_files_global'] = [
            {
                'name': f.name,
                'repository': f.repository,
                'importance': f.importance_score,
                'eating_now_score': f.eating_now_score,
                'loc': f.lines_of_code,
                'functions': f.function_count,
                'package': f.package
            }
            for f in top_global
        ]
        
        # Save report
        with open('aaps_enhanced_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("✅ Saved: aaps_enhanced_analysis.json")
        logger.info(f"📊 Source code stored for {files_with_source} files")
    
    def _populate_neo4j_enhanced(self):
        """Populate Neo4j with enhanced data including full source code"""
        logger.info("🗄️ Populating Neo4j with enhanced approach...")
        
        try:
            driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "password"),
                max_connection_lifetime=3600,
                max_connection_pool_size=100
            )
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        
        try:
            with driver.session() as session:
                # Clear database
                session.run("MATCH (n) DETACH DELETE n")
                
                # Create repository nodes with enhanced properties
                repo_data = []
                for repo_name in self.repositories.keys():
                    repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
                    if repo_files:
                        avg_eating_now = sum(f.eating_now_score for f in repo_files) / len(repo_files)
                        files_with_source = len([f for f in repo_files if f.has_source_code])
                        
                        repo_data.append({
                            'name': repo_name,
                            'file_count': len(repo_files),
                            'total_loc': sum(f.lines_of_code for f in repo_files),
                            'total_functions': sum(f.function_count for f in repo_files),
                            'avg_eating_now_score': round(avg_eating_now, 2),
                            'is_eating_now_repo': repo_name in ['EN_new', 'EN_old'],
                            'files_with_source_code': files_with_source
                        })
                
                if repo_data:
                    session.run("""
                        UNWIND $repos AS repo
                        CREATE (r:Repository {
                            name: repo.name,
                            file_count: repo.file_count,
                            total_loc: repo.total_loc,
                            total_functions: repo.total_functions,
                            avg_eating_now_score: repo.avg_eating_now_score,
                            is_eating_now_repo: repo.is_eating_now_repo,
                            files_with_source_code: repo.files_with_source_code
                        })
                    """, repos=repo_data)
                
                # Create file nodes with FULL source code storage
                file_batches = []
                current_batch = []
                
                for file_data in self.files_data.values():
                    # Store ALL source code in Neo4j
                    source_for_neo4j = file_data.source_code
                    # Limit very large files to prevent Neo4j issues
                    if len(source_for_neo4j) > 100000:  # 100KB limit per file
                        source_for_neo4j = source_for_neo4j[:100000] + "\n\n[... file truncated for Neo4j storage ...]"
                    
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
                        'eating_now_score': file_data.eating_now_score,
                        'complexity_score': file_data.complexity_score,
                        'source_code': source_for_neo4j,
                        'functions': file_data.functions[:50],  # Limit array size
                        'classes': file_data.classes[:20],
                        'imports': file_data.imports[:100],
                        'has_source_code': file_data.has_source_code,
                        'is_eating_now_critical': file_data.is_eating_now_critical,
                        'key_snippets': json.dumps(file_data.key_snippets) if file_data.key_snippets else "{}"
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
                            eating_now_score: file.eating_now_score,
                            complexity_score: file.complexity_score,
                            source_code: file.source_code,
                            functions: file.functions,
                            classes: file.classes,
                            imports: file.imports,
                            has_source_code: file.has_source_code,
                            is_eating_now_critical: file.is_eating_now_critical,
                            key_snippets: file.key_snippets
                        })
                        CREATE (r)-[:CONTAINS]->(f)
                    """, batch=batch)
                    
                    logger.info(f"📊 Neo4j: Processed {(i+1)*len(batch)} files...")
                
                # Create call relationships
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
                
                # Create indexes
                indexes = [
                    "CREATE INDEX file_repo_idx IF NOT EXISTS FOR (f:File) ON (f.repository)",
                    "CREATE INDEX file_importance_idx IF NOT EXISTS FOR (f:File) ON (f.importance_score)",
                    "CREATE INDEX file_eating_now_idx IF NOT EXISTS FOR (f:File) ON (f.eating_now_score)",
                    "CREATE INDEX file_package_idx IF NOT EXISTS FOR (f:File) ON (f.package)",
                    "CREATE INDEX repo_name_idx IF NOT EXISTS FOR (r:Repository) ON (r.name)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except:
                        pass
                
                # Create full-text index for source code
                try:
                    session.run("CREATE FULLTEXT INDEX file_source_idx IF NOT EXISTS FOR (f:File) ON f.source_code")
                    logger.info("✅ Created full-text index for source code")
                except Exception as e:
                    logger.warning(f"Could not create full-text index: {e}")
                
                # Verify population
                result = session.run("MATCH (f:File) RETURN count(f) as files")
                file_count = result.single()["files"]
                
                result = session.run("MATCH (f:File) WHERE f.has_source_code = true RETURN count(f) as files_with_source")
                files_with_source = result.single()["files_with_source"]
                
                result = session.run("MATCH ()-[c:CALLS]->() RETURN count(c) as calls")
                calls_count = result.single()["calls"]
                
                logger.info(f"✅ Enhanced Neo4j populated: {file_count} files, {files_with_source} with source code, {calls_count} relationships")
        
        finally:
            driver.close()
    
    def _generate_eating_now_visualizations(self):
        """Generate eating now focused visualizations"""
        logger.info("📊 Generating eating now visualizations...")
        
        # 1. Multi-repository overview with eating now metrics
        repo_data = []
        for repo_name in self.repositories.keys():
            repo_files = [f for f in self.files_data.values() if f.repository == repo_name]
            if repo_files:
                avg_eating_now = sum(f.eating_now_score for f in repo_files) / len(repo_files)
                files_with_source = len([f for f in repo_files if f.has_source_code])
                
                repo_data.append({
                    'repository': repo_name,
                    'files': len(repo_files),
                    'loc': sum(f.lines_of_code for f in repo_files),
                    'functions': sum(f.function_count for f in repo_files),
                    'avg_importance': sum(f.importance_score for f in repo_files) / len(repo_files),
                    'avg_eating_now': avg_eating_now,
                    'eating_now_files': len([f for f in repo_files if f.eating_now_score > 50]),
                    'files_with_source': files_with_source
                })
        
        if repo_data:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Eating Now Scores by Repository', 'Files with Source Code', 'Eating Now Files (>50)', 'Average Importance'),
                specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
            )
            
            repos = [d['repository'] for d in repo_data]
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
            
            fig.add_trace(go.Bar(x=repos, y=[d['avg_eating_now'] for d in repo_data], 
                               name='Avg Eating Now', marker_color=colors), row=1, col=1)
            fig.add_trace(go.Bar(x=repos, y=[d['files_with_source'] for d in repo_data], 
                               name='Files with Source', marker_color=colors), row=1, col=2)
            fig.add_trace(go.Bar(x=repos, y=[d['eating_now_files'] for d in repo_data], 
                               name='Eating Now Files', marker_color=colors), row=2, col=1)
            fig.add_trace(go.Bar(x=repos, y=[d['avg_importance'] for d in repo_data], 
                               name='Avg Importance', marker_color=colors), row=2, col=2)
            
            fig.update_layout(
                title='AAPS Enhanced Analysis - Eating Now Focus with Full Source Code',
                height=800,
                showlegend=False
            )
            
            pyo.plot(fig, filename='aaps_enhanced_overview.html', auto_open=False)
            logger.info("✅ Created: aaps_enhanced_overview.html")
        
        # 2. Eating now files analysis
        all_files = list(self.files_data.values())
        eating_now_files = [f for f in all_files if f.eating_now_score > 50]
        
        if eating_now_files:
            eating_now_files.sort(key=lambda x: x.eating_now_score, reverse=True)
            top_20 = eating_now_files[:20]
            
            # Create color mapping for repositories
            repo_colors = {'EN_new': '#ff6b6b', 'EN_old': '#4ecdc4', 'AAPS_source': '#45b7d1'}
            colors = [repo_colors.get(f.repository, '#gray') for f in top_20]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[f.importance_score for f in top_20],
                y=[f.eating_now_score for f in top_20],
                mode='markers+text',
                text=[f.name[:15] + "..." if len(f.name) > 15 else f.name for f in top_20],
                textposition="top center",
                marker=dict(
                    size=[15 + (10 if f.has_source_code else 0) for f in top_20],
                    color=colors,
                    symbol=['circle' if not f.has_source_code else 'diamond' for f in top_20],
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                             'Repository: %{marker.color}<br>' +
                             'Importance: %{x}<br>' +
                             'Eating Now Score: %{y}<br>' +
                             'Has Source: %{marker.symbol}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title='Top 20 Eating Now Files - Enhanced Analysis (Diamonds = Full Source Code)',
                xaxis_title='Importance Score',
                yaxis_title='Eating Now Relevance Score',
                width=1200,
                height=800
            )
            
            pyo.plot(fig, filename='aaps_enhanced_eating_now.html', auto_open=False)
            logger.info("✅ Created: aaps_enhanced_eating_now.html")

# CRITICAL: Global function for multiprocessing (must be at module level)
def analyze_file_batch_enhanced(work_items: List[Tuple[Path, str]]) -> List[EnhancedFileData]:
    """Enhanced file batch analysis for multiprocessing"""
    results = []
    analyzer = EnhancedFileAnalyzer()
    
    for file_path, repository in work_items:
        try:
            file_data = analyzer.analyze_file_enhanced(file_path, repository)
            if file_data:
                results.append(file_data)
        except Exception as e:
            # Continue processing other files even if one fails
            pass
    
    return results

def print_enhanced_summary(analyzer: EnhancedMultiRepoAnalyzer, total_time: float):
    """Print enhanced analysis summary"""
    total_files = len(analyzer.files_data)
    eating_now_files = [f for f in analyzer.files_data.values() if f.eating_now_score > 50]
    files_with_source = [f for f in analyzer.files_data.values() if f.has_source_code]
    
    print("\n" + "="*80)
    print("🎉 ENHANCED ANALYSIS COMPLETE!")
    print("🧠 FULL SOURCE CODE STORAGE + EATING NOW PRIORITIZATION")
    print("="*80)
    print(f"⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"💾 RAM Used: {TOTAL_RAM_GB:.1f}GB total, {AVAILABLE_RAM_GB:.1f}GB available")
    print(f"⚡ CPU Cores: {CPU_CORES}, Workers: {MAX_WORKERS}")
    print(f"📊 Files Processed: {total_files:,}")
    print(f"🍽️ Eating Now Relevant Files: {len(eating_now_files):,}")
    print(f"💾 Files with Source Code Stored: {len(files_with_source):,}")
    print(f"🚀 Processing Speed: {total_files/total_time:.1f} files/second")
    print(f"🧠 Storage Strategy: Full source code for ALL files")
    
    # Repository breakdown with eating now focus
    print(f"\n📚 REPOSITORY BREAKDOWN (EATING NOW FOCUSED):")
    for repo_name in REPOSITORIES.keys():
        repo_files = [f for f in analyzer.files_data.values() if f.repository == repo_name]
        if repo_files:
            total_loc = sum(f.lines_of_code for f in repo_files)
            total_funcs = sum(f.function_count for f in repo_files)
            avg_importance = sum(f.importance_score for f in repo_files) / len(repo_files)
            avg_eating_now = sum(f.eating_now_score for f in repo_files) / len(repo_files)
            eating_now_count = len([f for f in repo_files if f.eating_now_score > 50])
            files_with_source_count = len([f for f in repo_files if f.has_source_code])
            top_file = max(repo_files, key=lambda x: x.eating_now_score)
            
            print(f"  📦 {repo_name}:")
            print(f"     Files: {len(repo_files):,}")
            print(f"     Lines of Code: {total_loc:,}")
            print(f"     Functions: {total_funcs:,}")
            print(f"     Avg Importance: {avg_importance:.2f}")
            print(f"     🍽️ Avg Eating Now Score: {avg_eating_now:.2f}")
            print(f"     🍽️ Eating Now Files: {eating_now_count}")
            print(f"     💾 Files with Source Code: {files_with_source_count}")
            print(f"     🍽️ Top Eating Now File: {top_file.name} (score: {top_file.eating_now_score:.1f})")
    
    # Top eating now files globally
    all_files = list(analyzer.files_data.values())
    top_eating_now = sorted(all_files, key=lambda x: x.eating_now_score, reverse=True)[:8]
    
    print(f"\n🍽️ TOP EATING NOW FILES (CRITICAL FOR PLUGIN DEVELOPMENT):")
    for i, file_data in enumerate(top_eating_now, 1):
        has_source = file_data.has_source_code
        source_indicator = "💾" if has_source else "  "
        print(f"   {i:2d}.{source_indicator} {file_data.name} ({file_data.repository})")
        print(f"       Eating Now Score: {file_data.eating_now_score:.1f}")
        print(f"       Importance: {file_data.importance_score:.1f}")
        print(f"       Package: {file_data.package}")
        print(f"       Has Source Code: {'Yes' if has_source else 'No'}")
        if file_data.key_snippets:
            print(f"       Key Snippets: {len(file_data.key_snippets)}")
    
    # Memory efficiency stats
    total_source_chars = sum(len(f.source_code) for f in analyzer.files_data.values() if f.source_code)
    
    print(f"\n💾 ENHANCED STORAGE STATISTICS:")
    print(f"   Total Source Code Stored: {total_source_chars:,} characters")
    print(f"   Average Source per File: {total_source_chars/len(files_with_source):,.0f} chars" if files_with_source else "   No source code stored")
    print(f"   Storage Coverage: {len(files_with_source)}/{total_files} files ({(len(files_with_source)/total_files)*100:.1f}%)")
    print(f"   Full Text Search: Enabled via Neo4j full-text index")
    
    # Generated files
    print(f"\n📁 GENERATED FILES:")
    print("  📊 aaps_enhanced_analysis.json - Complete enhanced report with full source")
    print("  🌐 aaps_enhanced_overview.html - Enhanced overview")
    print("  🍽️ aaps_enhanced_eating_now.html - Eating now analysis")
    
    if NEO4J_AVAILABLE:
        print("  🗄️  Enhanced Neo4j database - With full source code storage and indexes")
    
    print(f"\n💡 NEXT STEPS FOR EATING NOW PLUGIN DEVELOPMENT:")
    print("  🔍 Explore data: python neo4j_utilities.py")
    print("  🤖 Start RAG: python ollama_neo4j_rag.py")
    print("  📊 Open visualizations")
    print("  🍽️ All files now have full source code access")
    print("  💾 Use full-text search for code exploration")
    print("  🧠 Enhanced RAG with complete source code context")
    print("="*80)

def main():
    """Main execution - ENHANCED MODE"""
    print("🚀 AAPS ENHANCED MULTI-REPOSITORY ANALYZER")
    print("🧠 FULL SOURCE CODE STORAGE + EATING NOW PRIORITIZATION")
    print("💾 COMPLETE FILE INDEXING FOR MAXIMUM RAG PERFORMANCE")
    print("="*80)
    print(f"🖥️  System: {TOTAL_RAM_GB:.1f}GB RAM, {CPU_CORES} CPU cores")
    print(f"⚡ Configuration: {MAX_WORKERS} workers, {CHUNK_SIZE} files/chunk")
    print(f"💾 Memory Target: {MAX_MEMORY_USAGE/(1024**3):.1f}GB ({(MAX_MEMORY_USAGE/psutil.virtual_memory().total)*100:.1f}%)")
    print(f"🧠 Storage Strategy: Full source code for ALL files")
    print(f"🍽️ Enhanced Features: Eating now scoring, full source storage, enhanced indexing")
    print("="*80)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedMultiRepoAnalyzer()
    
    try:
        # Run enhanced analysis
        start_time = time.time()
        
        success = analyzer.analyze_all_repositories_enhanced()
        
        total_time = time.time() - start_time
        
        if success:
            print_enhanced_summary(analyzer, total_time)
        else:
            print("\n❌ ENHANCED ANALYSIS FAILED!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        print(f"\n❌ ANALYSIS FAILED: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
