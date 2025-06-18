#!/usr/bin/env python3
"""
Docker Neo4j High-Performance Setup
Automated setup for Neo4j in Docker with 384GB RAM optimization
"""

import os
import subprocess
import logging
import time
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DockerNeo4jSetup:
    def __init__(self, container_name="neo4j-aaps", password="password"):
        self.container_name = container_name
        self.password = password
        self.neo4j_port = 7474
        self.bolt_port = 7687
        
    def stop_existing_container(self):
        """Stop and remove existing Neo4j container"""
        try:
            logger.info(f"Stopping existing container: {self.container_name}")
            subprocess.run(["docker", "stop", self.container_name], 
                         capture_output=True, check=False)
            subprocess.run(["docker", "rm", self.container_name], 
                         capture_output=True, check=False)
            logger.info("✅ Existing container stopped and removed")
        except Exception as e:
            logger.info(f"No existing container to stop: {e}")
    
    def create_config_directory(self):
        """Create local configuration directory"""
        config_dir = Path("./neo4j_config")
        data_dir = Path("./neo4j_data")
        logs_dir = Path("./neo4j_logs")
        
        config_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Created configuration directories")
        return config_dir, data_dir, logs_dir
    
    def generate_optimized_config(self, config_dir):
        """Generate optimized Neo4j configuration for Docker"""
        config_content = """
# Neo4j Configuration Optimized for 384GB RAM in Docker
# This configuration allocates substantial memory for high-performance

# Memory Settings (Aggressive for 384GB system)
server.memory.heap.initial_size=32g
server.memory.heap.max_size=64g
server.memory.pagecache.size=200g
server.memory.transaction.total.max=100g

# Network Settings
server.default_listen_address=0.0.0.0
server.bolt.listen_address=:7687
server.http.listen_address=:7474

# Transaction Settings
db.transaction.timeout=600s
db.transaction.bookmark_ready_timeout=30s
db.transaction.concurrent.maximum=1000

# Query Settings
dbms.query.cache_size=10000
cypher.runtime=slotted
cypher.lenient_create_relationship=true

# Performance Settings
db.checkpoint.interval.time=300s
db.checkpoint.interval.tx=1000000

# Index Settings
db.index.fulltext.max_clause_count=100000
db.index.default_schema_provider=native-btree-1.0

# Security
dbms.security.auth_enabled=true
dbms.security.procedures.unrestricted=*
dbms.security.procedures.allowlist=*

# Logging (reduced for performance)
server.logs.query.enabled=false
server.logs.query.threshold=5s
server.logs.debug.level=WARN

# JVM Additional Settings will be passed via Docker environment
"""
        
        config_file = config_dir / "neo4j.conf"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"✅ Generated optimized config: {config_file}")
        return config_file
    
    def create_docker_compose(self):
        """Create Docker Compose file for easy management"""
        compose_content = f"""
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15-enterprise
    container_name: {self.container_name}
    restart: unless-stopped
    ports:
      - "{self.bolt_port}:{self.bolt_port}"
      - "{self.neo4j_port}:{self.neo4j_port}"
    volumes:
      - ./neo4j_data:/data
      - ./neo4j_logs:/logs
      - ./neo4j_config:/var/lib/neo4j/conf
    environment:
      # Authentication
      - NEO4J_AUTH=neo4j/{self.password}
      
      # Memory Settings (matching our config)
      - NEO4J_server_memory_heap_initial__size=32g
      - NEO4J_server_memory_heap_max__size=64g
      - NEO4J_server_memory_pagecache_size=200g
      - NEO4J_server_memory_transaction_total_max=100g
      
      # JVM Tuning for Large Heap
      - NEO4J_server_jvm_additional=-XX:+UseG1GC
      - NEO4J_server_jvm_additional=-XX:+UnlockExperimentalVMOptions
      - NEO4J_server_jvm_additional=-XX:G1HeapRegionSize=32m
      - NEO4J_server_jvm_additional=-XX:MaxGCPauseMillis=200
      - NEO4J_server_jvm_additional=-XX:+ParallelRefProcEnabled
      - NEO4J_server_jvm_additional=-XX:+UseStringDeduplication
      - NEO4J_server_jvm_additional=-XX:+DisableExplicitGC
      
      # Performance Settings
      - NEO4J_db_transaction_timeout=600s
      - NEO4J_dbms_query_cache__size=10000
      - NEO4J_cypher_runtime=slotted
      
      # Accept License (for Enterprise features)
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      
      # Disable authentication for development (optional)
      # - NEO4J_AUTH=none
    
    # Resource limits for Docker
    deploy:
      resources:
        limits:
          memory: 320g  # Leave some RAM for the OS
        reservations:
          memory: 300g
    
    # Health check
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p {self.password} 'RETURN 1'"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
"""
        
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)
        
        logger.info("✅ Generated docker-compose.yml")
    
    def create_docker_run_script(self):
        """Create Docker run script as alternative to compose"""
        script_content = f"""#!/bin/bash
# High-Performance Neo4j Docker Run Script

# Stop existing container
docker stop {self.container_name} 2>/dev/null || true
docker rm {self.container_name} 2>/dev/null || true

# Run Neo4j with high-performance settings
docker run -d \\
  --name {self.container_name} \\
  --restart unless-stopped \\
  -p {self.bolt_port}:{self.bolt_port} \\
  -p {self.neo4j_port}:{self.neo4j_port} \\
  -v $(pwd)/neo4j_data:/data \\
  -v $(pwd)/neo4j_logs:/logs \\
  -v $(pwd)/neo4j_config:/var/lib/neo4j/conf \\
  --memory=320g \\
  --memory-reservation=300g \\
  --cpus=96 \\
  -e NEO4J_AUTH=neo4j/{self.password} \\
  -e NEO4J_server_memory_heap_initial__size=32g \\
  -e NEO4J_server_memory_heap_max__size=64g \\
  -e NEO4J_server_memory_pagecache_size=200g \\
  -e NEO4J_server_memory_transaction_total_max=100g \\
  -e NEO4J_server_jvm_additional=-XX:+UseG1GC \\
  -e NEO4J_server_jvm_additional=-XX:G1HeapRegionSize=32m \\
  -e NEO4J_server_jvm_additional=-XX:MaxGCPauseMillis=200 \\
  -e NEO4J_server_jvm_additional=-XX:+ParallelRefProcEnabled \\
  -e NEO4J_server_jvm_additional=-XX:+UseStringDeduplication \\
  -e NEO4J_db_transaction_timeout=600s \\
  -e NEO4J_dbms_query_cache__size=10000 \\
  -e NEO4J_cypher_runtime=slotted \\
  -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \\
  neo4j:5.15-enterprise

echo "🚀 Neo4j container starting with high-performance settings..."
echo "📊 Memory allocation: 320GB"
echo "🔧 CPUs: 96 cores"
echo "🌐 Web interface: http://localhost:{self.neo4j_port}"
echo "⚡ Bolt: bolt://localhost:{self.bolt_port}"
echo ""
echo "Waiting for Neo4j to start..."

# Wait for Neo4j to be ready
until docker exec {self.container_name} cypher-shell -u neo4j -p {self.password} "RETURN 1" 2>/dev/null; do
  echo "⏳ Waiting for Neo4j..."
  sleep 5
done

echo "✅ Neo4j is ready!"
"""
        
        with open('start_neo4j_hp.sh', 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod('start_neo4j_hp.sh', 0o755)
        
        logger.info("✅ Generated start_neo4j_hp.sh")
    
    def start_container_compose(self):
        """Start Neo4j using Docker Compose"""
        try:
            logger.info("Starting Neo4j with Docker Compose...")
            result = subprocess.run(["docker-compose", "up", "-d"], 
                                  capture_output=True, text=True, check=True)
            logger.info("✅ Neo4j container started with Docker Compose")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start with Docker Compose: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.warning("Docker Compose not found, will use docker run instead")
            return False
    
    def start_container_direct(self):
        """Start Neo4j using direct docker run command"""
        try:
            logger.info("Starting Neo4j with docker run...")
            
            # Stop existing container
            subprocess.run(["docker", "stop", self.container_name], 
                         capture_output=True, check=False)
            subprocess.run(["docker", "rm", self.container_name], 
                         capture_output=True, check=False)
            
            # Docker run command with high-performance settings
            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--restart", "unless-stopped",
                "-p", f"{self.bolt_port}:{self.bolt_port}",
                "-p", f"{self.neo4j_port}:{self.neo4j_port}",
                "-v", f"{os.getcwd()}/neo4j_data:/data",
                "-v", f"{os.getcwd()}/neo4j_logs:/logs", 
                "-v", f"{os.getcwd()}/neo4j_config:/var/lib/neo4j/conf",
                "--memory=320g",
                "--memory-reservation=300g",
                "--cpus=96",
                "-e", f"NEO4J_AUTH=neo4j/{self.password}",
                "-e", "NEO4J_server_memory_heap_initial__size=32g",
                "-e", "NEO4J_server_memory_heap_max__size=64g", 
                "-e", "NEO4J_server_memory_pagecache_size=200g",
                "-e", "NEO4J_server_memory_transaction_total_max=100g",
                "-e", "NEO4J_server_jvm_additional=-XX:+UseG1GC",
                "-e", "NEO4J_server_jvm_additional=-XX:G1HeapRegionSize=32m",
                "-e", "NEO4J_server_jvm_additional=-XX:MaxGCPauseMillis=200",
                "-e", "NEO4J_server_jvm_additional=-XX:+ParallelRefProcEnabled",
                "-e", "NEO4J_server_jvm_additional=-XX:+UseStringDeduplication",
                "-e", "NEO4J_db_transaction_timeout=600s",
                "-e", "NEO4J_dbms_query_cache__size=10000",
                "-e", "NEO4J_cypher_runtime=slotted",
                "-e", "NEO4J_ACCEPT_LICENSE_AGREEMENT=yes",
                "neo4j:5.15-enterprise"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("✅ Neo4j container started with docker run")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start with docker run: {e.stderr}")
            return False
    
    def wait_for_neo4j(self, timeout=120):
        """Wait for Neo4j to be ready"""
        logger.info("Waiting for Neo4j to be ready...")
        
        for i in range(timeout):
            try:
                result = subprocess.run([
                    "docker", "exec", self.container_name,
                    "cypher-shell", "-u", "neo4j", "-p", self.password,
                    "RETURN 1"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    logger.info("✅ Neo4j is ready!")
                    return True
                    
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                pass
            
            if i % 10 == 0:
                logger.info(f"⏳ Still waiting... ({i}/{timeout}s)")
            time.sleep(1)
        
        logger.error("❌ Neo4j failed to start within timeout")
        return False
    
    def apply_runtime_optimizations(self):
        """Apply runtime optimizations via cypher-shell"""
        try:
            logger.info("Applying runtime optimizations...")
            
            optimizations = [
                "CALL dbms.setConfigValue('db.transaction.timeout', '600s')",
                "CALL dbms.setConfigValue('dbms.query.cache_size', '10000')",
                "CALL dbms.setConfigValue('cypher.runtime', 'slotted')",
            ]
            
            for opt in optimizations:
                try:
                    subprocess.run([
                        "docker", "exec", self.container_name,
                        "cypher-shell", "-u", "neo4j", "-p", self.password,
                        opt
                    ], capture_output=True, text=True, check=True)
                    logger.info(f"✅ Applied: {opt}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"⚠️ Could not apply: {opt}")
            
            # Create performance indexes
            indexes = [
                "CREATE INDEX file_name_idx IF NOT EXISTS FOR (f:File) ON (f.name)",
                "CREATE INDEX file_path_idx IF NOT EXISTS FOR (f:File) ON (f.path)",
                "CREATE INDEX file_package_idx IF NOT EXISTS FOR (f:File) ON (f.package)",
                "CREATE INDEX file_importance_idx IF NOT EXISTS FOR (f:File) ON (f.importance_score)",
                "CREATE INDEX function_name_idx IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
            ]
            
            for index in indexes:
                try:
                    subprocess.run([
                        "docker", "exec", self.container_name,
                        "cypher-shell", "-u", "neo4j", "-p", self.password,
                        index
                    ], capture_output=True, text=True, check=True)
                    logger.info("✅ Created performance index")
                except subprocess.CalledProcessError:
                    logger.info("ℹ️ Index already exists")
            
            logger.info("✅ Runtime optimizations applied")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            return False
    
    def check_container_status(self):
        """Check container status and resource usage"""
        try:
            # Check if container is running
            result = subprocess.run([
                "docker", "ps", "--filter", f"name={self.container_name}", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            ], capture_output=True, text=True)
            
            if self.container_name in result.stdout:
                logger.info("✅ Container is running")
                print(result.stdout)
                
                # Check memory usage
                stats_result = subprocess.run([
                    "docker", "stats", self.container_name, "--no-stream", "--format", 
                    "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
                ], capture_output=True, text=True)
                
                print("\n📊 Resource Usage:")
                print(stats_result.stdout)
                
                return True
            else:
                logger.error("❌ Container is not running")
                return False
                
        except Exception as e:
            logger.error(f"Failed to check status: {e}")
            return False
    
    def generate_management_scripts(self):
        """Generate helpful management scripts"""
        
        # Stop script
        stop_script = f"""#!/bin/bash
echo "🛑 Stopping Neo4j container..."
docker stop {self.container_name}
echo "✅ Neo4j stopped"
"""
        with open('stop_neo4j.sh', 'w') as f:
            f.write(stop_script)
        os.chmod('stop_neo4j.sh', 0o755)
        
        # Status script
        status_script = f"""#!/bin/bash
echo "📊 Neo4j Container Status:"
docker ps --filter name={self.container_name} --format "table {{{{.Names}}}}\t{{{{.Status}}}}\t{{{{.Ports}}}}"

echo ""
echo "📈 Resource Usage:"
docker stats {self.container_name} --no-stream --format "table {{{{.Name}}}}\t{{{{.CPUPerc}}}}\t{{{{.MemUsage}}}}\t{{{{.MemPerc}}}}"

echo ""
echo "🔗 Connection Info:"
echo "  Web UI: http://localhost:{self.neo4j_port}"
echo "  Bolt: bolt://localhost:{self.bolt_port}"
echo "  Username: neo4j"
echo "  Password: {self.password}"
"""
        with open('status_neo4j.sh', 'w') as f:
            f.write(status_script)
        os.chmod('status_neo4j.sh', 0o755)
        
        # Logs script
        logs_script = f"""#!/bin/bash
echo "📋 Neo4j Logs (last 50 lines):"
docker logs {self.container_name} --tail 50 --follow
"""
        with open('logs_neo4j.sh', 'w') as f:
            f.write(logs_script)
        os.chmod('logs_neo4j.sh', 0o755)
        
        logger.info("✅ Generated management scripts")

def main():
    """Main setup function"""
    print("🐳 Docker Neo4j High-Performance Setup")
    print("💾 Configuring for 384GB RAM / 96 Core system")
    print("="*60)
    
    # Initialize setup
    setup = DockerNeo4jSetup()
    
    # Step 1: Create directories and configs
    logger.info("Step 1: Setting up directories and configuration...")
    config_dir, data_dir, logs_dir = setup.create_config_directory()
    setup.generate_optimized_config(config_dir)
    
    # Step 2: Generate Docker files
    logger.info("Step 2: Generating Docker configuration files...")
    setup.create_docker_compose()
    setup.create_docker_run_script()
    setup.generate_management_scripts()
    
    # Step 3: Stop existing container
    logger.info("Step 3: Stopping any existing Neo4j container...")
    setup.stop_existing_container()
    
    # Step 4: Start Neo4j
    logger.info("Step 4: Starting Neo4j container...")
    
    # Try Docker Compose first, fall back to docker run
    if not setup.start_container_compose():
        if not setup.start_container_direct():
            logger.error("❌ Failed to start Neo4j container")
            return
    
    # Step 5: Wait for startup
    logger.info("Step 5: Waiting for Neo4j to be ready...")
    if not setup.wait_for_neo4j():
        logger.error("❌ Neo4j failed to start properly")
        return
    
    # Step 6: Apply optimizations
    logger.info("Step 6: Applying runtime optimizations...")
    setup.apply_runtime_optimizations()
    
    # Step 7: Check status
    logger.info("Step 7: Checking final status...")
    setup.check_container_status()
    
    print("\n" + "="*60)
    print("✅ DOCKER NEO4J SETUP COMPLETE!")
    print("="*60)
    print("📋 Generated Files:")
    print("  🐳 docker-compose.yml - Docker Compose configuration")
    print("  🚀 start_neo4j_hp.sh - High-performance startup script")
    print("  🛑 stop_neo4j.sh - Stop container script")
    print("  📊 status_neo4j.sh - Check status script")
    print("  📋 logs_neo4j.sh - View logs script")
    print("\n🔗 Connection Info:")
    print(f"  🌐 Web UI: http://localhost:{setup.neo4j_port}")
    print(f"  ⚡ Bolt: bolt://localhost:{setup.bolt_port}")
    print(f"  👤 Username: neo4j")
    print(f"  🔑 Password: {setup.password}")
    print("\n💾 Memory Allocation:")
    print("  🧠 JVM Heap: 64GB")
    print("  💾 Page Cache: 200GB")
    print("  🔄 Transaction Memory: 100GB")
    print("  🐳 Container Limit: 320GB")
    print("\n🚀 Ready for High-Performance Analysis!")
    print("  Run: python high_performance_analyzer.py")
    print("="*60)

if __name__ == "__main__":
    main()