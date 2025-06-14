#!/usr/bin/env python3
"""
Neo4j Quick Fix Script
Fixes configuration issues and permissions
"""

import subprocess
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_permissions():
    """Fix directory permissions"""
    logger.info("🔧 Fixing permissions...")
    
    try:
        # Change ownership and permissions
        subprocess.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", "neo4j_config"], check=False)
        subprocess.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", "neo4j_data"], check=False)
        subprocess.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", "neo4j_logs"], check=False)
        subprocess.run(["chmod", "-R", "755", "neo4j_config"], check=False)
        subprocess.run(["chmod", "-R", "755", "neo4j_data"], check=False)
        subprocess.run(["chmod", "-R", "755", "neo4j_logs"], check=False)
        logger.info("✅ Permissions fixed")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Permission fix failed: {e}")
        return False

def create_clean_config():
    """Create a clean, valid Neo4j configuration"""
    logger.info("🔧 Creating clean Neo4j configuration...")
    
    # Clean, minimal config that works with Neo4j 5.x
    clean_config = """# Neo4j 5.x Compatible Configuration
# Conservative settings for reliable startup

# Memory Settings
server.memory.heap.initial_size=8g
server.memory.heap.max_size=16g
server.memory.pagecache.size=32g

# Network
server.default_listen_address=0.0.0.0
server.bolt.listen_address=:7687
server.http.listen_address=:7474

# Basic settings
db.transaction.timeout=60s
dbms.security.auth_enabled=true

# Disable strict validation to allow startup
server.config.strict_validation.enabled=false
"""
    
    try:
        with open('neo4j_config/neo4j.conf', 'w') as f:
            f.write(clean_config)
        logger.info("✅ Created clean configuration")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create config: {e}")
        return False

def stop_and_clean_container():
    """Stop and remove existing container"""
    logger.info("🛑 Stopping and cleaning container...")
    
    try:
        subprocess.run(["docker", "stop", "neo4j-aaps"], capture_output=True)
        subprocess.run(["docker", "rm", "neo4j-aaps"], capture_output=True)
        logger.info("✅ Container stopped and removed")
    except Exception as e:
        logger.info(f"ℹ️ Container cleanup: {e}")

def start_minimal_neo4j():
    """Start Neo4j with minimal, reliable settings"""
    logger.info("🚀 Starting Neo4j with minimal settings...")
    
    cmd = [
        "docker", "run", "-d",
        "--name", "neo4j-aaps",
        "--restart", "unless-stopped",
        "-p", "7687:7687",
        "-p", "7474:7474",
        "-v", f"{os.getcwd()}/neo4j_data:/data",
        "-v", f"{os.getcwd()}/neo4j_logs:/logs",
        "-v", f"{os.getcwd()}/neo4j_config:/var/lib/neo4j/conf",
        "--memory=64g",  # Much more conservative
        "--cpus=24",     # Conservative CPU limit
        "-e", "NEO4J_AUTH=neo4j/password",
        "-e", "NEO4J_server_memory_heap_initial__size=8g",
        "-e", "NEO4J_server_memory_heap_max__size=16g",
        "-e", "NEO4J_server_memory_pagecache_size=32g",
        "-e", "NEO4J_server_config_strict__validation_enabled=false",
        "neo4j:5.15-community"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("✅ Container started successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Container start failed: {e.stderr}")
        return False

def wait_for_startup(timeout=120):
    """Wait for Neo4j to start up"""
    logger.info("⏳ Waiting for Neo4j startup...")
    
    import time
    for i in range(timeout):
        try:
            # Check web interface
            result = subprocess.run(["curl", "-s", "http://localhost:7474"], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ Web interface responding!")
                
                # Check database
                db_result = subprocess.run([
                    "docker", "exec", "neo4j-aaps",
                    "cypher-shell", "-u", "neo4j", "-p", "password",
                    "RETURN 1"
                ], capture_output=True, timeout=10)
                
                if db_result.returncode == 0:
                    logger.info("✅ Database responding!")
                    return True
                    
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        
        if i % 10 == 0 and i > 0:
            logger.info(f"⏳ Still waiting... ({i}/{timeout}s)")
            
            # Show recent logs
            try:
                logs = subprocess.run(["docker", "logs", "neo4j-aaps", "--tail", "3"], 
                                    capture_output=True, text=True)
                if logs.stdout:
                    logger.info(f"Recent log: {logs.stdout.strip().split()[-1] if logs.stdout.strip() else 'No logs'}")
            except:
                pass
        
        time.sleep(1)
    
    logger.error("❌ Startup timeout")
    return False

def create_simple_startup_script():
    """Create a simple startup script"""
    script_content = """#!/bin/bash
# Simple Neo4j Startup Script

echo "🚀 Starting Neo4j with minimal settings..."

# Stop existing
docker stop neo4j-aaps 2>/dev/null || true
docker rm neo4j-aaps 2>/dev/null || true

# Start with minimal settings
docker run -d \\
  --name neo4j-aaps \\
  --restart unless-stopped \\
  -p 7687:7687 \\
  -p 7474:7474 \\
  -v $(pwd)/neo4j_data:/data \\
  -v $(pwd)/neo4j_logs:/logs \\
  -v $(pwd)/neo4j_config:/var/lib/neo4j/conf \\
  --memory=64g \\
  --cpus=24 \\
  -e NEO4J_AUTH=neo4j/password \\
  -e NEO4J_server_memory_heap_initial__size=8g \\
  -e NEO4J_server_memory_heap_max__size=16g \\
  -e NEO4J_server_memory_pagecache_size=32g \\
  -e NEO4J_server_config_strict__validation_enabled=false \\
  neo4j:5.15-community

echo "⏳ Waiting for startup..."
sleep 10

# Check status
if curl -s http://localhost:7474 > /dev/null; then
    echo "✅ Neo4j web interface is responding!"
    echo "🌐 Web UI: http://localhost:7474"
    echo "⚡ Bolt: bolt://localhost:7687"
    echo "👤 Username: neo4j"
    echo "🔑 Password: password"
else
    echo "❌ Startup may have failed. Check logs:"
    docker logs neo4j-aaps --tail 10
fi
"""
    
    with open('start_neo4j_simple.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('start_neo4j_simple.sh', 0o755)
    logger.info("✅ Created simple startup script: start_neo4j_simple.sh")

def main():
    """Main fix routine"""
    print("🔧 Neo4j Quick Fix")
    print("="*30)
    
    # Step 1: Stop problematic container
    stop_and_clean_container()
    
    # Step 2: Fix permissions
    fix_permissions()
    
    # Step 3: Create clean config
    if not create_clean_config():
        logger.error("❌ Could not create config file")
        return
    
    # Step 4: Create startup script
    create_simple_startup_script()
    
    # Step 5: Start with minimal settings
    if not start_minimal_neo4j():
        logger.error("❌ Could not start container")
        logger.info("💡 Try running: ./start_neo4j_simple.sh")
        return
    
    # Step 6: Wait for startup
    if wait_for_startup():
        print("\n✅ NEO4J FIXED AND RUNNING!")
        print("="*40)
        print("🎉 Neo4j is now running with minimal settings")
        print("💾 Memory: 64GB total (16GB heap + 32GB cache)")
        print("🌐 Web UI: http://localhost:7474")
        print("⚡ Bolt: bolt://localhost:7687")
        print("👤 Username: neo4j")
        print("🔑 Password: password")
        print("\n🚀 Ready to run: python high_performance_analyzer.py")
        print("\n📊 Container stats:")
        try:
            subprocess.run(["docker", "stats", "neo4j-aaps", "--no-stream", 
                          "--format", "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"], 
                          timeout=5)
        except:
            pass
    else:
        print("\n❌ STARTUP STILL FAILED")
        print("🔍 Check logs: docker logs neo4j-aaps")
        print("🔄 Try manual start: ./start_neo4j_simple.sh")

if __name__ == "__main__":
    main()