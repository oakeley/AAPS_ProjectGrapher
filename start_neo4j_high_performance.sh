#!/bin/bash
# Unified Neo4j Startup Script (high_performance mode)

echo "🚀 Starting Neo4j with high_performance settings..."

# Stop existing container
docker stop neo4j-aaps 2>/dev/null || true
docker rm neo4j-aaps 2>/dev/null || true

# Start with high_performance settings
docker run -d \
  --name neo4j-aaps \
  --restart unless-stopped \
  -p 7687:7687 \
  -p 7474:7474 \
  -v $(pwd)/neo4j_data:/data \
  -v $(pwd)/neo4j_logs:/logs \
  -v $(pwd)/neo4j_config:/var/lib/neo4j/conf \
  --memory=320g \
  --cpus=96 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_server_memory_heap_initial__size=32g \
  -e NEO4J_server_memory_heap_max__size=64g \
  -e NEO4J_server_memory_pagecache_size=200g \
  -e NEO4J_server_memory_transaction_total_max=100g \
  -e NEO4J_server_jvm_additional=-XX:+UseG1GC \
  -e NEO4J_server_jvm_additional=-XX:MaxGCPauseMillis=200 \
  -e NEO4J_server_jvm_additional=-XX:+DisableExplicitGC \
  -e NEO4J_db_transaction_timeout=300s \
  -e NEO4J_dbms_query_cache__size=5000 \
  -e NEO4J_cypher_runtime=slotted \
  -e NEO4J_server_config_strict__validation_enabled=false \
  neo4j:5.15-community

echo "⏳ Waiting for Neo4j to start..."

# Wait for startup with detailed feedback
for i in {1..120}; do
    if curl -s http://localhost:7474 > /dev/null 2>&1; then
        echo "✅ Neo4j web interface is responding!"
        break
    fi
    
    if [ $((i % 15)) -eq 0 ]; then
        echo "⏳ Still waiting... (${i}/120 attempts)"
        echo "📋 Recent logs:"
        docker logs neo4j-aaps --tail 3 2>/dev/null || echo "No logs yet"
    fi
    
    sleep 3
done

# Test database connection
echo "🔌 Testing database connection..."
for i in {1..20}; do
    if docker exec neo4j-aaps cypher-shell -u neo4j -p password "RETURN 1" 2>/dev/null; then
        echo ""
        echo "🎉 Neo4j is ready! (high_performance mode)"
        echo "💾 Memory: 320g container / 64g heap / 200g cache"
        echo "🌐 Web UI: http://localhost:7474"
        echo "⚡ Bolt: bolt://localhost:7687"
        echo "👤 Username: neo4j"
        echo "🔑 Password: password"
        echo ""
        echo "📊 Container stats:"
        docker stats neo4j-aaps --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null || echo "Stats not available"
        echo ""
        echo "🚀 Ready to run: python high_performance_analyzer.py"
        exit 0
    fi
    
    echo "⏳ Waiting for database... (attempt $i/20)"
    sleep 3
done

echo ""
echo "❌ Database connection failed after high_performance startup attempt"
echo "🔍 Check logs: docker logs neo4j-aaps"
echo "🔄 Try minimal mode: python docker_neo4j_setup.py --mode=limited"
exit 1
