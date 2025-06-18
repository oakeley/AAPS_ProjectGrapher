#!/bin/bash
# Enhanced Neo4j Startup Script - Intelligent Memory Management
# Configured for 48GB heap and 32GB page cache

echo "🚀 Starting Neo4j with intelligent memory allocation..."
echo "   Java Heap: 36GB - 48GB"
echo "   Page Cache: 32GB"
echo "   Docker Limit: 88GB"

# Stop existing container
docker stop neo4j-aaps 2>/dev/null || true
docker rm neo4j-aaps 2>/dev/null || true

# Start with intelligent settings
docker run -d \
  --name neo4j-aaps \
  --restart unless-stopped \
  -p 7687:7687 \
  -p 7474:7474 \
  -v $(pwd)/neo4j_data:/data \
  -v $(pwd)/neo4j_logs:/logs \
  -v $(pwd)/neo4j_config:/var/lib/neo4j/conf \
  --memory=88g \
  --memory-reservation=79g \
  --cpus=32 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_server_memory_heap_initial__size=36g \
  -e NEO4J_server_memory_heap_max__size=48g \
  -e NEO4J_server_memory_pagecache_size=32g \
  -e NEO4J_server_memory_transaction_total_max=12g \
  -e NEO4J_server_jvm_additional="-XX:+UseG1GC -XX:G1HeapRegionSize=32m -XX:MaxGCPauseMillis=200 -XX:+UseStringDeduplication" \
  -e NEO4J_db_transaction_timeout=600s \
  -e NEO4J_server_config_strict__validation_enabled=false \
  neo4j:5.15-community

echo "⏳ Waiting for Neo4j to start..."
sleep 15

# Verify startup
if docker exec neo4j-aaps cypher-shell -u neo4j -p password "RETURN 1" > /dev/null 2>&1; then
    echo "✅ Neo4j is ready!"
    echo "🌐 Web UI: http://localhost:7474"
    echo "⚡ Bolt: bolt://localhost:7687"
    echo "👤 Username: neo4j"
    echo "🔑 Password: password"
    echo "💾 Heap Allocation: 36GB - 48GB"
else
    echo "❌ Startup verification failed"
    echo "Check logs: docker logs neo4j-aaps"
fi
