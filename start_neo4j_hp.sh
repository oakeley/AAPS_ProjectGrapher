#!/bin/bash
# High-Performance Neo4j Docker Run Script

# Stop existing container
docker stop neo4j-aaps 2>/dev/null || true
docker rm neo4j-aaps 2>/dev/null || true

# Run Neo4j with high-performance settings
docker run -d \
  --name neo4j-aaps \
  --restart unless-stopped \
  -p 7687:7687 \
  -p 7474:7474 \
  -v $(pwd)/neo4j_data:/data \
  -v $(pwd)/neo4j_logs:/logs \
  -v $(pwd)/neo4j_config:/var/lib/neo4j/conf \
  --memory=320g \
  --memory-reservation=300g \
  --cpus=96 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_server_memory_heap_initial__size=32g \
  -e NEO4J_server_memory_heap_max__size=64g \
  -e NEO4J_server_memory_pagecache_size=200g \
  -e NEO4J_server_memory_transaction_total_max=100g \
  -e NEO4J_server_jvm_additional=-XX:+UseG1GC \
  -e NEO4J_server_jvm_additional=-XX:G1HeapRegionSize=32m \
  -e NEO4J_server_jvm_additional=-XX:MaxGCPauseMillis=200 \
  -e NEO4J_server_jvm_additional=-XX:+ParallelRefProcEnabled \
  -e NEO4J_server_jvm_additional=-XX:+UseStringDeduplication \
  -e NEO4J_db_transaction_timeout=600s \
  -e NEO4J_dbms_query_cache__size=10000 \
  -e NEO4J_cypher_runtime=slotted \
  -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
  neo4j:5.15-enterprise

echo "🚀 Neo4j container starting with high-performance settings..."
echo "📊 Memory allocation: 320GB"
echo "🔧 CPUs: 96 cores"
echo "🌐 Web interface: http://localhost:7474"
echo "⚡ Bolt: bolt://localhost:7687"
echo ""
echo "Waiting for Neo4j to start..."

# Wait for Neo4j to be ready
until docker exec neo4j-aaps cypher-shell -u neo4j -p password "RETURN 1" 2>/dev/null; do
  echo "⏳ Waiting for Neo4j..."
  sleep 5
done

echo "✅ Neo4j is ready!"
