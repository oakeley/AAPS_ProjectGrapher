#!/bin/bash
# Simple Neo4j Startup Script

echo "🚀 Starting Neo4j with minimal settings..."

# Stop existing
docker stop neo4j-aaps 2>/dev/null || true
docker rm neo4j-aaps 2>/dev/null || true

# Start with minimal settings
docker run -d \
  --name neo4j-aaps \
  --restart unless-stopped \
  -p 7687:7687 \
  -p 7474:7474 \
  -v $(pwd)/neo4j_data:/data \
  -v $(pwd)/neo4j_logs:/logs \
  -v $(pwd)/neo4j_config:/var/lib/neo4j/conf \
  --memory=64g \
  --cpus=24 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_server_memory_heap_initial__size=8g \
  -e NEO4J_server_memory_heap_max__size=16g \
  -e NEO4J_server_memory_pagecache_size=32g \
  -e NEO4J_server_config_strict__validation_enabled=false \
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
