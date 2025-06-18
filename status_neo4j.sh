#!/bin/bash
# Enhanced Neo4j Status Script

echo "📊 Neo4j Status Dashboard"
echo "=========================="

echo "🐳 Container Status:"
docker ps --filter name=neo4j-aaps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "📈 Resource Usage:"
docker stats neo4j-aaps --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

echo ""
echo "🧠 Java Memory Info:"
docker exec neo4j-aaps java -XX:+PrintFlagsFinal -version 2>/dev/null | grep -E "(InitialHeapSize|MaxHeapSize)" || echo "Could not retrieve Java memory info"

echo ""
echo "🔗 Connection Info:"
echo "  Web UI: http://localhost:7474"
echo "  Bolt: bolt://localhost:7687"
echo "  Username: neo4j"
echo "  Password: password"
echo "  Configured Heap: 36GB - 48GB"
echo "  Page Cache: 32GB"
