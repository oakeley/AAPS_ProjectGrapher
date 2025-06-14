#!/bin/bash
echo "📊 Neo4j Container Status:"
docker ps --filter name=neo4j-aaps --format "table {{.Names}}	{{.Status}}	{{.Ports}}"

echo ""
echo "📈 Resource Usage:"
docker stats neo4j-aaps --no-stream --format "table {{.Name}}	{{.CPUPerc}}	{{.MemUsage}}	{{.MemPerc}}"

echo ""
echo "🔗 Connection Info:"
echo "  Web UI: http://localhost:7474"
echo "  Bolt: bolt://localhost:7687"
echo "  Username: neo4j"
echo "  Password: password"
