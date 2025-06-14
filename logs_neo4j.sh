#!/bin/bash
echo "📋 Neo4j Logs (last 50 lines):"
docker logs neo4j-aaps --tail 50 --follow
