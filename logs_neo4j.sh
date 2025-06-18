#!/bin/bash
echo "📋 Neo4j Logs:"
docker logs neo4j-aaps --tail 50 --follow
