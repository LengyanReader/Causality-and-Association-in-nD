#!/bin/bash
# =============================================================================
# Causal Science KG — Full Bootstrap
# =============================================================================
# One command to rebuild everything from scratch.
#
# Usage:
#   bash kg/bootstrap.sh          # full rebuild
#   bash kg/bootstrap.sh --query  # only run queries after populate
# =============================================================================
set -e

NEO4J_AUTH="neo4j/causal123"
echo "=== Causal Science Knowledge Graph Bootstrap ==="

# 1. Ensure Neo4j is running
if ! docker ps --format '{{.Names}}' | grep -q neo4j-causal; then
    echo "Starting Neo4j..."
    docker run -d --name neo4j-causal \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=$NEO4J_AUTH \
        neo4j:latest
    echo "Waiting for Neo4j to be ready..."
    for i in $(seq 1 30); do
        curl -s http://localhost:7474 > /dev/null 2>&1 && break
        sleep 2
    done
fi

# 2. Apply schema (constraints + indexes)
echo ""
echo "--- Applying schema ---"
cat kg/schema.cypher | docker exec -i neo4j-causal cypher-shell -u neo4j -p causal123

# 3. Populate from project data
echo ""
echo "--- Populating knowledge graph ---"
python kg/populate.py

# 4. Export to SQLite
echo ""
echo "--- Exporting to SQLite ---"
python kg/export_sqlite.py

# 5. Run verification queries
echo ""
echo "--- Verification queries ---"
cat kg/queries.cypher | docker exec -i neo4j-causal cypher-shell -u neo4j -p causal123 2>&1 | head -60

echo ""
echo "=== Bootstrap complete ==="
echo "  Neo4j:  http://localhost:7474 (neo4j/causal123)"
echo "  SQLite: kg/causal_kg.sqlite"
echo "  Queries: kg/queries.cypher"
