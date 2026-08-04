"""Causal Science Knowledge Graph.

Usage:
  python kg/populate.py          # populate Neo4j from project data
  python kg/export_sqlite.py      # export Neo4j to SQLite
  bash kg/bootstrap.sh             # full rebuild from scratch

Query:
  cat kg/queries.cypher | docker exec -i neo4j-causal cypher-shell -u neo4j -p causal123

Node types:  Concept | Reference | Method | Station | Variable | GalleryCase | Principle | BridgeLevel
Edges:       PROVES | REQUIRES | PREREQUISITE_FOR | TEACHES | PRECEDES | CAUSES | NON_CAUSES |
             IMPLEMENTED_BY | USES | USES_VARIABLE | DEMONSTRATES
"""
