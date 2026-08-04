"""Export Neo4j KG to SQLite for offline/portable use.

Produces kg/causal_kg.sqlite with tables mirroring the Neo4j node types.
"""
import sqlite3, sys
from pathlib import Path
from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parents[1]
DB_PATH = REPO / "kg" / "causal_kg.sqlite"
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "causal123")

NODE_LABELS = ["Concept", "Reference", "Method", "Station", "Variable",
               "GalleryCase", "Principle", "BridgeLevel"]

def export():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()

    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    with driver.session() as s:
        # Export nodes
        for label in NODE_LABELS:
            rows = list(s.run(f"MATCH (n:{label}) RETURN properties(n) AS props"))
            if not rows:
                continue
            # Collect all keys
            keys = set()
            for r in rows:
                keys.update(r["props"].keys())
            keys = sorted(keys)
            # Create table
            cols = ", ".join(f'"{k}" TEXT' for k in keys)
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{label}" ({cols})')
            # Insert
            placeholders = ", ".join("?" * len(keys))
            for r in rows:
                vals = [json_safe(r["props"].get(k)) for k in keys]
                conn.execute(f'INSERT INTO "{label}" VALUES ({placeholders})', vals)
            print(f"  {label}: {len(rows)} rows")

        # Export relationships
        rel_rows = list(s.run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] AS from_type, a.id AS from_id,
                   type(r) AS relationship,
                   labels(b)[0] AS to_type, b.id AS to_id
        """))
        conn.execute("""CREATE TABLE relationships (
            from_type TEXT, from_id TEXT, relationship TEXT, to_type TEXT, to_id TEXT
        )""")
        for r in rel_rows:
            conn.execute("INSERT INTO relationships VALUES (?,?,?,?,?)",
                        (r["from_type"], r["from_id"], r["relationship"],
                         r["to_type"], r["to_id"]))
        print(f"  Relationships: {len(rel_rows)} rows")

    conn.commit()
    conn.close()
    driver.close()
    print(f"\nExported to {DB_PATH} ({DB_PATH.stat().st_size:,} bytes)")


def json_safe(v):
    """Convert lists/dicts to JSON strings for SQLite storage."""
    import json
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


if __name__ == "__main__":
    export()
