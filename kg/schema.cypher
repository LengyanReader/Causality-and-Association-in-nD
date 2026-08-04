// =============================================================================
// Causal Science Knowledge Graph — Neo4j Schema
// =============================================================================
// Run:  cat kg/schema.cypher | docker exec -i neo4j-causal cypher-shell -u neo4j -p causal123
// =============================================================================

// ── Constraints (primary keys) ──
CREATE CONSTRAINT concept_id      IF NOT EXISTS FOR (n:Concept)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT reference_id    IF NOT EXISTS FOR (n:Reference)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT method_id       IF NOT EXISTS FOR (n:Method)       REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT station_id      IF NOT EXISTS FOR (n:Station)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT gallery_id      IF NOT EXISTS FOR (n:GalleryCase)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT principle_id    IF NOT EXISTS FOR (n:Principle)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT bridge_id       IF NOT EXISTS FOR (n:BridgeLevel)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT variable_id     IF NOT EXISTS FOR (n:Variable)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT assumption_id   IF NOT EXISTS FOR (n:Assumption)   REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT theorem_id      IF NOT EXISTS FOR (n:Theorem)      REQUIRE n.id IS UNIQUE;

// ── Indexes (for fast lookup) ──
CREATE INDEX concept_name        IF NOT EXISTS FOR (n:Concept)      ON (n.name);
CREATE INDEX reference_short     IF NOT EXISTS FOR (n:Reference)    ON (n.short);
CREATE INDEX method_name         IF NOT EXISTS FOR (n:Method)       ON (n.name);
CREATE INDEX concept_rung        IF NOT EXISTS FOR (n:Concept)      ON (n.rung);
CREATE INDEX reference_year      IF NOT EXISTS FOR (n:Reference)    ON (n.year);
CREATE INDEX variable_role       IF NOT EXISTS FOR (n:Variable)     ON (n.role);

// ── Full-text indexes for search ──
CREATE FULLTEXT INDEX concept_search IF NOT EXISTS
FOR (n:Concept) ON EACH [n.name, n.definition];
CREATE FULLTEXT INDEX reference_search IF NOT EXISTS
FOR (n:Reference) ON EACH [n.short, n.full];

// ── Node property definitions ──
// Concept:    { id, name, definition, formal_def, rung, aka, glossary }
// Reference:  { id, short, full, year, doi, url, type }
// Method:     { id, name, description, rung, class, assumptions, inputs, outputs }
// Station:    { id, number, name, emoji, question, output_artifact }
// GalleryCase:{ id, name, method, key_result, reference }
// Principle:  { id, number, name, statement }
// BridgeLevel:{ id, level, name, topic, insight, limitation }
// Variable:   { id, name, role, type, description }
// Assumption: { id, name, definition, testable, formal }
// Theorem:    { id, name, statement, reference }
