// =============================================================================
// Causal Science Knowledge Graph — Query Library
// =============================================================================
// Usage: cat kg/queries.cypher | docker exec -i neo4j-causal cypher-shell -u neo4j -p causal123
// =============================================================================

// ── 1. TRACE: Full assumption provenance for a method ──
// "What assumptions does AIPW require, who proved them, and when?"
MATCH (m:Method {id: "aipw"})-[:REQUIRES]->(c:Concept)<-[:PROVES]-(r:Reference)
RETURN m.name AS method, c.name AS assumption, r.short AS proven_by, r.year AS year
ORDER BY r.year;

// ── 2. CHAIN: Prerequisite chain from foundations to a concept ──
// "What do I need to understand before 'double robustness'?"
MATCH path = (foundation)-[:PREREQUISITE_FOR*1..4]->(target:Concept {id: "double-robustness"})
WHERE NOT (()-[:PREREQUISITE_FOR]->(foundation))
RETURN [n IN nodes(path) | n.name] AS learning_path, length(path) AS depth
ORDER BY depth;

// ── 3. ECOSYSTEM: All intellectual inputs to a causal concept ──
// "Where does 'ignorability' come from — across all fields?"
MATCH (c:Concept {id: "ignorability"})<-[r:PREREQUISITE_FOR|PROVES]-(source)
RETURN labels(source)[0] AS field, coalesce(source.short, source.name) AS source,
       type(r) AS relationship
ORDER BY field;

// ── 4. GALLERY: Which real-world cases use a given method?
// "Where has DiD been applied, and what did it find?"
MATCH (m:Method)-[:APPLIES]-(g:GalleryCase)
WHERE toLower(m.name) CONTAINS toLower($method)
RETURN g.name AS case_study, g.method AS method_used, g.result AS key_finding;

// ── 5. DAG: What does the NomNom DAG encode?
// "Show me all confounders and what they affect"
MATCH (v:Variable {role: "proxy_confounder"})-[:CAUSES]->(target)
RETURN v.id AS confounder, collect(target.id) AS affects;

// ── 6. CURRICULUM: What does each math bridge level teach?
MATCH (b:BridgeLevel)-[:TEACHES]->(c:Concept)
RETURN b.level AS level, b.name AS bridge_level,
       collect(c.name) AS concepts_taught
ORDER BY b.level;

// ── 7. STATION: What concepts does each UCL station use?
MATCH (s:Station)-[:USES]->(c:Concept)
RETURN s.number AS station, s.name AS station_name,
       collect(c.name) AS concepts_used
ORDER BY s.number;

// ── 8. TIMELINE: Knowledge evolution over time
MATCH (r:Reference) WHERE r.year IS NOT NULL
WITH CASE WHEN r.year < 1900 THEN 'pre-1900'
          WHEN r.year < 1950 THEN '1900-1949'
          WHEN r.year < 1980 THEN '1950-1979'
          WHEN r.year < 2000 THEN '1980-1999'
          WHEN r.year < 2020 THEN '2000-2019'
          ELSE '2020+' END AS era, r
RETURN era, count(r) AS references, collect(r.short)[..5] AS examples
ORDER BY era;

// ── 9. CENTRALITY: Most connected concepts (hub nodes)
MATCH (c:Concept)-[r]-()
RETURN c.name AS concept, c.rung AS rung, count(r) AS connections
ORDER BY connections DESC LIMIT 10;

// ── 10. GAPS: Stations with no concept links (completeness check)
MATCH (s:Station) WHERE NOT (s)-[:USES]->(:Concept)
RETURN s.id AS station, s.question AS question;

// ── 11. ABSENT: Falsifiable absent-edge claims in the NomNom DAG
MATCH (a:Variable)-[r:NON_CAUSES]->(b:Variable)
RETURN a.id + ' -/-> ' + b.id AS absent_edge, r.falsifiable AS testable;

// ── 12. CROSS-DISCIPLINE: All fields contributing to causal science
MATCH (c:Concept)<-[:PROVES]-(r:Reference)
WHERE c.rung = 2
RETURN c.name AS causal_concept,
       collect(r.short)[..3] AS proven_by
ORDER BY c.name;

// ── 13. FRONTIER: Active research areas (concepts with recent refs)
MATCH (c:Concept)<-[:PROVES]-(r:Reference)
WHERE r.year >= 2020
RETURN r.year AS year, r.short AS reference, collect(c.name)[..2] AS concepts
ORDER BY r.year DESC;

// ── 14. FULL PATH: From raw data to causal estimate
MATCH path = (start:Station {number: 0})-[:PRECEDES*]->(end:Station {number: 5})
RETURN [s IN nodes(path) | s.name] AS pipeline, length(path) AS steps;

// ── 15. DESIGN: Which design principles are implemented by which stations?
MATCH (p:Principle)-[:IMPLEMENTED_BY]->(target)
RETURN p.number AS principle, p.name AS principle_name,
       labels(target)[0] AS implemented_in, coalesce(target.name, target.id) AS target
ORDER BY p.number;
