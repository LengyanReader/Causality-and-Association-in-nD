# Knowledge Graph for Causal Science — Feasibility Analysis

> **Status: IMPLEMENTED** (2026-08-05) — The KG is live at `kg/` with 133 nodes, 154 relationships.  
> See [README.md](../README.md#-knowledge-graph-kg) and the [interactive browser](../docs/demo/kg_browser.html).

## 1. What Already Exists (Seed Material)

The project already contains substantial semi-structured knowledge that could seed a KG:

| Source | Structured content | KG-ready entities |
|--------|-------------------|-------------------|
| Literature review | 17 sections, ~140 refs (Hume 1748→TARGET 2025) | References, concepts, theorems, debates |
| Demo glossary | 17 terms with definitions | Concept nodes with plain-English descriptions |
| Math bridge | 6 levels, each with a "bridge insight" | Propositions about capability boundaries |
| UCL stations | 9 stations, typed contracts, sensors, actuators | Process nodes, artifact types, health signals |
| NomNom DAG | 11 nodes, 17 edges, typed roles | Causal variable ontology (confounder/mediator/collider/instrument/neg.control) |
| Design principles | 6 principles (P1–P6) | Meta-level claims about methodology |
| Deep dives | 4 pages (DGP, back-door, E-value, DAG building) | Method explanations with references |
| Tier-1 gallery | 6 classic reproductions | Concept→method→evidence triples |
| Demo thinking chains | 9 stations × layered reasoning | Step-by-step causal reasoning traces |
| Implementation plan | Phase roadmap, risk register | Project-level metadata |

**Estimated raw entity count:** 200+ references, 50+ concepts, 30+ methods, 20+ formal relationships, 6 gallery cases.

## 2. What a KG Would Add

### 2.1 Entity Types (Proposed Schema)

```
REFERENCE (140+)
  - short_name, full_citation, year, url, doi
  - type: book | paper | software | dataset

CONCEPT (50+)
  - name, definition, rung (1|2|3), formal_definition (LaTeX)
  - type: estimand | assumption | criterion | theorem | paradox | method_class
  - examples: "ATE", "back-door criterion", "d-separation", "ignorability",
              "Simpson's paradox", "Neyman orthogonality", "do-operator",
              "potential outcomes", "structural equation", "collider bias"

METHOD (30+)
  - name, description, rung, assumptions, inputs, outputs
  - type: estimator | identification | discovery | sensitivity | monitoring
  - examples: "AIPW", "DML", "PC algorithm", "FCI", "GES", "TMLE",
              "synthetic control", "DiD", "RDD", "IV", "E-value"

RELATIONSHIP (edges between entities)
  - REQUIRES: method → assumption (AIPW REQUIRES ignorability)
  - PROVES: reference → theorem (Pearl 1995 PROVES back-door criterion)
  - APPLIES: gallery_case → method (LaLonde APPLIES propensity matching)
  - GENERALIZES: concept_a → concept_b (DML GENERALIZES AIPW)
  - CONTRADICTS: claim_a → claim_b (Fisher CONTRADICTS Doll & Hill)
  - ASSUMES: method → assumption (IV ASSUMES exclusion restriction)
  - PART_OF: concept → framework (do-operator PART_OF SCM framework)
  - PRECEDES: station_a → station_b (IDENTIFY PRECEDES MODEL)
  - HAS_SENSOR: station → health_signal (MODEL HAS_SENSOR balance_check)
  - BRIDGE_INSIGHT: level → proposition (Level 1 BRIDGE_INSIGHT "Bayes never leaves rung 1")
```

### 2.2 Query Power

With this schema, you could answer:

- **"What assumptions does AIPW require, and which references justify them?"**
  → AIPW -[REQUIRES]-> ignorability -[JUSTIFIED_BY]-> Rubin 1974, Rosenbaum & Rubin 1983

- **"Which methods work on Rung 3 (counterfactuals)?"**
  → abduction-action-prediction, structural counterfactuals, mediation formulas

- **"What does the literature say about collider bias?"**
  → Berkson 1946 -[DESCRIBES]-> collider_bias -[APPEARS_IN]-> NomNom DAG (node S)

- **"If I change the DAG (remove W→T), what downstream results change?"**
  → W→T removal → back-door criterion fails → ATE not identified → all estimates invalid

- **"Which gallery cases use DiD, and what assumptions do they need?"**
  → Card-Krueger -[APPLIES]-> DiD -[ASSUMES]-> parallel trends

## 3. Implementation Options

### Option A: Lightweight JSON-LD / Linked Data (Recommended Start)

**What:** A single JSON-LD file (`causal_kg.json`) served alongside the demo, with a simple SPARQL-like query interface in JavaScript.

**Schema:** Lightweight ontology defined inline using schema.org/CIDOC-CRM patterns:
```json
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@id": "concept:back-door-criterion",
      "@type": "causal:Concept",
      "name": "Back-Door Criterion",
      "rung": 2,
      "definition": "A set Z satisfies the back-door criterion relative to (T,Y) if...",
      "formalDefinition": "T ⊥ Y | Z in G_BD",
      "provenBy": ["ref:pearl-1995"],
      "usedIn": ["station:identify", "gallery:lalonde-nsw"]
    }
  ]
}
```

**Pros:** Zero dependencies, browser-queryable, integrates with demo, version-controlled in git  
**Cons:** Manual population, no inference engine, limited scalability  
**Effort:** 2-3 weeks for MVP with ~100 entities

### Option B: Neo4j Graph Database

**What:** A proper graph database with Cypher queries, running locally or in the cloud.

**Pros:** Full graph query power, inference, visualization, scalable  
**Cons:** Requires Neo4j installation, more complex deployment, harder to integrate with static demo  
**Effort:** 4-6 weeks for MVP, ongoing maintenance

### Option C: Python-Native NetworkX + JSON

**What:** Build the KG as a NetworkX MultiDiGraph, export to JSON for the demo, query in Python.

**Pros:** Leverages existing Python stack (already used for DAGs), type-safe, testable  
**Cons:** Not browser-queryable without JSON export, Python-only  
**Effort:** 2-3 weeks for MVP

## 4. Integration with Existing Demo

The KG would enhance the demo in concrete ways:

1. **Glossary → Concept nodes:** The 17 glossary terms become first-class entities with typed relationships to references, methods, and DAG nodes.

2. **"Why this adjustment set?" → Graph query:** Clicking a DAG node could show its KG context: "U is a CONFOUNDER (proven by: Pearl 1995 §3.3), APPEARS IN: Berkson 1946, REQUIRES: proxy W"

3. **Station thinking chains → Traced reasoning:** Each mental operation in the thinking blocks could reference KG entities, making the reasoning traceable: "Mental operation 2 CITES d-separation (Pearl 1995), APPLIES back-door criterion"

4. **Literature review → Queryable reference graph:** Instead of a flat markdown file, references become queryable: "Show me all papers that discuss instrumental variables and were published after 2000"

5. **Math bridge → Formal curriculum graph:** Each level's "bridge insight" becomes a formal proposition with prerequisites and consequences: "Level 3 REQUIRES Level 2, PROVES do-calculus rules R1-R3"

## 5. Value Assessment

### High-Value Use Cases

| Use case | Value | Feasibility |
|----------|-------|-------------|
| Auto-generate "further reading" from concept mentions | High — enriches demo without manual curation | Easy — query by concept, return top-k references |
| Validate adjustment sets by querying DAG→KG | High — catches errors in station logic | Medium — requires formalizing d-separation in KG |
| Track assumption provenance across the pipeline | High — directly supports P1 (assumptions as artifacts) | Easy — link assumptions to references |
| Generate test suites from KG implications | Medium — supports P4 (continuous refutation) | Hard — requires inference engine |
| Cross-reference gallery cases by method | Medium — helps users find relevant examples | Easy — query method→gallery_case |

### Key Benefits

1. **Single source of truth for concepts:** Currently, the same concept (e.g., "ignorability") is defined differently in the glossary, the literature review, the math bridge, and station thinking chains. A KG unifies these.

2. **Assumption traceability:** P1 says "assumptions are first-class artifacts." A KG makes this operational — every assumption links to the references that justify it, the methods that require it, and the tests that check it.

3. **Discoverability:** A new user exploring the demo can follow KG links from any concept to related concepts, references, methods, and examples.

4. **Machine readability:** An LLM/agent could query the KG to answer causal questions grounded in the project's curated knowledge, rather than hallucinating.

## 6. Risks and Challenges

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Population burden:** Manual entry of 200+ entities | High | Start with auto-extraction from existing structured sources (glossary→concepts, refs→reference nodes); only manually curate relationships |
| **Schema drift:** Ontology evolves as understanding deepens | Medium | Version the ontology alongside the code; use git for schema changes |
| **Synchronization:** KG must stay in sync with literature review and demo | Medium | Generate KG from canonical sources (lit review, demo glossary) rather than maintaining independently |
| **Over-engineering:** KG becomes a project in itself, distracting from causal science work | High | Start with Option A (lightweight JSON-LD), only expand if it proves valuable |
| **Query complexity:** Users may not know how to query a KG | Low | Provide pre-built queries in the demo UI; no need for users to write SPARQL/Cypher |

## 7. Recommended Path

### Phase 1: Concept Graph (1-2 weeks)
- Extract all ~50 concepts from glossary + math bridge + station definitions
- Define a minimal schema: Concept(id, name, definition, rung, type)
- Add `RELATED_TO` edges between concepts
- Export as `causal_kg.json` in the demo directory
- Add a "Knowledge Graph" tab/section to the demo that renders concept→concept links

### Phase 2: Reference Integration (1-2 weeks)
- Parse the literature review's bibliography into structured Reference nodes
- Add `PROVES`, `DESCRIBES`, `INTRODUCES` edges from references to concepts
- Auto-generate "further reading" links in the demo from KG queries

### Phase 3: Method & Gallery Integration (1-2 weeks)
- Add Method nodes (AIPW, DML, PC, FCI, etc.)
- Link methods to assumptions (REQUIRES) and references (PROVEN_BY)
- Link gallery cases to methods (APPLIES)
- Enable queries like "which methods could replace AIPW for this estimand?"

### Phase 4: Reasoning Traces (optional, 2-3 weeks)
- Encode the thinking chains from each station as formal reasoning steps
- Each step links to the KG entities it references
- Enable "explain this estimate" queries that trace from data to conclusion through the KG

## 8. Concrete First Step

Create `causal_kg.json`:

```json
{
  "ontology": {
    "entity_types": ["Concept", "Reference", "Method", "Assumption", "Theorem", "GalleryCase"],
    "relationship_types": ["REQUIRES", "PROVES", "APPLIES", "GENERALIZES", "ASSUMES", "PART_OF", "CONTRADICTS"]
  },
  "concepts": [
    {
      "id": "back-door-criterion",
      "name": "Back-Door Criterion",
      "rung": 2,
      "definition": "A set Z satisfies the back-door criterion relative to (T,Y) if no node in Z is a descendant of T, and Z d-separates T from Y in the back-door graph.",
      "formal": "T ⊥ Y | Z in G_BD",
      "introduced_by": "pearl-1995",
      "used_in_stations": ["identify"],
      "related_to": ["d-separation", "adjustment-set", "confounding", "do-operator"]
    }
  ]
}
```

Populate with the 17 glossary terms first, then expand.

## 9. Verdict

**Feasible and valuable — start small, prove value, expand incrementally.**

The project already has the raw material: structured concepts, typed relationships, and a CI pipeline that could validate KG consistency. The lightweight JSON-LD approach (Option A) is low-risk, low-cost, and directly integrable with the existing demo. The main risk is scope creep — treating the KG as a separate project rather than a knowledge layer that serves the causal science work. Start with the glossary→concepts mapping and one query use case ("further reading"), then expand based on demonstrated value.

## 10. The Demo-as-KG-Consumer Architecture (Concrete Vision)

Taking the demo as the driving example, here is how a shared KG transforms the architecture:

### Current state (hardcoded knowledge):
```
generate.py  ──builds──►  data.json (70KB hardcoded content strings)
                              │
index.html   ◄──reads─────┘   (fetches, renders)
deep_*.html                   (separate copies of same knowledge)
lit_review.md                 (independent, no machine interface)
math_bridge/                  (independent, no machine interface)
```

Every knowledge element (definition, reference, method description, concept) is copied into the place where it's displayed. Changing a definition means editing generate.py, regenerating data.json, and pushing. There is no way for the lit review to "feed" the demo or vice versa.

### Target state (KG as single source of truth):
```
                    ┌─────────────────────────────┐
                    │     causal_kg.json           │
                    │  (versioned in git,          │
                    │   validated in CI)           │
                    │                              │
                    │  Concepts (80+)              │
                    │  References (160+)           │
                    │  Methods (20+)               │
                    │  Relationships (500+ edges)  │
                    │  Gallery cases (6+)          │
                    │  Design principles (6)       │
                    │  Math bridge levels (6)      │
                    └──────┬──────────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                   ▼
    generate.py      index.html          lit_review.md
    (reads KG for    (fetches KG,        (exports to KG
     definitions,     queries for         on update)
     inserts into     "further reading",
     data.json)       populates glossary)
          │                │
          ▼                ▼
    data.json         Demo renders
    (dynamic refs)    KG-powered
                      tooltips, links,
                      concept maps
```

### What changes in the demo:

**1. Glossary → KG query.** Instead of 17 hardcoded terms in generate.py, the demo fetches `causal_kg.json` and renders all Concept nodes with `show_in_glossary: true`. Adding a new glossary term means adding one node to the KG — no code change.

**2. "Further reading" → KG query.** When the demo mentions "back-door criterion," it queries `back-door-criterion -[PROVEN_BY]-> Reference` and renders linked references. No manual curation per mention.

**3. Concept tooltips → KG lookup.** The `applyTooltips()` function currently uses a hardcoded `decomposition_tooltips` dict in generate.py. With a KG, every concept node has a `definition` and `short_definition` field. The tooltip system queries the KG instead of a manual dict.

**4. Cross-references → KG traversal.** Currently, the demo can't answer "what methods relate to this concept?" With a KG: `concept -[USED_BY]-> method -[APPLIED_IN]-> gallery_case`.

**5. Multi-project reuse.** The same `causal_kg.json` could power:
- This demo (browser-based causal walkthrough)
- A Jupyter notebook extension (KG-powered context for code cells)
- A literature review dashboard (queryable reference graph)
- An LLM context provider (ground causal answers in curated knowledge)

### Concrete first deliverable:

```json
// causal_kg.json — MVP (~100 nodes, served alongside demo)
{
  "concepts": [
    {
      "id": "back-door-criterion",
      "name": "Back-Door Criterion",
      "short_def": "Graphical test for which variables to adjust for",
      "definition": "A set Z satisfies the back-door criterion relative to (T,Y) if...",
      "rung": 2,
      "glossary": true,
      "proven_by": ["ref:pearl-1995"],
      "related_to": ["d-separation", "adjustment-set", "confounding"],
      "used_by_methods": ["aipw", "ipw", "tmle"]
    }
  ],
  "references": [
    {
      "id": "ref:pearl-1995",
      "short": "Pearl (1995)",
      "full": "Pearl, J. (1995). Causal diagrams for empirical research. Biometrika, 82(4), 669-688.",
      "doi": "10.1093/biomet/82.4.669",
      "proves": ["back-door-criterion", "d-separation"],
      "year": 1995
    }
  ]
}
```

The demo's `generate.py` would add ~10 lines:
```python
# Load KG for dynamic definitions
kg = json.loads(Path("docs/demo/causal_kg.json").read_text())
data["kg"] = kg  # pass through to demo
```

And `index.html` would query it for glossary, tooltips, and cross-references instead of using hardcoded strings.
