// Neo4j schema reference — automatically applied by src/graph_builder.py on startup.
// Run manually:  cat scripts/init_neo4j.cypher | cypher-shell -u neo4j -p opinion-graph

CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT event_id IF NOT EXISTS
FOR (e:Event) REQUIRE e.event_id IS UNIQUE;

CREATE INDEX event_sentiment IF NOT EXISTS FOR (e:Event) ON (e.sentiment);
CREATE INDEX event_timestamp IF NOT EXISTS FOR (e:Event) ON (e.timestamp);

// Example multi-hop query: withdrawal-related negative events and timeline
// MATCH (seed:Entity)<-[sr:INVOLVES]-(se:Event)
// WHERE seed.name = '提现' OR sr.aspect = 'withdrawal' OR sr.aspect_label = '提现'
//    OR toLower(se.label) CONTAINS toLower('提现')
// WITH collect(DISTINCT seed) AS seeds
// UNWIND seeds AS s
// MATCH path = (s)<-[:INVOLVES*1..2]-(ev:Event)
// WHERE ev.sentiment < -0.15
// RETURN ev.label, ev.timestamp, ev.sentiment, [n IN nodes(path) | coalesce(n.name, n.label)]
// ORDER BY ev.timestamp DESC LIMIT 50;
