# AI-GlobalSignalGrid
Global Signal Grid (GSG) is an on-demand AI system that detects top geopolitical, economic, and cultural hotspots using GPT-4 and builds multilingual Google News RSS feeds. Powered by FastAPI, LangGraph, and async pipelines, it returns structured JSON signals for global situational awareness. No DB, no scheduler fully agentic, modular, and fast.


MASX Agentic AI System Design
MASX Agentic AI is a modular, multi-agent system for geopolitical intelligence gathering. It emphasizes
deterministic execution (all LLM calls use temperature=0 ) and collaborative reasoning among
specialized agents. The system orchestrates a daily pipeline that pulls news from Google News RSS and
global events from GDELT 2.0, analyzes and clusters them into “hotspot” topics, and stores structured
results (with embeddings) and logs in a Supabase (Postgres + pgvector) database. This design uses
LangGraph for directed control flow, AutoGen for dynamic agent collaboration, and CrewAI principles for
role-based task delegation.
 Architecture Diagram
The MASX Agentic AI architecture consists of a central Orchestrator and a team of specialized agents (a
“crew”). The Orchestrator (built with LangGraph) defines a directed graph of steps and coordinates the
agents through a reason→plan→act→reflect loop. Each agent is an autonomous unit with a specific role
and tools, and all interactions produce structured JSON outputs for reliability. The diagram below illustrates
the main components and data flows:
flowchart TD
 subgraph Orchestrator [Orchestrator (LangGraph Directed Graph)]
 direction TB
 A1(Reason & Plan<br/>- determine focus,<br/> prepare queries)
 A2(Act<br/>- delegate to fetching/analysis tools)
 A3(Reflect<br/>- check results,<br/> update plan if needed)
 end
 A1 -->|Invoke| B[Domain Classifier]
 A1 -->|Invoke| C[Query Planner]
 C --> D[News Fetcher]
 C --> E[Event Fetcher]
 D & E --> F[Merge & Deduplicator]
 F --> G[Language Resolver]
 G -->|if non-English| H[Translator]
 H --> G
 G --> I[Entity Extractor]
 I --> J[Event Analyzer]
 J --> K[Fact-Checker]
 K --> L[Validator]
 L --> M[Memory Manager]
 M --> N[Supabase DB<br/>(Postgres + pgvector)]
 subgraph Logging/Auditor
 O1(Track steps & data)
 O2(Validate JSON schemas)
1
 O3(Monitor agent outputs)
 end
 O1 & O2 & O3 --> Orchestrator
 M -->|Embed vectors| N
 N -->|Store logs & results| Logging/Auditor
Figure: System architecture and data flow. The Orchestrator uses a LangGraph-defined workflow to call
each agent in sequence (some in parallel where possible). For example, the Orchestrator first calls Domain
Classifier and Query Planner (Reason/Plan phase), then triggers News Fetcher and Event Fetcher to
gather data (Act phase). The results are merged, translated if needed, and analyzed by subsequent agents
(Event Analyzer, etc.), with validation and fact-checking before storing results. Throughout, a Logging/
Auditor component records each step and enforces schema compliance. This directed graph approach
ensures a controlled loop where the agent team iteratively makes decisions and actions until completion
. The LangGraph framework allows defining such workflows with conditional transitions and
feedback loops (e.g. after taking an action, the agent can loop back to reasoning) , ensuring the
system can reason, act, and then reflect on results in a deterministic loop.
 Agent Types & Roles
MASX Agentic AI employs a CrewAI-style team of agents, each with a clear role, specific allowed tools,
memory access level, and delegation behavior. This division of labor follows CrewAI’s principle of assigning
specialized roles to agents to tackle complex tasks collectively . Below we describe each agent’s identity
and responsibilities:
Domain Classifier – Role: Identifies the broad domain or category of content to focus on (e.g.
political conflict, health crisis, economic event). It may classify incoming cues or previous outputs to
set context (for instance, determining if today’s run should emphasize security, economy, etc.). Tools:
Primarily an LLM prompt for classification (with a predefined label schema) at temperature 0.
Memory Access: Can read recent context or past domain outputs (short-term JSON state) to inform
classification. Delegation: Minimal – outputs a domain label or tags which the Orchestrator uses to
guide the Query Planner. Does not call sub-agents itself.
Query Planner – Role: Formulates search queries and filters for data sources based on the domain
and any known context. It decides what to ask Google News and GDELT. For example, if Domain is
“Political Unrest,” it might generate keyword queries (“protest, coup, unrest location X”) and GDELT
filters (date range = last 24h, relevant themes). Tools: An LLM used to generate or refine queries
(ensuring they follow Google News RSS syntax or GDELT API format) ; also can perform vector
database lookups to avoid repeating recent queries (e.g., retrieving past hotspot embeddings to
plan new queries). Memory Access: Read access to long-term memory (via the Memory Manager) to
see if similar events were recently handled (helps avoid duplicative queries). Delegation: If uncertain
about a query, it could invoke an AutoGen conversation with the Domain Classifier or Entity Extractor
(e.g., “Do these keywords sufficiently cover the domain?”). Normally, it outputs a set of queries/
parameters which the Orchestrator then passes to fetcher agents.
News Fetcher – Role: Gathers news articles from Google News RSS feeds based on the Query
Planner’s output. It automates what the existing google_rss_generator.py did, but as an agent
1 2
3 4
2
•
•
5
•
2
subtask. Tools: Uses HTTP requests to Google News RSS endpoints (e.g., the news.google.com/
rss/search?q=<query> API). Google’s RSS feed can return up to 100 results per query ,
including title, snippet, URL, source, and timestamp. A lightweight RSS parser is used to extract
articles. Memory Access: None (it simply fetches fresh data). Delegation: If a feed fails or yields zero
results, it could notify the Orchestrator (which might adjust the query or proceed regardless). It
doesn’t invoke other agents itself – it just returns the list of articles found.
Event Fetcher – Role: Integrates GDELT 2.0 events as a parallel news source. It queries the GDELT
Doc API for events/news matching our criteria (e.g., last day’s events globally, filtered by themes or
keywords from Query Planner). Tools: Uses the gdeltdoc Python client to perform an
article_search with specified filters. For example, it might use filters for date range = today,
language = all or specific, and keywords/entities of interest. The GDELT Doc API returns a list of news
articles as a DataFrame with fields like URL, title, publication date (“seendate”), source domain,
language, etc. . Memory Access: None (fetches live data). Delegation: If certain specialized data is
needed (e.g., a timeline of volume or tone), Event Fetcher could use GDELT’s timeline APIs as a
tool. Generally, it returns raw event articles. Like News Fetcher, it doesn’t call other agents itself.
Merge/Deduplicator – Role: Combines and cleans the results from multiple sources (Google News
and GDELT). It identifies duplicate or highly similar articles/events and merges them to avoid
redundancy. Tools: Algorithmic text similarity checks (e.g., comparing URLs, titles, or using content
embeddings for similarity). It may use a fuzzy matching library or even vector similarity via pgvector
to cluster near-duplicates. Memory Access: Can retrieve recent vectors of past news to ensure new
items aren’t repeats of yesterday’s (if necessary). Delegation: Purely a utility agent – it doesn’t invoke
others. It produces a unified list of unique articles/events for analysis.
Language Resolver – Role: Determines if any fetched content is in a foreign language and routes it
for translation. It scans article metadata (or content, if available) for language codes. For example,
GDELT provides a language field for each article , and RSS feeds might require language
detection on titles. Tools: Could use a simple language detection library or rely on metadata. Memory
Access: Not needed (operates on current batch). Delegation: For any non-English text, it delegates by
invoking the Translator agent, then receives the translated text and replaces the original content
with it (or attaches an English version for downstream agents). It ensures all subsequent analysis
sees English text. If content is already English, it simply passes data along.
Translator – Role: Translates non-English news content into English. This agent is only engaged via
the Language Resolver when needed. Tools: An external translation API (like Google Translate) or an
LLM capable of translation. For deterministic output, a rule-based translator or a high-temperature
LLM is avoided – instead we use a reliable service or a multilingual model at temp=0 with a strict
prompt to output translated text. Memory Access: Possibly none (translates on the fly), though if
similar text was translated recently, it could consult memory to reuse translations. Delegation: Does
not call other agents. It returns the English text (with original reference) to the Language Resolver.
Entity Extractor – Role: Extracts key entities (people, organizations, locations, etc.) from the news
content. After translation, this agent processes article titles or full text to identify entities for each
article/event. These entities will help in clustering and reasoning about the events. Tools: Could use
an NLP library (like spaCy’s NER) for full determinism, or an LLM prompt that returns a JSON list of
entities found (with a fixed schema). Given the need for structured output, an LLM at temp=0 with a
6
•
7
8
9
•
•
8
•
•
3
well-defined function (e.g., “extract_entities”) is feasible. Memory Access: Can optionally cross-check
against a knowledge base (vector search) if needed (e.g., to link entities to known identities), but
primarily it works on the provided text. Delegation: If the text is very large or complex, it might ask a
sub-agent (like a Summarizer agent, not explicitly listed) to condense the text before extraction –
however, typically it handles extraction directly. It outputs an enriched data structure: for each
article, attached entities.
Event Analyzer – Role: Analyzes the merged, translated, entity-tagged news set to identify
“hotspots” – clusters of related articles that likely correspond to a single real-world event or theme. It
essentially converts raw articles into structured events or topics. This may involve clustering by
similarity (e.g., articles that share many entities or keywords) and summarizing each cluster. Tools: An
LLM is utilized here for higher-level reasoning – e.g., to label clusters or summarize what the cluster
is about. The Event Analyzer might use a prompt like: “Group these articles into events and provide a
short summary of each event, in JSON.” The LLM’s output is schema-enforced (list of events with fields
like event_name , article_urls , summary ). Because the LLM is constrained to JSON format,
Pydantic will validate that the output has, say, a list of clusters each containing the collected URLs.
Memory Access: Read/write access to vector memory – for example, it can store each cluster’s
embedding (by averaging member article embeddings) to represent “hotspot nodes” for future
similarity searches. It might also retrieve historical clusters from memory to see if this event has
occurred before or relates to ongoing narratives. Delegation: If the Event Analyzer is unsure about an
event (say conflicting info in articles), it can engage the Fact-Checker agent via an AutoGen multiturn dialog (asking a question like “These two reports differ on casualty numbers – which is correct?”).
The Event Analyzer then integrates that feedback into its summary. Primarily, it produces the
hotspot list: e.g., [{"hotspot": "<event description>", "articles": [URL1,
URL2, ...], ...}, ...] .
Fact-Checker – Role: Verifies critical facts and consistency across sources for the identified events. It
acts as a failsafe to reduce misinformation. When invoked (either by Event Analyzer or Orchestrator
if a result looks suspicious), the Fact-Checker will cross-verify details. Tools: It may use an LLM with a
specialized prompt to compare statements, or even call an external tool (like a search engine or a
database of known facts) if available. For instance, it could run a quick search (if online access is
permitted in cloud deployment) to see if other reputable outlets corroborate a claim. Alternatively, it
queries the Supabase vector memory to find similar events or past knowledge. Memory Access: Read
access to past embeddings and data (to see if the same event was recorded with different facts).
Delegation: It can’t delegate further (no deeper agent hierarchy); it returns either a validation (e.g.,
“facts consistent”) or a correction annotation (e.g., “Article X’s number looks like an outlier compared
to others”). The Orchestrator or Event Analyzer can use this info to flag or correct the final output
(e.g., exclude an unreliable source or add a note).
Validator – Role: Ensures the final data output conforms to the expected schema and quality
standards. It checks that each hotspot has the required fields (e.g., name, list of URLs, etc.), that all
URLs are valid and not broken (it might do a quick HTTP HEAD request for status), and that the JSON
is properly structured. It also ensures all LLM outputs were parsed correctly and all required steps
ran. Tools: Pydantic (or similar JSON schema validators) to enforce structure. It may also include
simple business rules (e.g., ensure no hotspot is empty, no duplicate URL across hotspots after
merging, etc.). Memory Access: Possibly logs some metadata, but mainly works on the current result.
•
•
•
4
Delegation: Does not call other agents; it either passes a clean bill or raises an exception for any issue
(which could trigger the Orchestrator to adjust or re-run certain parts).
Memory Manager – Role: Handles all interactions with the Supabase Postgres database and the
pgvector long-term memory. It is responsible for storing new data (hotspots, articles, embeddings)
and retrieving relevant historical data for agents that request it. Essentially, it abstracts the database
operations so that other agents don’t directly query the DB. Tools: Supabase Python client or REST
interface, specifically using the pgvector extension for similarity search. For example, it can take an
embedding (like an event summary embedding) and find the nearest past events in vector space (to
detect if this “hotspot” has happened before or is related to an ongoing thread). Supabase’s toolkit
allows storing and querying embeddings at scale , and the Memory Manager ensures these
operations are done efficiently and securely. Memory Access: Full access (read/write) to the long-term
memory store. Delegation: It doesn’t delegate to other agents, but serves requests. If the vector
search yields ambiguous results, it might rank them or apply thresholds. It returns data to the
requester agent or orchestrator in JSON (e.g., list of similar events or confirmation of save).
Logging/Auditor – Role: Maintains a detailed audit log of every step in the pipeline and enforces
compliance with any constraints (such as content or security policies). It tracks agent invocations,
tool usage, outputs, execution time, and errors. This agent (or component) has a dual function:
logging and monitoring. Tools: Logging library to write structured logs (JSON lines) for each action,
and simple rule-checkers to ensure no agent output violates guidelines (e.g., if an LLM somehow
returned unstructured text or sensitive data, the auditor would flag it). It might also utilize any builtin moderation APIs of the LLM to filter content. Memory Access: Write access to a logging table in the
database (or a log file). It can also read past logs for debugging or pattern detection (like noticing if a
certain agent frequently fails). Delegation: Does not call others (except possibly sending alerts to a
human operator if something critical occurs). It runs in the background or after each agent step to
append the log. The Orchestrator might treat it as an observer agent that wraps each step execution.
Each agent is thus constrained to specific tools and data. This separation follows the CrewAI paradigm,
where a crew of agents coordinate via structured workflows, each contributing their expertise . Interagent communication is enabled but controlled – e.g. via the Orchestrator or using AutoGen “conversations”
when needed. All LLM interactions yield JSON outputs that adhere to predefined Pydantic schemas,
ensuring deterministic parsing and easing inter-agent data passing.
🛠 Tools per Agent
Each agent has access to a limited set of tools or external resources to perform its tasks. The table below
summarizes the key tools/technologies each agent uses or calls:
Agent Key Tools & APIs
Domain
Classifier
OpenAI GPT-4 (or similar LLM) for text classification (system prompt with domain
categories, output as JSON).
Query Planner OpenAI GPT-4 for query generation (guided prompt templates); Supabase PGVector
for retrieving related past event embeddings (read-only).
•
10
•
11 12
5
Agent Key Tools & APIs
News Fetcher
Google News RSS feed (HTTP GET to news.google.com/rss endpoints) – uses
queries with date filters etc. ; RSS parser library.
Event Fetcher
GDELT Doc API via gdeltdoc Python client for article search and timeline data;
handles API keys and rate limits internally.
Merge/
Deduplicator
Python text processing libraries (e.g. difflib or fuzzy matching) for title similarity;
optional use of vector embeddings (via Memory Manager) to cluster near-duplicates.
Language
Resolver
langdetect or similar lightweight language detection library; simple rule engine to
decide if translation is needed based on language codes.
Translator External translation API (Google Translate or DeepL) for high-accuracy deterministic
translation; or a local multilingual model (e.g., Helsinki NLP opus MT) if on-prem.
Entity
Extractor
spaCy NLP model for Named Entity Recognition (ensures consistency in extraction);
alternatively GPT-4 with a function call for extract_entities (returns entities in
JSON).
Event Analyzer
OpenAI GPT-4 (zero-temp) for analyzing and summarizing clusters (complex reasoning
prompt with few-shot examples of clustering); possibly uses k-means or hierarchical
clustering on embeddings as a pre-step before LLM summarization.
Fact-Checker
OpenAI GPT-4 for reasoning and cross-checking (prompted with conflicting
statements to analyze); optional web search tool (if allowed) via an API or a knowledge
base query; Supabase vector similarity search to find corroborating documents in
memory.
Validator Pydantic or JSON Schema for output validation; Python requests for URL health
checks; simple consistency checks (e.g., date formats, no missing fields).
Memory
Manager
Supabase Python client / PostgREST API – to INSERT new entries (hotspots, article
metadata, embeddings) and SELECT for vector search (utilizing pgvector
similarity queries) .
Logging/
Auditor
Standard logging library (structured JSON logs); LangGraph’s built-in moderation
hooks (to catch if an agent’s output deviated or contained disallowed content) ;
potentially OpenAI content filter API for safety.
Each agent’s tools are isolated – for example, the News Fetcher cannot directly call the database or LLM, it
only fetches RSS feeds. This principle of least privilege in tool access is a security feature and also makes the
system modular: you can adjust or replace tools for one agent without affecting others. For instance, if
moving on-prem with no internet, the News Fetcher’s RSS tool could be replaced by an internal news feed
reader, while the rest of the agents remain the same.
 Multi-Agent Workflow Examples
The MASX system supports multiple reusable workflows composed of the above agents. Here we describe
three key orchestrations (flows) that demonstrate how agents collaborate in different scenarios. Each flow is
5
7
10
13
6
implemented as a directed subgraph of the LangGraph orchestrator, ensuring the sequence and conditional
logic are deterministic.
Flow Example 1: Summarize → Reason → Query → Generate → Store
Summarize: At the start of a daily run, the Orchestrator triggers a summarization of context. For
instance, it may ask the Memory Manager for a brief summary of yesterday’s hotspots or retrieve the
top events from the past week (via a vector query), then have the Domain Classifier or an LLM agent
summarize the trending themes. This summary (e.g., “Tensions increasing in Region X; Economic
crisis in Country Y…”) sets the stage.
Reason: Using that context, the Orchestrator (or a dedicated reasoning agent) decides what the
focus of this run should be. It might reason about which domains or regions require attention today.
For example, if the summary indicated rising tensions in Region X, the system “reasons” that
geopolitical conflict is a key domain for today. This could be an LLM step that outputs a rationale and
selected focus (structured, e.g., {"focus_domain": "Conflict", "priority_regions":
["Region X","Region Y"]} ).
Query: Given the focus, the Query Planner generates precise queries. For each priority region or
topic, it formulates Google News RSS queries (e.g., q="Region X conflict when:1d" ) and
GDELT filters (e.g., keyword = Region X, theme = Conflict, last 24h). These queries are deterministic
and reproducible. The Orchestrator then calls the News Fetcher and Event Fetcher in parallel (if
supported) with these queries.
Generate: Once data is fetched, the Event Analyzer takes over to generate structured results. It
clusters articles into events and generates summaries for each cluster (using LLM reasoning).
Essentially, it “generates” the final hotspot nodes (with titles and associated URLs). For example, it
might output: Hotspot 1: “Coup in Country Z” with 5 related article URLs, Hotspot 2: “Border clashes in
Region X” with 3 URLs, etc. This generation is guided by the previous reasoning context (so it knows
these are conflict events and frames them accordingly).
Store: The Memory Manager then stores the results: each hotspot node is saved as a row in the
database (with fields like date, hotspot title, article list, summary text, etc.), and important fields are
embedded into vectors (e.g., concatenation of title and summary is embedded via OpenAI or local
model) and stored in a pgvector column for future retrieval. The system also stores execution
metadata (timestamp, domain focus, number of articles, any errors) in a log table. The output for the
day is effectively a list of article URLs per hotspot, now in the database for analysts or downstream
systems to use.
(This flow illustrates the core daily pipeline: the system summarizes context, reasons a plan, queries sources,
generates structured intelligence, and stores it. It ensures continuity by referencing prior days (via memory) and
focusing on the most relevant areas.)
Flow Example 2: Detect → Classify → Ask Subagent → Respond → Visualize
Detect: In this scenario, the system detects something during its run that triggers a special
handling. For example, suppose the Event Analyzer notices an anomaly – one cluster has a drastically
1.
2.
3.
4.
5.
1.
7
different tone or a single source claim. This could be a detection of potential misinformation or
simply an uncertainty (ambiguity) in the data. The Orchestrator can flag this and initiate a sub-flow.
Classify: The Orchestrator delegates the flagged issue to an appropriate agent for classification. For
instance, if the issue is a suspicious claim in an article, the Domain Classifier (or a similar classifier
agent) might classify the type of anomaly: is it a factual discrepancy? sensationalism? If it’s languagerelated (e.g., an article in an unknown language slipped in), the Language Resolver classifies which
language. Essentially, this step identifies what kind of follow-up is needed.
Ask Subagent: Based on the classification, the Orchestrator invokes a specialized sub-agent to
handle it. Using AutoGen’s dynamic agent chat capabilities, it can spawn an on-the-fly conversation
between, say, the Event Analyzer and the Fact-Checker. For example, the Event Analyzer might send a
message: “Agent FactChecker, article X reports 1000 troops, while others report 100. Can you verify which
is likely correct?” The Fact-Checker agent, using its tools, might do a quick cross-source analysis and
respond with a reasoned answer. This multi-turn dialog continues until the ambiguity is resolved. (If
the issue was language, a Translator sub-agent would be asked to translate, etc.)
Respond: The sub-agent (Fact-Checker in this case) responds with a result: e.g., “Verified: likely 100 is
correct – Article X appears unreliable or a typo.” The Event Analyzer receives this and updates the event
data (perhaps marking Article X as unverified or adjusting the summary to note the correction). In an
AutoGen framework, these agents exchange messages in JSON or a controlled format until the
querying agent is satisfied . The Orchestrator ensures this side-conversation doesn’t stall or
stray off course (with timeouts or step limits).
Visualize: Finally, the outcome is integrated into the system’s output. “Visualize” here means
presenting or utilizing the result in a human-friendly way. For example, the Logging/Auditor might
log that a fact-check occurred and tag the hotspot with a warning. Or, if this were an interactive
session, the system could output a chart or text highlighting the differences. In our pipeline (which
has no UI), visualize could equate to structuring the final JSON such that it’s easy to see the resolution
(perhaps adding a field like "verification": "Article X corrected" ). If there were a
reporting dashboard consuming the DB, this info would be clearly shown (like a special flag on that
hotspot).
(This flow shows the system’s ability to dynamically react to uncertainties: it detects an issue, classifies it, engages
a sub-agent in a focused dialogue to resolve the ambiguity, and then incorporates the answer back. The use of
AutoGen conversable agents allows such on-demand collaboration between roles .)
Flow Example 3: Trigger → Delegate → Fetch Tool Output → Update Plan → Re-Execute
Trigger: This flow demonstrates an internal loop for iterative refinement or scheduled re-runs. The
trigger could be time-based (the internal scheduler signaling the start of a daily run at midnight) or
event-based (e.g., an agent finds incomplete data and triggers a second pass). Since we avoid
external cron/Airflow, the Orchestrator itself contains a scheduling loop – for example, after finishing
a run, it schedules the next one by sleeping until the next day or by maintaining an internal clock.
Triggers can also occur mid-run: e.g., the Validator might trigger a re-run if validation fails.
2.
3.
4.
14 15
5.
16 17
1.
8
Delegate: On trigger, the Orchestrator delegates tasks to the agents according to the defined
workflow graph. In a scheduled run, it simply kicks off the sequence from Domain Classifier onward
(as in Flow 1). In a mid-run correction scenario, the Orchestrator might delegate a specific subset of
tasks. For instance, if validation found a missing field in one hotspot, the Orchestrator can delegate a
sub-plan: maybe call the Event Analyzer again for that cluster or ask the Query Planner to fetch
additional articles to fill the gap. LangGraph enables branching logic, so the Orchestrator can decide
which branch (subgraph) to run based on the state .
Fetch Tool Output: The agents perform their actions and fetch results from tools as usual. The
Orchestrator collects these outputs. Importantly, this step emphasizes that after delegation, the
Orchestrator waits for and gathers outputs from tools or APIs. For instance, if this flow was triggered
by a need to get more data on an ongoing crisis, the Orchestrator delegates to News/Event Fetchers
with a narrower query (maybe a specific location), then fetches their outputs (article lists). The
Orchestrator monitors these calls – if an API fails or returns empty, it notes that. The Logging/
Auditor logs each tool call with status (success/failure, records fetched, etc.).
Update Plan: After receiving tool outputs, the Orchestrator (or an agent like the Query Planner)
updates the plan. This is the reflect part of the loop – analyzing whether the goal is met or more
steps are needed . For example, if the additional articles fetched still don’t have a needed
detail, the plan might be updated to call the Fact-Checker or even to adjust the query and try again.
The state (short-term JSON) is updated: e.g., state["fetched_articles"] = N . If N is below a
threshold, maybe plan to broaden the query. The Orchestrator can modify its graph path accordingly
(LangGraph allows conditional edges and looping back to earlier nodes based on such conditions
).
Re-Execute: With a new plan in place, the Orchestrator executes the necessary steps. This could
mean looping back to a previous stage in the LangGraph. For example, the graph might loop from
the “Reflect” node back to the “Query” node if state["needs_more"] is true, thereby re-invoking
the Query Planner with a revised prompt (perhaps asking for different keywords). All LLM calls
remain deterministic, so re-execution with the same state yields the same result – but here the state
has changed due to the updated plan, so the loop gradually converges to a satisfactory result. Once
the Validator approves the output, the loop ends. The final data is then stored (or updated in DB if it
was a mid-run fix). The internal scheduler then sets the next trigger if this was a daily run. All
intermediate steps have been logged, so one can trace the plan updates in the logs (each re-plan
event recorded with reason).
(This flow highlights the system’s resilience and autonomy: it can trigger itself on schedule, delegate tasks to
agents, incorporate tool outputs, adjust its plan, and re-run parts of the workflow without human intervention.
This is crucial for daily reliable operation – if anything goes wrong or data is incomplete, the system self-corrects
in a controlled, deterministic manner rather than silently failing.)
2.
18 19
3.
4.
4 20
3
5.
9
 Security Strategies
Building a production-grade agent system requires robust security and safety measures. MASX Agentic AI
integrates several strategies to ensure reliable and safe execution:
Schema-Enforced Outputs: Every LLM agent is prompted to return data in a specific JSON schema.
We use Pydantic (or similar) to validate each output against the expected format. This prevents
prompt injections or model hallucinations from corrupting the process – if the output isn’t valid JSON
or misses fields, the Validator (or Logging/Auditor) catches it and the Orchestrator can correct
course. By strictly requiring JSON, we reduce ambiguity and ensure deterministic parsing. For
example, the Event Analyzer’s prompt explicitly says: “Respond ONLY in this JSON schema: …” and any
deviation is treated as an error.
Role-Based Access Control: Each agent is only allowed to perform tasks within its role and use its
designated tools. The Orchestrator (and LangGraph’s design) acts as a controller that prevents an
agent from arbitrarily calling tools it shouldn’t. For instance, the Translator cannot directly write to
the database, and the Query Planner cannot fetch news by itself – they must communicate through
the orchestrated channels. This containment is a form of access control that minimizes the impact of
any single agent’s failure or misbehavior. If an agent tries to go “off-script” (which could only happen
if an LLM deviated from instructions), LangGraph would not have a transition for that, effectively
halting the process – a fail-safe to prevent cascading errors.
Moderation and Content Filtering: We incorporate prompt filtering and moderation at multiple
levels. The Logging/Auditor agent monitors outputs for any disallowed content. Additionally, we can
utilize the LLM’s own content filter or OpenAI’s moderation API on any user-facing text. While this
system isn’t user-interactive (no direct user prompts), it still handles external data (news content)
which might be sensitive or malicious. The system sanitizes inputs (e.g., stripping HTML from RSS
feeds, removing scripts in text) and ensures that when the LLM sees text, it’s just plain news content.
LangGraph also allows adding moderation nodes – e.g., we could have a node that checks the LLM’s
proposed action before execution . We have configured such quality-control loops to prevent
agents from “veering off course” . For example, before executing a tool call generated by an
LLM, the Orchestrator can verify the tool name and parameters against an allowlist.
Deterministic Execution & Reproducibility: Security also means reliability. With temperature=0
for all LLM calls, the system avoids nondeterministic variability. The same inputs will produce the
same outputs, which is crucial for an auditable intelligence pipeline. We log random seeds for any
non-LLM processes if applicable. Each daily run is traceable: given the logs and stored intermediate
data, one can reproduce the flow. This makes it easier to identify and fix issues (a form of
auditability).
Data Validation and Sanitization: The Validator agent not only checks schema but also sanitizes
data going into storage. For instance, it ensures that strings are UTF-8 and free of control characters,
dates are in standard format, and arrays of URLs contain valid URL strings. This prevents malicious
data from propagating. In addition, before inserting into the database, the Memory Manager uses
parameterized queries to avoid SQL injection (though in our case data is not user-provided, it’s still a
good practice). By the time data reaches long-term memory, it’s been through multiple checks.
•
•
•
13
21 22
•
•
10
Prompt Guardrails: All system and agent prompts are crafted to minimize risky outputs. Agents are
instructed on what not to do (e.g., the Fact-Checker’s prompt might say “do not make up facts, only
use given sources or say you are uncertain”). CrewAI’s approach to agent backstories and goals can
also be used to constrain behavior (for example, giving each agent a clear persona and scope to
prevent it from drifting beyond its duty). The Orchestrator’s instructions via system prompts ensure
each LLM knows it must follow the plan and format.
Credential & API Security: The system uses environment variables for API keys (e.g., OpenAI key,
Supabase service key, GDELT API key if needed) and the Memory Manager or tools load these
securely. No keys are hard-coded. In on-prem deployments, these can be managed via vaults or
secure config files. Also, external API usage is minimal (just news and translation); if those fail or
return unexpected results, the system handles it gracefully (logging the failure, possibly retrying
with backoff, but never exposing keys or crashing uncontrolled).
Human Oversight Hooks: While the system is autonomous, we include hooks for human review.
The Logging/Auditor could, for example, email a report if a major anomaly is detected or if the FactChecker had to correct something significant. This ensures that if something truly novel or
dangerous is found (say a breaking crisis event), a human analyst can be alerted to verify the AI’s
conclusions. The architecture supports a “human-in-the-loop” mode where certain agent outputs
require approval before proceeding – LangGraph’s human approval nodes could be toggled on
as needed in high-stakes deployments.
In summary, MASX Agentic AI’s security comes from structured outputs, strict role separation,
moderated LLM interactions, and thorough logging. By validating at each step and keeping agents
constrained, we maintain trust in the system’s autonomy.
📊 Logging Format
The system produces extensive logs in JSON format for each step, which are stored in the database (or a
file) for traceability. Each log entry captures the who, what, when, and result of an action. We define a
consistent schema for log records, for example:
{
"run_id": "2025-07-04T0000Z",
"timestamp": "2025-07-04T00:15:30Z",
"agent": "News Fetcher",
"action": "fetch_rss",
"parameters": {"query": "Region+X+unrest+when:1d"},
"result": {"articles_fetched": 42},
"status": "success",
"message": "Fetched 42 articles from Google News RSS for query 'Region X
unrest'."
}
run_id ties logs to a particular daily run (could be a timestamp or an incrementing ID).
•
•
•
23
•
11
timestamp is the exact time of the step.
agent and action describe which agent performed what. Agents have predefined action names
(e.g., DomainClassifier.classify_domain , EventFetcher.fetch_articles , etc.) making it
easy to filter logs.
parameters captures inputs to that action – in the above, the query used for RSS fetch. For an LLM
agent, this might include a short descriptor of the prompt or relevant state (but we avoid full prompt
text to keep logs concise; however, we can log a prompt hash or ID for reference).
result gives key output metrics. We typically log counts or IDs rather than full content. For
example, the Fact-Checker might log {"verdict": "corrected", "reason": "source X had
a typo"} . Or the Event Analyzer could log {"hotspots_identified": 5} .
status indicates success/failure of the action. On failure, an error field would appear with an
error message or stack trace, and potentially the Orchestrator would log a follow-up entry for the
retry action.
message is a human-readable summary for quick scanning (optional but helpful for auditors).
Logs are written step-by-step. The Logging/Auditor agent appends an entry after each agent does its work.
In addition, the Orchestrator logs high-level events like “Daily run started” and “Daily run completed in X
seconds” with run-level info (like total articles processed, number of hotspots, etc.).
We also maintain a pipeline execution log that summarizes each run in a single entry, for reporting. For
example:
{
"run_id": "2025-07-04T0000Z",
"date": "2025-07-04",
"domains": ["Conflict"],
"hotspot_count": 5,
"articles_count": 123,
"duration_minutes": 4.2,
"errors": 0
}
This is stored in a runs table, whereas detailed step logs go into a logs table. By storing logs in
Postgres (Supabase), we can query them easily (e.g., find how often Fact-Checker corrected something, or
trend the number of hotspots per day). The use of JSON fields in Postgres enables flexible queries on
structured log data.
All logs include timestamps, and ordering is guaranteed by the sequence of the Orchestrator’s execution
(which is single-threaded per run for deterministic behavior). If parallel fetching is done, we still log those
fetches as separate entries with their timestamps.
Sample Log Sequence (abbreviated):
[
{
•
•
•
•
•
•
12
"run_id": "2025-07-04T0000Z", "timestamp": "00:00:00Z",
"agent": "Orchestrator", "action": "start_run", "parameters": {"date":
"2025-07-04"}, "status": "success"
},
{
"run_id": "2025-07-04T0000Z", "timestamp": "00:00:02Z",
"agent": "Domain Classifier", "action": "classify_domain",
"parameters": {"input_summary": "..."},
"result": {"domain": "Conflict"}, "status": "success"
},
{
"agent": "Query Planner", "action": "generate_queries", "parameters":
{"domain": "Conflict"},
"result": {"queries": ["Region+X+unrest when:1d", ...]}, "status":
"success", "timestamp": "00:00:04Z", "run_id": "2025-07-04T0000Z"
},
...
{
"agent": "Validator", "action": "validate_output", "result": {"valid":
true}, "status": "success", "timestamp": "00:04:10Z", "run_id":
"2025-07-04T0000Z"
},
{
"agent": "Memory Manager", "action": "store_results", "parameters":
{"hotspots": 5, "articles": 123}, "status": "success", "timestamp":
"00:04:15Z", "run_id": "2025-07-04T0000Z"
},
{
"agent": "Orchestrator", "action": "end_run", "result": {"duration_sec":
255}, "status": "success", "timestamp": "00:04:15Z", "run_id":
"2025-07-04T0000Z"
}
]
This illustrates how one can trace every step. If an error occurs (say the RSS fetch fails), we’d see a status
"failure" and an error message, followed by perhaps a retry entry. The Logging/Auditor ensures no log is
lost, even if a crash happens (it will flush logs on exception).
In summary, the logging format is structured, comprehensive, and stored alongside outputs. It serves both
as an audit trail and a debug tool to refine the system.
13
 Extensibility Strategy
MASX Agentic AI is designed with extensibility and adaptability in mind. Key strategies to allow easy
extension, maintenance, and deployment flexibility include:
Modular Agent Architecture: Each agent is implemented as an independent module with a clear
interface (inputs/outputs). Adding a new agent is straightforward: define its role and tools, integrate
it into the LangGraph workflow, and update the Orchestrator’s directed graph. For example, if we
wanted to add a “Sentiment Analyzer” agent to gauge tone of news (positive/negative), we could
insert it after Entity Extractor. Its output (e.g., a sentiment score) could then be used by Event
Analyzer to prioritize events. Thanks to the directed graph design, we can insert or remove nodes
without rewriting the entire logic – just adjust the edges and conditions.
Swap-in of LLMs or Tools: The system is LLM-agnostic as long as the model can follow instructions
to output the required JSON. We currently use GPT-4 via OpenAI API, but we can swap to an opensource model (like Llama 2) deployed locally. The prompts and expected schema remain the same.
We might need to adjust system messages for different models, but the role of each agent doesn’t
change. Similarly, tools like translation API or news API can be replaced with equivalents. For
instance, if Google News RSS becomes unavailable, we could integrate a different news API (or even
web scraping via a Browser tool) by modifying the News Fetcher’s toolset. Because the output format
(list of articles) is consistent, other agents won’t need changes. The codebase or CrewAI
configuration would load different tool classes depending on environment (cloud vs on-prem).
Cloud and On-Prem Deployments: To support on-premise deployments, we minimize external
dependencies. All critical components (Orchestrator, agents, vector DB) can run in a self-contained
environment. Supabase can be self-hosted or replaced with a plain Postgres + pgvector in an onprem server . The GDELT integration could be adjusted to use offline data if internet access is not
allowed – e.g., using GDELT data dumps if provided internally, or skip it and rely on internal news
sources. The design allows toggling features via config. We use environment flags to enable/disable
network calls. For example, USE_GDELT=true/false could include or skip the Event Fetcher
agent. If on-prem, one might set USE_TRANSLATOR=false to avoid external API and instead rely
on a local multilingual model for translation. The orchestrator will only activate agents whose
features are enabled. This conditional inclusion is built into the directed graph (certain branches can
be no-ops if disabled).
Internal Scheduling vs External: The pipeline is built to run as a persistent process (or a container)
that triggers itself daily, but it can also be orchestrated by external schedulers if needed. For
extensibility, we separated scheduling logic from core logic. If a user wants to use Airflow or Cron in
cloud, they can call the pipeline’s run function externally. Conversely, for simplicity, the system can
internally schedule with Python’s scheduling libraries or a simple sleep-loop. All scheduling
configurations (timing, time zone, etc.) are in a config file or environment variable, making it easy to
change (e.g., to twice-daily runs).
Scalability and Parallelism: Extensibility also means the system can scale with more data or more
agents. The design allows parallel execution of independent agents – for example, News Fetcher and
Event Fetcher run in parallel threads or async tasks to speed up data gathering. We could also
parallelize the handling of different domains or regions by spawning sub-crews for each (if in the
•
•
•
10
•
•
14
future we want separate agent crews focusing on different topics simultaneously). LangGraph
supports hierarchical agents and concurrency where appropriate . Our architecture could be
extended to a hierarchy: e.g., a top-level Orchestrator spawns multiple regional sub-orchestrators
(each with their own crew of agents) to cover more ground, then a results-merging agent combines
their outputs. This would be an extension for higher throughput.
CrewAI and AutoGen Configurability: Because we followed CrewAI patterns, one can leverage
CrewAI’s framework to modify agent definitions or add new roles without heavy refactoring. CrewAI’s
config (roles, tasks, crew assembly) can be externalized, enabling non-developers to adjust agent
roles or priorities. AutoGen’s conversation patterns are also configurable – for instance, one could
extend the Fact-Checker <-> Event Analyzer dialogue with more turns or integrate a
“UserProxyAgent” for a human analyst to jump in if needed . These frameworks are designed
for extension, so our system can evolve (e.g., adding a UI agent or chat interface later for an analyst
to query the system directly).
Logging and Monitoring for Continuous Improvement: We treat the logs as a feedback
mechanism. By analyzing logs over time, developers can identify where agents struggle or
bottlenecks occur. This informs extensions – e.g., if the logs show frequent fact-check corrections, it
might be worth integrating a knowledge base to the Fact-Checker or improving the initial query
strategy. The system’s modular design means improvements can be localized (improve one agent or
add a new one to handle a recurring issue). Monitoring dashboards can be built on the log data to
visualize performance and errors, guiding future enhancements.
Hot-Swapping and A/B Testing: In a production scenario, we can run multiple agent versions in
parallel for testing. Since each agent’s output is JSON, we could test a new version of, say, the Event
Analyzer (maybe one using a different LLM or algorithm) by running it in parallel on the same input
and comparing outputs (this could even be done as an additional agent that evaluates differences).
The orchestration could route to either the old or new agent based on a config flag (enabling A/B
testing without affecting the whole pipeline).
Portability: The entire system can be containerized (Docker) with all dependencies (LLM model,
translation model, etc. if self-hosted). This makes it easy to deploy on cloud or on-prem Kubernetes.
The design avoids reliance on any proprietary cloud-only service (Supabase can be cloud or selfhosted, OpenAI can be swapped with local models, etc.). This ensures no vendor lock-in and allows
moving the system as needed.
Documentation and Diagram-as-Code: The architecture diagram and flows are kept as
documentation (like this markdown and possibly Mermaid diagrams) that developers can update as
they extend the system. We encourage keeping the LangGraph visualization up-to-date so one can
clearly see the flow when adding new nodes or agents.
In essence, MASX Agentic AI is built to grow and adapt. Whether adding new data sources (e.g., social
media feeds), new analytical agents (risk scoring, influence estimation), or deploying in a constrained
environment, the system’s modular crew of agents and flexible orchestrator can accommodate changes. By
borrowing patterns from projects like the AI hedge fund (multi-expert agents contributing to a common
goal) , we ensure that adding an expert agent only enriches the system’s output without requiring an
21 24
•
25 26
•
•
•
•
27 28
15
overhaul. This extensible, lego-block architecture means MASX can continuously evolve to meet future
intelligence needs.
LangGraph: A Framework for Building Agentic AI with Directed Graphs | by
Shinimarykoshy | Medium
https://medium.com/@shinimarykoshy1996/langgraph-a-framework-for-building-agentic-ai-with-directed-graphs-096bf7af07b6
Building a multi agent system using CrewAI | by Vishnu Sivan | The Pythoneers | Medium
https://medium.com/pythoneers/building-a-multi-agent-system-using-crewai-a7305450253e
Google News RSS Search Parameters: The Missing Docs | NewsCatcher
https://www.newscatcherapi.com/blog/google-news-rss-search-parameters-the-missing-documentaiton
GitHub - alex9smith/gdelt-doc-api: A Python client for the GDELT 2.0 Doc API
https://github.com/alex9smith/gdelt-doc-api
AI & Vectors | Supabase Docs
https://supabase.com/docs/guides/ai
LangGraph
https://www.langchain.com/langgraph
Multi-agent Conversation Framework | AutoGen 0.2
https://microsoft.github.io/autogen/0.2/docs/Use-Cases/agent_chat/
How Does AI Hedge Fund Work: Clearly Explained
http://anakin.ai/blog/how-does-ai-hedge-fund-work/
1 3 4 18 19 20
2 11 12
5 6
7 8 9
10
13 21 22 23 24
14 15 16 17 25 26
27 28
16