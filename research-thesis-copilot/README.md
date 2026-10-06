# Research & Thesis Co-Pilot

A LangGraph + RAG + n8n system that turns a firehose of arXiv papers into
literature reviews, gap analyses and citation-checked answers — automatically
kept up to date.

Built for the "academic" half of an AI/automation engineer's portfolio:
reading papers doesn't scale, so this delegates the first pass (find, index,
summarize, flag gaps) to agents and keeps you in the loop for the judgment
calls (which gap is worth a thesis).

## Why this is more than "a RAG chatbot over PDFs"

1. **Grounded by construction, not by prompt** — a dedicated `citation_check`
   node re-reads every draft against the retrieved context and rejects
   claims it can't verify, looping the draft back for revision (bounded to
   2 retries) before shipping a flagged answer. This is the same
   self-correction pattern used in the portfolio's Fake-News-Detector
   project, applied here to literature review instead of fact-checking.
2. **One graph, three research tasks** — `literature_review`,
   `gap_analysis` and `citation_qa` are the same LangGraph state machine
   with a different system prompt selected at the `draft` node, so adding a
   fourth task (e.g. `methodology_critique`) is a one-line change.
3. **n8n owns the clock, the API owns the reasoning** — the
   FastAPI service is stateless and has no opinion about *when* it runs;
   n8n's schedule triggers decide cadence (daily ingestion, weekly digest),
   which means you can change "daily" to "every 6 hours" in n8n without
   touching a line of Python.

## Architecture

```
            ┌────────────────────┐        ┌─────────────────────────────┐
  arXiv ───▶│  n8n: daily cron   │──HTTP─▶│ FastAPI  /ingest/arxiv       │
            │  ingestion workflow│        │          /ingest/pdf         │
            └────────────────────┘        └──────────────┬───────────────┘
                                                           │ upsert
                                                           ▼
                                                 ┌───────────────────┐
                                                 │ Chroma vector store│
                                                 │ (local embeddings) │
                                                 └─────────┬──────────┘
                                                           │ retrieve(k)
                                                           ▼
            ┌────────────────────┐        ┌─────────────────────────────┐
  you ─────▶│  POST /ask         │───────▶│  LangGraph: retrieve → draft │
            │  {question, mode}  │        │  → citation_check → (loop)  │
            └────────────────────┘        │  → finalize                │
                                           └──────────────┬──────────────┘
                                                           │ final_answer + sources
            ┌────────────────────┐        ┌───────────────▼──────────────┐
  Monday ──▶│ n8n: weekly digest │──HTTP─▶│ GET /digest/weekly            │
  08:00     │ workflow           │        │ (new papers + 1 gap-analysis) │
            └─────────┬──────────┘        └───────────────────────────────┘
                       │ format + send
                       ▼
                 Gmail / Slack
```

## Project layout

```
research-thesis-copilot/
├── app/
│   ├── ingestion/        # arXiv fetch, PDF loading, chunk+embed
│   ├── rag/               # Chroma vector store + retriever
│   ├── agents/            # LangGraph state, nodes, compiled graph
│   ├── api/server.py      # FastAPI surface n8n calls
│   ├── cli.py             # run ingestion/ask from the terminal
│   └── config.py
├── n8n/
│   ├── workflow-paper-ingestion.json   # daily: fetch -> index -> download PDFs -> Slack
│   └── workflow-weekly-digest.json     # weekly: digest -> format -> email
├── tests/test_graph.py     # mocks the LLM, tests the graph's control flow
├── docker-compose.yml       # copilot-api + n8n, one command
└── requirements.txt
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY

# index a batch of papers and ask a question, no n8n needed yet
python -m app.cli ingest --category cs.CL --max-results 10
python -m app.cli ask "What retrieval strategies do these papers use?" --mode literature_review

# or run the API and point n8n at it
python -m app.api.server
```

Then in n8n: **Import from File** both workflows under `n8n/`, point the
HTTP Request nodes at your API's address (default assumes `localhost:8080`,
change it if you run via `docker-compose up`), connect a Gmail/Slack
credential, and toggle each workflow **Active**.

## Run the tests

```bash
pytest tests/ -v
```

The tests monkeypatch the Claude call so they run offline and assert the
graph's *control flow* (the revision loop actually loops, and exits after
2 attempts) rather than model output quality.

## Extending it

- **Add Zotero/Mendeley sync**: swap `fetch_arxiv.py` for a Zotero API
  client; everything downstream (chunking, the graph) is already
  source-agnostic since it only cares about `(text, metadata)` pairs.
- **Add a `methodology_critique` mode**: add one entry to
  `SYSTEM_PROMPTS` in `app/agents/nodes.py` — the graph topology doesn't
  change.
- **Swap local embeddings for an API model**: replace the
  `SentenceTransformerEmbeddingFunction` in `app/rag/vectorstore.py` with
  `chromadb.utils.embedding_functions.OpenAIEmbeddingFunction` (or any
  Chroma-supported embedding function).
