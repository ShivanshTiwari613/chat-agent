# Chat Agent

An AI agent with tool use, retrieval over uploaded documents, and a persistent
E2B Python sandbox for code execution. Ships with both a terminal CLI and a
Flask server that streams agent events over Server-Sent Events.

## Features

- **LLM-driven agent loop** built on LangChain (defaults to Google Gemini).
- **Persistent E2B sandbox** for running Python, shell commands, and generated
  code with state preserved across turns.
- **Document RAG** — upload PDFs, DOCX, code, CSV, etc. Files are chunked,
  embedded with `sentence-transformers`, indexed with FAISS + BM25, and stored
  per-session under a namespaced index.
- **Web search** via Tavily.
- **Session persistence** — chat history, uploaded files, and auto-generated
  chat titles are stored in SQLite/Postgres via SQLAlchemy.
- **Two entry points**: a CLI (`main.py`) and a streaming HTTP server
  (`server.py`).

## Requirements

- Python 3.10+
- An E2B account (for sandboxed code execution)
- A Google AI Studio key (for Gemini)
- A Tavily key (for the search tool)

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env` and set `GOOGLE_API_KEY`, `E2B_API_KEY`, and `TAVILY_API_KEY`.
See `config/settings.py` for the full list of tunables (model name,
temperature, token limits, alternate LLM providers, etc.).

## Running

### CLI

```bash
python main.py
```

In-session commands:

| Command           | Description                                       |
| ----------------- | ------------------------------------------------- |
| `/upload <path>`  | Upload and index a file (PDF, CSV, code, etc.)    |
| `exit` / `quit`   | End the session                                   |
| `Ctrl+C`          | Graceful shutdown                                 |

### HTTP server

```bash
python server.py
```

Listens on `http://0.0.0.0:5000`. Endpoints:

| Method   | Route                  | Purpose                                       |
| -------- | ---------------------- | --------------------------------------------- |
| `GET`    | `/sessions`            | List saved sessions (id, title, created_at)   |
| `POST`   | `/upload`              | Upload files for a session (multipart form)   |
| `POST`   | `/chat`                | Stream a chat turn as SSE agent events        |
| `DELETE` | `/session/<plan_id>`   | Tear down a session and its sandbox           |

`/chat` accepts JSON of the form `{"plan_id": "...", "message": "...", "chat_history": [...]}`
and streams events with `type` in `{status, tool, observation, result, error}`.

## Project layout

```
app/
  agent/        # LLM engine + system prompt
  api/          # Pydantic schema for streamed agent events
  sandbox/      # E2B sandbox handler
  tool/         # Tool implementations (search, code, file, terminal)
  utils/        # DB models, logger, title generator
config/         # Pydantic settings loaded from .env
persistent_storage/  # Per-session uploads + FAISS indexes (gitignored)
main.py         # CLI entry point
server.py       # Flask + SSE entry point
```

## Notes

- All runtime configuration lives in `config/settings.py` and is loaded from
  `.env` via `pydantic-settings`.
- If you supply your own `E2B_TEMPLATE_ID`, the template must be based on the
  E2B `code-interpreter` runtime.
- `persistent_storage/` and `.env` are gitignored — keep them that way.
