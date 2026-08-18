# UnTangle

**Make sense of your complex documents.**

UnTangle is a document Q&A assistant. Upload a PDF or text file, then ask questions about it in plain English. Answers are generated from your document using a local LLM — your files stay on your machine.

---

## Features

- Upload **PDF** and **`.txt`** files
- Ask questions in natural language and get streaming answers
- **100% local** inference via [Ollama](https://ollama.ai)
- **RAG pipeline** powered by LangChain + Chroma
- **Per-user, per-conversation** document storage
- **Chat history & resume** when PostgreSQL persistence is enabled
- Password-protected login

---

## How It Works

```
Upload document → Split into chunks → Embed with Ollama → Store in Chroma
                                                              ↓
User asks question → Retrieve relevant chunks → LLM answers from context
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | Recommended |
| [Ollama](https://ollama.ai/download) | Must be running locally |
| Chat model | e.g. `gemma4:31b-cloud`, `llama3`, `mistral` |
| Embedding model | `nomic-embed-text` (required) |
| Docker *(optional)* | For PostgreSQL chat persistence |
| Conda *(optional)* | Project tested with env `dl` |

### Pull Ollama models

```bash
ollama pull nomic-embed-text
ollama pull gemma4:31b-cloud   # or any chat model you prefer
```

Verify Ollama is running:

```bash
ollama list
```

---

## Quick Start

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd docllm
```

### 2. Create a virtual environment

**Option A — Conda (recommended)**

```bash
conda create -n dl python=3.11 -y
conda activate dl
```

**Option B — venv**

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
AUTH_USERNAME=admin
AUTH_PASSWORD=your-secure-password
CHAINLIT_AUTH_SECRET=your-long-random-secret-string
OLLAMA_MODEL=gemma4:31b-cloud
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Generate a random auth secret (Linux / macOS):

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 5. Run the app

```bash
chainlit run app.py --host 0.0.0.0 --port 8000
```

Open your browser at **http://localhost:8000** and sign in with your `AUTH_USERNAME` / `AUTH_PASSWORD`.

---

## How to Use

### Step 1 — Sign in

When the app opens, enter the username and password from your `.env` file.

### Step 2 — Upload a document

Click the **attachment** icon in the chat input and select a `.pdf` or `.txt` file.

You can attach a file in the same message as your question, or upload it first in a separate message.

When processing finishes, you will see:

```
Added your-file.pdf.
```

### Step 3 — Ask questions

Type a question and press Enter. Examples:

- *What is this document about?*
- *Summarize the key findings in bullet points.*
- *What does section 3 say about liability?*
- *Explain the main technical terms in simple language.*

Answers stream in real time and are based on the uploaded document.

### Step 4 — Upload more documents

Attach additional files in later messages. Each new file is indexed and added to the current conversation's knowledge base.

### Step 5 — Resume a past conversation

If chat persistence is enabled (see below), use the **history panel** on the left to browse and reopen previous conversations. Your documents and chat context are restored automatically.

---

## Configuration

All settings are read from `.env`. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_USERNAME` | `admin` | Login username |
| `AUTH_PASSWORD` | `changeme` | Login password |
| `CHAINLIT_AUTH_SECRET` | — | Secret for session tokens (required) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `gemma4:31b-cloud` | Chat model name |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_TEMPERATURE` | `0.2` | LLM temperature (0 = deterministic) |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_TOP_K` | `3` | Number of chunks sent to the LLM |
| `DATA_DIR` | `data` | Root folder for user/thread data |
| `DATABASE_URL` | — | PostgreSQL URL for chat history |
| `APP_PORT` | `8000` | Port when running via Docker |

---

## Chat History & Persistence (Optional)

To save conversations and resume them later, you need PostgreSQL and a `CHAINLIT_AUTH_SECRET`.

### Start backing services

```bash
cd chainlit-datalayer
docker compose up -d postgres azurite
cd ..
```

Make sure `.env` contains:

```env
DATABASE_URL=postgresql://root:root@localhost:5432/postgres
CHAINLIT_AUTH_SECRET=<your-secret>
```

Then restart the app. Signed-in users will see their conversation history in the sidebar.

> **Note:** Without `DATABASE_URL`, the app still works for document Q&A — you just won't have saved chat history.

---

## Run with Docker

Runs the app together with PostgreSQL. Ollama must still run on the host machine.

```bash
cp .env.example .env
# Edit .env with your credentials and secrets

docker compose up --build
```

Open **http://localhost:8000**.

The app container connects to Ollama via `http://host.docker.internal:11434` by default.

---

## Project Structure

```
docllm/
├── app.py                  # Chainlit app (auth, upload, chat handlers)
├── config.py               # Settings loaded from .env
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── chainlit.md             # Welcome screen shown in the UI
├── prompts/
│   └── rag_prompt.yaml     # System prompt for the LLM
├── utils/
│   ├── core.py             # FileManager, VectorStore, DocLLM
│   └── session.py          # Per-user session initialization
├── data/                   # Per-user/per-thread documents (auto-created)
└── chainlit-datalayer/     # PostgreSQL + Azurite for chat persistence
```

---

## Troubleshooting

### "Please upload a PDF or `.txt` file before asking questions"

No document has been indexed yet. Attach a file before asking your question.

### "Sorry, something went wrong while generating a response"

- Check that Ollama is running: `ollama list`
- Confirm your chat model is pulled: `ollama pull <OLLAMA_MODEL>`
- Confirm the embedding model is pulled: `ollama pull nomic-embed-text`
- Check the terminal for error logs

### Login fails

- Verify `AUTH_USERNAME` and `AUTH_PASSWORD` in `.env`
- Ensure `CHAINLIT_AUTH_SECRET` is set
- Restart the app after changing `.env`

### Chat history not showing

- `DATABASE_URL` must be set and PostgreSQL must be running
- You must be signed in (history is per authenticated user)
- Run `docker compose up -d postgres` in `chainlit-datalayer/`

### Unsupported file type

Only `.pdf` and `.txt` files are supported. Convert other formats before uploading.

### Slow first response

The first question after uploading a large document may take longer while Ollama loads the model. Subsequent questions are faster.

---

## Built With

- [Chainlit](https://chainlit.io) — Chat UI
- [LangChain](https://www.langchain.com) — RAG orchestration
- [Ollama](https://ollama.ai) — Local LLM and embeddings
- [Chroma](https://www.trychroma.com) — Vector database
- [PostgreSQL](https://www.postgresql.org) — Chat persistence (optional)

---

## License

Add your license here.
