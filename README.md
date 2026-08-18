# UnTangle

**Make sense of your complex documents.**

UnTangle is a document Q&A assistant. Upload a PDF or text file, then ask questions about it in plain English. Answers are generated from your document using a local LLM — your files stay on your machine.

> **Important:** This app **requires** the [Chainlit persistence layer](https://github.com/Chainlit/chainlit-datalayer) to be running before you start it. Without PostgreSQL and blob storage, the app will not work.

---

## Features

- Upload **PDF** and **`.txt`** files
- Ask questions in natural language and get streaming answers
- **100% local** inference via [Ollama](https://ollama.ai)
- **RAG pipeline** powered by LangChain + Chroma
- **Per-user, per-conversation** document storage
- **Chat history & resume** via PostgreSQL persistence
- Password-protected login

---

## How It Works

```
Upload document → Split into chunks → Embed with Ollama → Store in Chroma
                                                              ↓
User asks question → Retrieve relevant chunks → LLM answers from context
```

Chat threads, uploaded files, and session data are persisted through the Chainlit data layer (PostgreSQL + Azure Blob Storage / Azurite).

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | Recommended |
| [Docker](https://docs.docker.com/get-docker/) | **Required** — runs the persistence layer |
| [Ollama](https://ollama.ai/download) | Must be running locally |
| [Node.js](https://nodejs.org/) | Required for Prisma migrations in the data layer |
| Chat model | e.g. `gemma4:31b-cloud`, `llama3`, `mistral` |
| Embedding model | `nomic-embed-text` (required) |
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

Follow these steps **in order**. Do not skip the persistence layer setup.

### 1. Clone the project

```bash
git clone https://github.com/Nileshtn/UnTangle.git`
cd docllm
```

### 2. Clone the persistence layer

The `chainlit-datalayer` folder is not included in this repo. Clone it into the project root:

```bash
git clone https://github.com/Chainlit/chainlit-datalayer.git
```

Your folder structure should look like:

```
docllm/
├── app.py
├── chainlit-datalayer/    ← cloned separately
└── ...
```

### 3. Create a Python environment and install dependencies

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

Install app dependencies:

```bash
pip install -r requirements.txt
pip install asyncpg azure-storage-blob aiohttp
```

### 4. Start the persistence layer containers

```bash
cd chainlit-datalayer
docker compose up -d
```

This starts:
- **PostgreSQL** — stores chat threads, users, and messages
- **Azurite** — local Azure Blob Storage for uploaded file attachments

Wait until the containers are healthy:

```bash
docker compose ps
```

### 5. Initialize the database schema

Still inside `chainlit-datalayer/`:

```bash
npm install
npx prisma migrate deploy
```

### 6. Initialize blob storage (first time only)

Still inside `chainlit-datalayer/`:

```bash
python init_azure_storage.py
```

This creates the `my-container` blob container used for file uploads.

Go back to the project root:

```bash
cd ..
```

### 7. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
AUTH_USERNAME=admin
AUTH_PASSWORD=your-secure-password
CHAINLIT_AUTH_SECRET=your-long-random-secret-string

DATABASE_URL=postgresql://root:root@localhost:5432/postgres

BUCKET_NAME=my-container
APP_AZURE_STORAGE_ACCOUNT=devstoreaccount1
APP_AZURE_STORAGE_ACCESS_KEY=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==
APP_AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:10000/devstoreaccount1;QueueEndpoint=http://localhost:10001/devstoreaccount1;TableEndpoint=http://localhost:10002/devstoreaccount1
DEV_AZURE_BLOB_ENDPOINT=http://localhost:10000/devstoreaccount1

OLLAMA_MODEL=gemma4:31b-cloud
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Generate a random auth secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 8. Run the app

Make sure:
1. Docker containers are running (`docker compose ps` in `chainlit-datalayer/`)
2. Ollama is running (`ollama list`)

Then start UnTangle:

```bash
conda activate dl          # if using conda
chainlit run app.py --host 0.0.0.0 --port 8000
```

Open your browser at **http://localhost:8000** and sign in with your `AUTH_USERNAME` / `AUTH_PASSWORD`.

---

## Startup Order (Every Time)

Whenever you want to use UnTangle, start services in this order:

```bash
# 1. Start persistence layer
cd chainlit-datalayer
docker compose up -d
cd ..

# 2. Make sure Ollama is running
ollama list

# 3. Start the app
conda activate dl
chainlit run app.py --host 0.0.0.0 --port 8000
```

To stop the persistence layer when you're done:

```bash
cd chainlit-datalayer
docker compose down
```

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

Use the **history panel** on the left to browse and reopen previous conversations. Your documents and chat context are restored automatically.

---

## Configuration

All settings are read from `.env`. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_USERNAME` | `admin` | Login username |
| `AUTH_PASSWORD` | `changeme` | Login password |
| `CHAINLIT_AUTH_SECRET` | — | Secret for session tokens (**required**) |
| `DATABASE_URL` | — | PostgreSQL URL (**required**) |
| `BUCKET_NAME` | — | Azure blob container name (**required**) |
| `APP_AZURE_STORAGE_*` | — | Azure / Azurite storage credentials (**required**) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `gemma4:31b-cloud` | Chat model name |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_TEMPERATURE` | `0.2` | LLM temperature (0 = deterministic) |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_TOP_K` | `3` | Number of chunks sent to the LLM |
| `DATA_DIR` | `data` | Root folder for user/thread data |

---

## Run with Docker (Full Stack)

The root `docker-compose.yml` runs the app together with PostgreSQL. You still need Ollama on the host and the `chainlit-datalayer` blob storage (Azurite) for file uploads.

```bash
# 1. Start blob storage from the persistence layer
cd chainlit-datalayer
docker compose up -d azurite
python init_azure_storage.py   # first time only
cd ..

# 2. Start the app stack
cp .env.example .env
# Edit .env with your credentials and secrets
docker compose up --build
```

Open **http://localhost:8000**.

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
└── chainlit-datalayer/     # Persistence layer (clone separately — see step 2)
    ├── compose.yaml        # PostgreSQL + Azurite containers
    ├── prisma/             # Database schema
    └── init_azure_storage.py
```

---

## Troubleshooting

### App fails to start or crashes immediately

The persistence layer is probably not running. Check:

```bash
cd chainlit-datalayer
docker compose ps          # postgres and azurite should be "Up"
npx prisma migrate deploy  # re-run if tables are missing
```

### "Please upload a PDF or `.txt` file before asking questions"

No document has been indexed yet. Attach a file before asking your question.

### File uploads fail

- Make sure Azurite is running: `docker compose ps` in `chainlit-datalayer/`
- Re-run the blob storage init: `python init_azure_storage.py`
- Verify `BUCKET_NAME` and `APP_AZURE_STORAGE_*` values in `.env`

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
- Run `docker compose up -d` in `chainlit-datalayer/`

### Database connection refused

PostgreSQL is not running. Start it:

```bash
cd chainlit-datalayer
docker compose up -d postgres
```

### Unsupported file type

Only `.pdf` and `.txt` files are supported. Convert other formats before uploading.

### Slow first response

The first question after uploading a large document may take longer while Ollama loads the model. Subsequent questions are faster.

---

## Built With

- [Chainlit](https://chainlit.io) — Chat UI
- [Chainlit Data Layer](https://github.com/Chainlit/chainlit-datalayer) — Chat & file persistence
- [LangChain](https://www.langchain.com) — RAG orchestration
- [Ollama](https://ollama.ai) — Local LLM and embeddings
- [Chroma](https://www.trychroma.com) — Vector database
- [PostgreSQL](https://www.postgresql.org) — Chat persistence
- [Azurite](https://github.com/Azure/Azurite) — Local blob storage for file uploads

---

## License

Add your license here.
