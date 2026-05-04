# Private Enterprise AI Agent (RAG)
Production-style private knowledge assistant using:
- LangGraph orchestration
- Chroma vector retrieval
- Provider-flexible LLM/embeddings (`ollama` or `openai`)

This project is optimized for your workflow:
1. Push repo to remote.
2. Clone in Google Colab.
3. Run ingestion + Q&A in notebook cells.
4. Keep development defaults safe and production-hardening explicit.

## Architecture
```text
User Question
    -> LangGraph (retrieve -> generate)
    -> Chroma Retriever (top-k chunks)
    -> LLM response constrained to internal context
    -> Source-aware answer
```

## Project layout
```text
private_ai_agent/
├── data/                       # Internal source documents (.txt/.md/.pdf)
├── src/private_agent/
│   ├── config.py               # Typed env-driven settings
│   ├── ingest.py               # Document loading/chunking/indexing
│   ├── graph.py                # Retrieval + generation graph
│   ├── logging_utils.py        # Standardized logging configuration
│   ├── modeling.py             # Ollama/OpenAI model factories
│   └── main.py                 # Interactive and one-shot CLI entrypoint
├── .env.example
├── pyproject.toml
└── README.md
```

## Local quickstart (dev)
```bash
uv venv
uv sync --extra dev
```

### Runtime profile (choose one)
Copy `.env.example` to `.env` and update values.

#### A) Private/local models with Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral-small3.2
ollama pull nomic-embed-text
```

`.env`:
```dotenv
PRIVATE_AGENT_LLM_PROVIDER=ollama
PRIVATE_AGENT_LLM_MODEL=mistral-small3.2
PRIVATE_AGENT_EMBEDDING_PROVIDER=ollama
PRIVATE_AGENT_EMBEDDING_MODEL=nomic-embed-text
```

#### B) Colab-friendly API models with OpenAI
`.env`:
```dotenv
PRIVATE_AGENT_LLM_PROVIDER=openai
PRIVATE_AGENT_LLM_MODEL=gpt-5-mini
PRIVATE_AGENT_EMBEDDING_PROVIDER=openai
PRIVATE_AGENT_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=replace_me
```

## Ingest and run
```bash
uv run python -m private_agent.ingest --rebuild
uv run python -m private_agent.main --question "Who approves a £12,000 invoice?"
uv run python -m private_agent.main
```

## Google Colab runbook (step-by-step)
Use this flow after you push your repo.

### 1) Start clean Colab runtime
- Runtime -> Change runtime type -> keep latest runtime unless you need pinning.

### 2) Clone your private repo securely
Create a fine-grained GitHub token with read access to the target repository.
Then in Colab:
```python
from google.colab import userdata
import base64
import subprocess

token = userdata.get("GITHUB_TOKEN")  # set via Colab Secrets (key icon)
repo_url = "https://github.com/<ORG_OR_USER>/<REPO>.git"
auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()

subprocess.run(
    ["git", "-c", f"http.extraheader=AUTHORIZATION: basic {auth}", "clone", repo_url],
    check=True,
)
```

### 3) Install project in notebook kernel
```python
%cd <REPO>
!pip install -qU pip setuptools wheel
!pip install -q -e .
```

### 4) Set runtime environment in Colab
#### OpenAI profile (recommended for Colab simplicity)
```python
from google.colab import userdata
import os

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["PRIVATE_AGENT_LLM_PROVIDER"] = "openai"
os.environ["PRIVATE_AGENT_LLM_MODEL"] = "gpt-5-mini"
os.environ["PRIVATE_AGENT_EMBEDDING_PROVIDER"] = "openai"
os.environ["PRIVATE_AGENT_EMBEDDING_MODEL"] = "text-embedding-3-small"
```

#### Ollama profile (optional in Colab)
```python
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
ollama_process = subprocess.Popen(["ollama", "serve"])
time.sleep(5)
!ollama pull mistral-small3.2
!ollama pull nomic-embed-text
```

### 5) Build vector DB and test
```python
!python -m private_agent.ingest --rebuild
!python -m private_agent.main --question "What is the invoice policy for amounts above £10,000?"
```

## Hardening checklist (enterprise)
1. Secrets and identity:
- Use per-user/per-project API keys, never shared personal keys.
- Keep keys only in environment/secrets manager, never in git.

2. Environment separation:
- Keep distinct dev/staging/prod configs and API projects.
- Set spend and rate limits per environment.

3. Data governance:
- Ingest only approved internal documents.
- Remove sensitive raw data that the assistant should never expose.

4. Retrieval quality:
- Tune `PRIVATE_AGENT_CHUNK_SIZE`, `PRIVATE_AGENT_CHUNK_OVERLAP`, and `PRIVATE_AGENT_RETRIEVER_K` with eval data.
- Rebuild index when source docs change materially.

5. Reliability and observability:
- Keep `PRIVATE_AGENT_LOG_LEVEL=INFO` (or `DEBUG` during incident triage).
- Add automated regression prompts in CI before production release.

6. Human-in-the-loop safety:
- Keep mandatory human review for HR/legal/finance/security edge decisions.
- Treat model output as decision support, not final authority.

## References (current guidance)
- OpenAI production best practices: https://platform.openai.com/docs/guides/production-best-practices/model-overview
- OpenAI API key safety: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
- OpenAI model catalog: https://platform.openai.com/docs/models
- OpenAI embeddings guide: https://platform.openai.com/docs/guides/embeddings
- GitHub PAT guidance: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
- Colab runtime version FAQ: https://research.google.com/colaboratory/runtime-version-faq.html
