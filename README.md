## Email Assistant

Minimal agent that triages emails and drafts responses with LangChain + LangGraph. Works with only an OpenAI API key (Gmail tools fall back to mock data if Google APIs aren’t configured).

### Requirements
- Python 3.11+
- OpenAI API key

### 1) Environment
Create a `.env` in the project root:
```bash
OPENAI_API_KEY=sk-...your_openai_key...
# Optional (LangSmith)
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=ls_...your_langsmith_key...
# LANGSMITH_PROJECT=email-assistant
```

### 2) Install
- Using uv (recommended if you use uv.lock):
```bash
uv sync
```
- Or with pip/venv:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 3) Quick run
Runs the built-in demo in `src/email_assistant/email_assistant.py`:
```bash
python -m src.email_assistant.email_assistant | cat
```

### 4) LangGraph Dev UI (Studio)
Create `langgraph.json` at project root:
```json
{
  "$schema": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/langgraph.schema.json",
  "env": ".env",
  "python_path": ["src"],
  "dependencies": [
    "langgraph>=0.4.2",
    "langchain>=0.3.9",
    "langchain-core>=0.3.59",
    "langchain-openai",
    "python-dotenv"
  ],
  "graphs": {
    "email-assistant": "email_assistant.email_assistant:email_assistant"
  }
}
```
Install CLI and start the server:
```bash
python -m pip install -U "langgraph-cli[inmem]"
langgraph dev
```
Open the Studio URL printed in the terminal (or use `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`).

### Notes
- Gmail tools auto-fallback to mock data; Google setup is optional.
- Suggested `.gitignore` entries to keep secrets out of git:
```
.env
.env.*
secrets/
**/.secrets/
**/client_secret*.json
**/token*.json
notebooks/*.ipynb_checkpoints/
notebooks/token.pickle
__pycache__/
*.pyc
*.egg-info/
.venv/
```
