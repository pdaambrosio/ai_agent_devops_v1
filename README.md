# ai_agent_devops_v1

A learning project that builds a DevOps/IT-infrastructure specialist agent on top of a local [Ollama](https://ollama.com/) model using [LangChain](https://www.langchain.com/) LCEL chains.

The repository contains two progressive examples:

1. **`basic_agent.py`** — a stateless DevOps Q&A agent built with a simple LCEL chain.
2. **`basic_agent_memory.py`** — the same agent, extended with an in-memory chat history so it can answer follow-up questions in context.

## Requirements

- Python `>=3.11` (see `.python-version`)
- [Ollama](https://ollama.com/download) installed and running locally on `http://127.0.0.1:11434`
- A pulled model. The scripts default to `ministral-3:3b`:
  ```bash
  ollama pull ministral-3:3b
  ```
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip` for dependency management.

## Dependencies

Declared in `pyproject.toml`:

- `langchain`
- `langchain-community`
- `langchain-ollama`
- `ollama`
- `python-dotenv`

## Setup

Using `uv`:

```bash
uv sync
```

Or using `pip` with a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Make sure the Ollama server is running before launching either script:

```bash
ollama serve
```

## Usage

### Stateless agent

```bash
uv run basic_agent.py
```

Type a DevOps/infrastructure question at the `Você:` prompt. Type `sair` to exit.

### Agent with memory

```bash
uv run basic_agent_memory.py
```

Commands available at the prompt:

- `sair` — quit
- `limpar` — clear the conversation history
- anything else — sent to the agent, which has access to previous turns

## Project layout

```
.
├── basic_agent.py          # Stateless LCEL chain over OllamaLLM
├── basic_agent_memory.py   # Same agent + in-memory history
├── main.py                 # Placeholder entry point
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency set
└── LICENSE
```

## How it works

Both scripts wire the same building blocks:

```
RunnablePassthrough → PromptTemplate → OllamaLLM → StrOutputParser
```

The memory variant additionally maintains a `history_chat` list of role-tagged turns and injects a formatted transcript into the prompt template before each call, so the model sees the prior exchanges as context.

## License

See [LICENSE](LICENSE).
