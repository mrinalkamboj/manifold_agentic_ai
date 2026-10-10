# Session 9: policy RAG

Loads five fictional policy files, splits them into passages, embeds and saves them, retrieves evidence, and returns a structured answer with passage references.

Use this folder as the working directory.

## Setup

Use Python **3.12**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. `OPENAI_MODEL` selects the chat model. `OPENAI_EMBEDDING_MODEL` selects the embedding model. Do not print or commit the key.

With Anaconda: `conda create --prefix ./myenv python=3.12`, then `conda activate ./myenv`, and install the same lock.

## Commands

```bash
python demo.py load
python demo.py chunk
python demo.py index
python demo.py search "What is the refund estimate for a damaged item?"
python demo.py ask "What is the refund estimate for a damaged item?"
```

`search` retrieves passages. `ask` retrieves passages and then calls the chat model when evidence is available.

```bash
python demo.py index
python demo.py ask "What is the policy for lunar mining royalties?"
```

A second `index` with unchanged documents reports `reused`.

## Offline rehearsal

```bash
python demo.py load
python demo.py chunk
python demo.py index --offline
python demo.py search "What is the refund estimate for a damaged item?" --offline
python demo.py ask "What is the refund estimate for a damaged item?" --offline
python demo.py ask "What is the policy for lunar mining royalties?" --offline
```

`--offline` uses keyword vectors and a retrieved excerpt. It does not call a model. Output is labelled offline. Live and offline indexes are stored separately.

## Files

| File | Purpose |
|---|---|
| `demo.py` | Commands |
| `rag.py` | Load, split, index, retrieve, answer |
| `config.py` | Settings from `.env` |
| `schemas.py` | Question, answer, and saved-index shapes |
| `providers.py` | Embedding and chat clients |
| `prompts.py` | Model input |
| `data/policies/` | Fictional policies |
| `ASSIGNMENT.md` | What to submit |
| `tests/test_rag.py` | Checks that do not need an API key |

## Checks

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

| Result | Next action |
|---|---|
| Build the index first | Run `python demo.py index` in the same mode as the question |
| Documents or index configuration changed | Run the index command again |
| No useful retrieved passage | Read `search` output and the policy file |
| Model cited an unsupplied passage | Inspect the prompt and the retrieved IDs |
| Missing key or unavailable provider | Set `.env`, or rerun with `--offline` |
| Corrupt saved snapshot | Move that snapshot out of `.rag-index` and rebuild |
