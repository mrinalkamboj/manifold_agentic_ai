# Session 10: RAG inside a workflow

Continuation of Session 9. Retrieval is a tool, connected to a three-path workflow, then compared with LCEL and LlamaIndex.

This folder is self-contained. `rag_core/` is the Session 9 retrieval code with local paths. The same five fictional policy files are included.

## Setup

Use **Python 3.12** and a separate environment from Session 9.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
cp .env.example .env
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and copy the environment file with `Copy-Item .env.example .env`. With Anaconda, use `conda create --prefix ./myenv python=3.12`, then `conda activate ./myenv` and install the same lock.

Set `OPENAI_API_KEY` in `.env`. Do not print or commit the key. This folder does not load a Session 9 `.env`.

## Commands

Run from this folder:

```bash
python prepare.py
python 01_retrieval_tool.py "What is the refund estimate for a damaged item?"
python 02_routed_workflow.py "What is the refund estimate for a damaged item?"
python 02_routed_workflow.py "What is the weather in Mumbai right now?"
python 02_routed_workflow.py "Explain RAG in one sentence."
python 02_routed_workflow.py "What is the policy for lunar mining royalties?"
python 03_lcel_rag.py "What is the refund estimate for a damaged item?"
python 04_llamaindex_rag.py "What is the refund estimate for a damaged item?" --search-only
python 04_llamaindex_rag.py "What is the refund estimate for a damaged item?"
```

`01_retrieval_tool.py` prints retrieved passages. `02_routed_workflow.py` prints the route and the nodes that ran.

## Offline rehearsal

Add `--offline` to preparation and every command:

```bash
python prepare.py --offline
python 01_retrieval_tool.py "What is the refund estimate for a damaged item?" --offline
python 02_routed_workflow.py "What is the refund estimate for a damaged item?" --offline
python 02_routed_workflow.py "What is the weather in Mumbai right now?" --offline
python 02_routed_workflow.py "Explain RAG in one sentence." --offline
python 02_routed_workflow.py "What is the policy for lunar mining royalties?" --offline
python 03_lcel_rag.py "What is the refund estimate for a damaged item?" --offline
python 04_llamaindex_rag.py "What is the refund estimate for a damaged item?" --search-only --offline
python 04_llamaindex_rag.py "What is the refund estimate for a damaged item?" --offline
```

`--offline` does not call a model or the weather service. Output is labelled offline.

```bash
python source_failure.py
```

That command uses a temporary folder with no index. The prepared index in this folder stays in place.

## Optional

```bash
python 05_optional_quality.py "What is the refund estimate for a damaged item?" --offline
```

## Files

| File | Purpose |
|---|---|
| `prepare.py` | Build or reuse the index |
| `01_retrieval_tool.py`, `policy_tool.py` | Retrieval tool |
| `02_routed_workflow.py`, `workflow.py` | Three-path workflow |
| `weather_tool.py` | Mumbai weather call |
| `03_lcel_rag.py`, `lcel_pipeline.py` | LCEL path |
| `04_llamaindex_rag.py`, `llama_comparison.py` | LlamaIndex path |
| `05_optional_quality.py` | Optional top-k run |
| `source_failure.py` | Missing-index run |
| `common.py` | Shared result shapes and command logging |
| `rag_core/` | Retrieval functions |
| `data/policies/` | Fictional policies |
| `ASSIGNMENT.md` | What to submit |
| `tests/test_session10.py` | Checks that do not need an API key |

## Checks

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

| Result | Next action |
|---|---|
| Policy tool returns unavailable | Run `prepare.py` in the same mode |
| Documents changed | Run `prepare.py` again |
| Weather unavailable | The live call is Mumbai only; use `--offline` to rehearse |
| Wrong route | Read the question and the route in the output |
| Unsupplied citation rejected | Compare the cited IDs with the retrieved passages |
| LlamaIndex output differs | Compare its retrieved nodes with the other commands |
