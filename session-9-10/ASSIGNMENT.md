# Session 9 assignment: policy Q&A assistant

## Goal

Build a small RAG assistant over fictional policies you write. A user asks a question. Your program retrieves relevant passages, then returns an answer with the passage IDs used.

Estimated effort: 2–3 hours. Use the session 9 code as your starting point.

## Your documents

Create at least three `.md` or `.txt` files in your project's `data/policies/` folder. Choose one domain: a fictional bookstore, training centre, subscription service, or delivery desk. Label every document as a fictional classroom policy.

Include at least six checkable facts across the files. For example: cancellation window, replacement eligibility, delivery investigation procedure, and whom to contact for an exception. Keep each file between roughly 300 and 1,500 characters so there is enough text to inspect chunking.

Do not copy the class policies unchanged. Do not use private customer data, passwords, or real company secrets.

## Required behaviour

1. Load the text files and retain the filename as source metadata.
2. Split documents into passages that retain source and a stable passage ID.
3. Build a vector index with live embeddings and save it for reuse.
4. Retrieve up to `top_k` passages for a question. Print their text and IDs.
5. Give the question and retrieved text to a model. Return `status`, `answer`, and `citations` as validated structured output.
6. Check that every cited ID was supplied to the model. Remember this validates the reference, not the truth of every sentence.
7. For a question unsupported by your documents, report insufficient evidence rather than inventing a policy.
8. Demonstrate the unchanged index being reused. After changing a policy file, rebuild before answering again.

Keep execution straightforward. You do not need a web interface, LangGraph, MCP, a cloud deployment, or multiple agents for this assignment.

## Evidence to submit

Submit a repository link or zip containing:

- Source code and your fictional policy files.
- Dependencies and a README with exact setup/run instructions.
- Output for **five supported questions**. For each, include the relevant retrieved passage and the final structured answer. Make at least one question paraphrase the document rather than repeat its exact wording.
- Output for **two unsupported questions**. Choose topics the documents actually do not address. Explain why the evidence is insufficient.
- Output showing a second index run reporting reuse, then a policy change and successful rebuild.
- A 5–8 sentence reflection explaining one failure, what you inspected, and how you fixed it.

Use a real model for at least one supported question and one unsupported question when API access is available. Clearly label every offline transcript. Offline keyword matching can verify the program flow but cannot prove semantic retrieval or model response quality. If access is unavailable, submit the rehearsal evidence and state that live evidence is pending.

Exclude `.env`, virtual environments, caches, and generated index snapshots from the submission. Provide the documents and instructions so the reviewer can rebuild the index.

## Rubric: 20 points

| Area | Points | What the reviewer checks |
|---|---:|---|
| Documents and passage metadata | 4 | Original fictional corpus, source filenames, passage IDs |
| Indexing and retrieval | 5 | Embeddings, saved index, retrieved passage evidence, reuse/rebuild |
| Grounded structured answers | 5 | Correct response shape, valid supplied citations, unsupported-question behaviour |
| Reproducibility and error handling | 3 | Setup works, missing/changed index produces a useful result, no committed key |
| Reflection and explanation | 3 | Learner can explain the flow and diagnose one failure |

AI coding tools are allowed. Be prepared to explain one retrieved passage and change one policy rule yourself.

## Optional stretch

Only after the required work is complete: compare two chunk sizes on the same question, or try a different `top_k`. Record what changed. These experiments are optional and receive no penalty if omitted. We will use session 10 to continue the core build before deciding whether there is time for quality work.
