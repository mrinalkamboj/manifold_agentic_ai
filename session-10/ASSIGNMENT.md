# Session 10 assignment: routed knowledge desk

## Goal

Extend your Session 9 policy assistant into a small LangGraph workflow. A single request chooses a policy, weather, or chat path. Show which nodes ran and what information source the response used.

Estimated effort: 2–3 hours using the supplied working starter. Continue your own fictional policy corpus where possible. A prepared corpus is available if your earlier assignment is incomplete.

## Required build

1. Create a validated route choice: `policy`, `weather`, or `chat`, and a Python `route()` that maps it to the allowed graph node.
2. Expose policy retrieval as a tool. Return passage IDs, filenames, text, and scores. The tool retrieves; the later answer step generates.
3. On the policy path, request the retrieval tool, execute it through ToolNode, and return a structured policy answer whose citation IDs were supplied in the evidence. Keep one retrieval round for this assignment.
4. On the weather path, call a read-only live HTTP tool. You may retain the class's Mumbai-only tool. Preserve the observation timestamp and report provider failure honestly.
5. On the chat path, produce a plain response without policy retrieval or weather tools.
6. Print the actual visited nodes and selected path, not a hard-coded expected path.
7. Distinguish `insufficient_evidence` from `unavailable`. A timeout or missing index is not proof that no policy exists.
8. Keep API keys outside source, and preserve index reuse/stale-document detection from Session 9.

The assignment handles one primary intent per request. You do not need multi-turn memory, multiple research agents, MCP integration, a web interface, or cloud deployment.

## Evidence

Submit a repository link or zip with source, fictional documents, dependencies, README, and output for:

- A supported policy question: actual graph path, retrieved source IDs, and structured answer.
- A Mumbai weather request: actual path and the live observation, or a clearly labelled fixture if access is unavailable.
- A plain chat request: actual path and no tool use.
- An unsupported policy question: an insufficient-evidence result, with a brief explanation of what is missing.
- One controlled source failure: unavailable result. A missing index in a temporary working copy or a simulated timeout is sufficient; do not break your only working environment.

Include a 5–8 sentence reflection: which route or data boundary failed first, what you inspected, and what changed.

Include a small comparison note after running the prepared examples: when the routed graph helps, what LCEL changes about composition, and what LlamaIndex's source nodes show. Three short paragraphs are enough. No independent LCEL or LlamaIndex application is required.

When access permits, include one live policy response and one live weather result. Label every offline output. Keyword matching, a fake tool request, a weather fixture, and MockLLM output verify plumbing only; they do not establish live model quality. If provider access is unavailable, state which live evidence remains pending.

Exclude `.env`, environments, caches, generated indexes, and unrelated files. A reviewer should be able to rebuild the index from your supplied corpus.

## Rubric: 20 points

| Area | Points | Evidence |
|---|---:|---|
| Routing and executed path | 4 | Real route mapping and node sequence for three paths |
| Retrieval tool and policy answer | 5 | ToolNode executes retrieval; IDs/text retained; supplied citations checked |
| Weather and chat boundaries | 4 | Live/fixture distinction, timestamp, no tools on chat path |
| Empty evidence and failed source | 3 | Insufficient evidence differs from unavailable |
| Reproducibility and explanation | 4 | README, source corpus, reflection, brief framework comparison |

AI coding tools are allowed. Be ready to explain one node and change one routing rule yourself.

## Optional only

Compare `top_k` values with `05_optional_quality.py`, or add a fourth route. These are stretch tasks. There is no penalty for skipping them. Finish the core workflow first.
