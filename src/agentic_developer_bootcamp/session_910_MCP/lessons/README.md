# Session 7-8 lessons — routing and tools

Same incident-desk use case as earlier sessions. One idea per file. Copy `.env.example` to `.env` and put your OpenAI key in it.

From this folder, with the virtual environment active:

```bash
python 01_conditional_nodes.py
python 02_tools_local.py
python incident_tools.py
python 04_tools_from_module.py
python 05_tools_live_api.py
python 07_tools_mcp.py
python 08_full_conditional_demo.py
python 09_integrated_demo.py
```

| File | What you should see |
| --- | --- |
| `01_conditional_nodes.py` | You write the router. Ordinary Python `if/else` picks the next node. |
| `02_tools_local.py` | Tool lives in the same file. `tools_condition` routes if the model asked for a tool. |
| `incident_tools.py` | Tools as a module. No graph. Run it to call the functions yourself. |
| `04_tools_from_module.py` | Same graph as 02, but tools are imported. |
| `05_tools_live_api.py` | A tool that calls a real HTTP API (Open-Meteo, no extra key). |
| `mcp_server.py` | FastMCP server. 07 starts this for you over stdio. |
| `07_tools_mcp.py` | Tools loaded from the MCP server. Must use `ainvoke`. |
| `08_full_conditional_demo.py` | You classify the ask, then each path uses a different tool style. |
| `09_integrated_demo.py` | Both routers in one graph. Local + module + live API + MCP tools. |

Two routers show up in this folder:

1. **You route** — a function you write, like `route()` in 01 and 08.
2. **The model routes** — last AI message has `tool_calls`, so `tools_condition` goes to `tools`.

## Git Repo
### git clone https://github.com/manifoldailearning/agentic-developer-bootcamp.git
