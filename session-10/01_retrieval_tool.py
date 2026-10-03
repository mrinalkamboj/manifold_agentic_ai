from common import run_cli
from policy_tool import make_policy_tool
from rag_core.schemas import Question


def run(question, settings, offline):
    question = Question(text=question).text
    return {"mode": "offline" if offline else "live", "tool": "search_policies",
            **make_policy_tool(settings, offline).invoke({"query": question})}


if __name__ == "__main__":
    run_cli(run, "Step 1: call the retrieval tool directly")
