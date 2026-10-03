import logging

from langchain_core.tools import tool

from common import Evidence, evidence_from_hits
from rag_core.rag import retrieve


def make_policy_tool(settings, offline: bool = False):
    @tool
    def search_policies(query: str) -> dict:
        """Find evidence in the fictional payments policies for a short question.
        Return source IDs and passage text. Does not issue refunds or generate an answer.
        """
        try:
            return evidence_from_hits(retrieve(query, settings, offline)).model_dump()
        except Exception as exc:
            logging.getLogger("policy_tool").warning("retrieval_failed error_type=%s", type(exc).__name__)
            return Evidence(status="unavailable", passages=[],
                message="Policy retrieval is unavailable. Check the index, run prepare.py in the same mode, and verify provider access.").model_dump()
    return search_policies
