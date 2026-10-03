import argparse
import json
import logging
import time
from typing import Literal
from uuid import uuid4

from langchain_core.documents import Document
from pydantic import BaseModel, Field, FiniteFloat, model_validator

from rag_core.config import load_settings
from rag_core.prompts import build_messages
from rag_core.providers import make_answer_model, offline_answer
from rag_core.rag import validate_answer
from rag_core.schemas import Answer


class Passage(BaseModel):
    passage_id: str
    source: str
    text: str = Field(min_length=1)
    score: FiniteFloat


class Evidence(BaseModel):
    status: Literal["found", "not_found", "unavailable"]
    passages: list[Passage]
    message: str = ""

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "found") != bool(self.passages):
            raise ValueError("Only a found result may contain passages, and it must contain some")
        return self


def evidence_from_hits(hits) -> Evidence:
    return Evidence(status="found" if hits else "not_found", passages=[
        Passage(passage_id=doc.metadata["chunk_id"], source=doc.metadata["source"],
                text=doc.page_content, score=score) for doc, score in hits
    ])


def hits_from_evidence(evidence: Evidence):
    return [(Document(page_content=p.text, metadata={"chunk_id": p.passage_id, "source": p.source}), p.score)
            for p in evidence.passages]


def pack_answer(answer: Answer, hits, offline: bool) -> dict:
    validate_answer(answer, hits)
    return {"mode": "offline" if offline else "live", **answer.model_dump(),
            "retrieved_sources": [{"passage_id": d.metadata["chunk_id"],
                                   "source": d.metadata["source"], "score": round(s, 4)} for d, s in hits]}


def insufficient_result(offline: bool) -> dict:
    return pack_answer(Answer(status="insufficient_evidence", answer="The supplied policies do not contain enough evidence.", citations=[]), [], offline)


def answer_from_evidence(question, evidence: Evidence, settings, offline: bool) -> dict:
    if evidence.status == "unavailable":
        return {"mode": "offline" if offline else "live", "status": "unavailable",
                "answer": evidence.message, "citations": [], "retrieved_sources": []}
    hits = hits_from_evidence(evidence)
    if not hits:
        return insufficient_result(offline)
    answer = offline_answer(hits) if offline else make_answer_model(settings).invoke(build_messages(question, hits))
    return pack_answer(answer, hits, offline)


def run_cli(action, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("question")
    parser.add_argument("--offline", action="store_true", help="Labelled fixture/keyword rehearsal; no live calls")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    request_id = uuid4().hex[:12]
    started = time.perf_counter()
    try:
        result = action(args.question, load_settings(), args.offline)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        logging.info("request_completed request_id=%s elapsed_ms=%.1f mode=%s", request_id,
                     (time.perf_counter() - started) * 1000, "offline" if args.offline else "live")
    except Exception as exc:
        logging.error("request_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        if isinstance(exc, (ValueError, FileNotFoundError)):
            parser.exit(1, f"Could not complete this request: {exc}\n")
        parser.exit(1, f"Check provider/network access. Reference {request_id}, error {type(exc).__name__}.\n")
