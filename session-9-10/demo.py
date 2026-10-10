import argparse
import json
import logging
import sys
import time
from uuid import uuid4

from config import load_settings
from rag import answer_question, build_index, load_documents, retrieve, split_documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Session 9: policy RAG, one step at a time")
    parser.add_argument("step", choices=["load", "chunk", "index", "search", "ask"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--offline", action="store_true", help="Keyword/excerpt rehearsal, not semantic RAG")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    request_id = uuid4().hex[:12]
    started = time.perf_counter()
    try:
        settings = load_settings()
        if args.step == "load":
            result = [{"source": d.metadata["source"], "text": d.page_content}
                      for d in load_documents(settings)]
        elif args.step == "chunk":
            result = [{**d.metadata, "characters": len(d.page_content), "text": d.page_content}
                      for d in split_documents(load_documents(settings), settings)]
        elif args.step == "index":
            result = build_index(settings, args.offline)
        else:
            if not args.question:
                parser.error("search and ask need a quoted question")
            if args.step == "search":
                result = [{**d.metadata, "score": round(s, 4), "text": d.page_content}
                          for d, s in retrieve(args.question, settings, args.offline)]
            else:
                result = answer_question(args.question, settings, args.offline)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        logging.info("request_completed request_id=%s step=%s mode=%s elapsed_ms=%.1f",
                     request_id, args.step, "offline" if args.offline else "live",
                     (time.perf_counter() - started) * 1000)
        return 0
    except Exception as exc:
        logging.error("request_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        if isinstance(exc, (ValueError, FileNotFoundError, UnicodeError)):
            print(f"Could not complete this step: {exc}", file=sys.stderr)
        else:
            print("The step failed. Check provider access, network, dependencies, or the local index. "
                  f"Reference: {request_id}. Error type: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
