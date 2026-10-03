from common import run_cli
from rag_core.rag import retrieve


def run(question, settings, offline):
    return {"optional": True, "mode": "offline" if offline else "live", "question": question,
            "observations": [{"top_k": k, "passages": [
                {"source": doc.metadata["source"], "passage_id": doc.metadata["chunk_id"],
                 "score": round(score, 4), "text": doc.page_content}
                for doc, score in retrieve(question, settings.model_copy(update={"top_k": k}), offline)]}
                for k in (1, 3, 5)],
            "note": "Top-k changes how many candidates may return. Inspect evidence; a larger number does not guarantee a better answer."}


if __name__ == "__main__":
    run_cli(run, "Optional only: compare retrieved passages with different top-k values")
