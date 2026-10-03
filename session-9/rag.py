import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings
from prompts import build_messages
from providers import embedding_identity, make_answer_model, make_embeddings, offline_answer
from schemas import Answer, IndexSnapshot, Question

logger = logging.getLogger("rag")


def load_documents(settings: Settings) -> list[Document]:
    if not settings.data_dir.is_dir():
        raise ValueError(f"Document folder does not exist: {settings.data_dir}")
    paths = sorted(p for p in settings.data_dir.iterdir() if p.suffix in {".md", ".txt"} and p.is_file())
    if not paths or len(paths) > 100:
        raise ValueError("Use between 1 and 100 .md or .txt documents")
    documents = []
    for path in paths:
        if path.is_symlink() or path.stat().st_size > 200_000:
            raise ValueError(f"Use a regular text file smaller than 200 KB: {path.name}")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(page_content=text, metadata={"source": path.name}))
    if not documents:
        raise ValueError("The document files contain no text")
    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:10]
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['source']}:{chunk.metadata['start_index']}:{content_hash}"
        )
        chunk.id = chunk.metadata["chunk_id"]
    return chunks


def corpus_fingerprint(documents, settings: Settings, offline: bool) -> str:
    payload = {
        "format": 1, "splitter": "recursive-character-v1",
        "embedding": embedding_identity(settings, offline),
        "chunk_size": settings.chunk_size, "overlap": settings.chunk_overlap,
        "documents": [(doc.metadata["source"], doc.page_content) for doc in documents],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def index_path(settings: Settings, offline: bool) -> Path:
    return settings.index_dir / ("offline.json" if offline else "live.json")


def read_snapshot(path: Path) -> IndexSnapshot:
    if path.stat().st_size > 50_000_000:
        raise ValueError("Index exceeds the small local demo's 50 MB limit")
    return IndexSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def atomic_save(path: Path, snapshot: IndexSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(snapshot.model_dump_json())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def build_index(settings: Settings, offline: bool = False) -> dict:
    documents = load_documents(settings)
    fingerprint = corpus_fingerprint(documents, settings, offline)
    path = index_path(settings, offline)
    if path.exists() and read_snapshot(path).fingerprint == fingerprint:
        return {"status": "reused", "index": str(path), "mode": "offline" if offline else "live"}
    chunks = split_documents(documents, settings)
    store = InMemoryVectorStore(make_embeddings(settings, offline))
    store.add_documents(chunks, ids=[chunk.id for chunk in chunks])
    snapshot = IndexSnapshot(
        fingerprint=fingerprint, mode="offline" if offline else "live",
        embedding_model=embedding_identity(settings, offline), records=store.store,
    )
    atomic_save(path, snapshot)
    logger.info("index_built mode=%s documents=%s chunks=%s", snapshot.mode, len(documents), len(chunks))
    return {"status": "built", "documents": len(documents), "chunks": len(chunks),
            "index": str(path), "mode": snapshot.mode}


def open_index(settings: Settings, offline: bool = False) -> InMemoryVectorStore:
    path = index_path(settings, offline)
    if not path.exists():
        raise ValueError("Build the index first: python demo.py index" + (" --offline" if offline else ""))
    snapshot = read_snapshot(path)
    expected = corpus_fingerprint(load_documents(settings), settings, offline)
    if (snapshot.fingerprint != expected or snapshot.embedding_model != embedding_identity(settings, offline)
            or snapshot.mode != ("offline" if offline else "live")):
        raise ValueError("Documents or index configuration changed. Run the index command again")
    store = InMemoryVectorStore(make_embeddings(settings, offline))
    store.store = {key: record.model_dump() for key, record in snapshot.records.items()}
    return store


def retrieve(question: str, settings: Settings, offline: bool = False):
    question = Question(text=question).text
    store = open_index(settings, offline)
    hits = store.similarity_search_with_score(question, k=settings.top_k)
    return [(doc, float(score)) for doc, score in hits if score >= settings.min_score]


def validate_answer(answer: Answer, hits) -> Answer:
    supplied_ids = {doc.metadata["chunk_id"] for doc, score in hits}
    if not set(answer.citations).issubset(supplied_ids):
        raise ValueError("The model cited a passage that was not supplied")
    return answer


def answer_question(question: str, settings: Settings, offline: bool = False) -> dict:
    question = Question(text=question).text
    hits = retrieve(question, settings, offline)
    if not hits:
        answer = Answer(status="insufficient_evidence",
                        answer="I do not have enough evidence in the supplied policies to answer that.",
                        citations=[])
    elif offline:
        answer = offline_answer(hits)
    else:
        answer = make_answer_model(settings).invoke(build_messages(question, hits))
    answer = validate_answer(answer, hits)
    return {"mode": "offline" if offline else "live", **answer.model_dump(),
            "retrieved_sources": [
                {"chunk_id": doc.metadata["chunk_id"], "source": doc.metadata["source"],
                 "score": round(score, 4)} for doc, score in hits
            ]}
