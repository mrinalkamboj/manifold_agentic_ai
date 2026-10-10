import hashlib
import math
import os
import re

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import Settings
from schemas import Answer

STOP_WORDS = set("a an the is are was were to of for in on and or with what how can i my we do does should when it this that our about please policy policies classroom demonstration fictional document documents supplied".split())


class OfflineKeywordEmbeddings(Embeddings):
    dimensions = 512

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in STOP_WORDS:
                continue
            bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.dimensions
            vector[bucket] += 1
        length = math.sqrt(sum(value * value for value in vector))
        return [value / length for value in vector] if length else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def embedding_identity(settings: Settings, offline: bool) -> str:
    return "offline-keyword-v1-512" if offline else settings.embedding_model


def make_embeddings(settings: Settings, offline: bool) -> Embeddings:
    if offline:
        return OfflineKeywordEmbeddings()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env, or use --offline to rehearse")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        request_timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        check_embedding_ctx_length=False,
    )


def make_answer_model(settings: Settings):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env before asking in live mode")
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    ).with_structured_output(Answer, method="json_schema")


def offline_answer(hits) -> Answer:
    document = hits[0][0]
    return Answer(
        status="answered",
        answer="OFFLINE REHEARSAL: retrieved excerpt:\n" + document.page_content,
        citations=[document.metadata["chunk_id"]],
    )
