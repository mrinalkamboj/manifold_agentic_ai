import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import ValidationError

from config import ROOT, Settings
from providers import OfflineKeywordEmbeddings, make_answer_model, make_embeddings
from rag import (answer_question, atomic_save, build_index, index_path,
                 load_documents, read_snapshot, retrieve, split_documents)
from schemas import Answer, IndexSnapshot, VectorRecord


class RagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(index_dir=Path(self.temp.name) / "index")

    def build(self):
        return build_index(self.settings, offline=True)

    def test_chunks_preserve_source_and_stable_ids(self):
        docs = load_documents(self.settings)
        first = split_documents(docs, self.settings)
        second = split_documents(docs, self.settings)
        self.assertEqual([d.id for d in first], [d.id for d in second])
        for chunk in first:
            original = next(doc for doc in docs if doc.metadata["source"] == chunk.metadata["source"])
            start = chunk.metadata["start_index"]
            self.assertEqual(original.page_content[start:start + len(chunk.page_content)], chunk.page_content)
            self.assertLessEqual(len(chunk.page_content), self.settings.chunk_size)

    def test_saved_index_is_reused_without_embedding_documents(self):
        self.assertEqual(self.build()["status"], "built")
        with patch("providers.OfflineKeywordEmbeddings.embed_documents", side_effect=AssertionError("Unexpected rebuild")):
            self.assertEqual(self.build()["status"], "reused")

    def test_questions_do_not_reembed_the_corpus(self):
        self.build()
        with patch("providers.OfflineKeywordEmbeddings.embed_documents", side_effect=AssertionError("Unexpected corpus embedding")):
            hits = retrieve("refund estimate damaged item", self.settings, offline=True)
        self.assertTrue(any("80 percent" in doc.page_content for doc, score in hits))
        self.assertTrue(any(doc.metadata["source"] == "damaged-items.md" for doc, score in hits))

    def test_changed_document_blocks_query_until_rebuild(self):
        data_dir = Path(self.temp.name) / "policies"
        data_dir.mkdir()
        file = data_dir / "delivery.md"
        file.write_text("Late delivery investigation starts after 48 hours.")
        self.settings = self.settings.model_copy(update={"data_dir": data_dir})
        self.build()
        file.write_text("Late delivery investigation starts after 72 hours.")
        with self.assertRaisesRegex(ValueError, "changed"):
            retrieve("late delivery", self.settings, True)
        self.assertEqual(self.build()["status"], "built")
        hits = retrieve("late delivery", self.settings, True)
        self.assertIn("72 hours", hits[0][0].page_content)

    def test_chunk_setting_change_requires_rebuild_but_top_k_does_not(self):
        self.build()
        more_hits = self.settings.model_copy(update={"top_k": 5})
        retrieve("refund estimate damaged", more_hits, True)
        changed_chunks = self.settings.model_copy(update={"chunk_size": 600})
        with self.assertRaisesRegex(ValueError, "changed"):
            retrieve("refund estimate damaged", changed_chunks, True)

    def test_live_mode_cannot_use_the_offline_snapshot(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "Build the index first"):
            retrieve("damaged refund", self.settings, False)

    def test_empty_evidence_skips_generation(self):
        self.build()
        with patch("rag.make_answer_model", side_effect=AssertionError("Unexpected generation")):
            result = answer_question("What is the policy for lunar mining royalties?", self.settings, True)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["citations"], [])

    def test_live_generation_receives_passages_and_validates_citations(self):
        self.build()
        hits = retrieve("refund estimate damaged item", self.settings, True)
        citation = hits[0][0].metadata["chunk_id"]
        with patch("rag.retrieve", return_value=hits), patch("rag.make_answer_model") as factory:
            factory.return_value.invoke.return_value = Answer(status="answered", answer="An 80 percent estimate.", citations=[citation])
            result = answer_question("refund estimate damaged item", self.settings, False)
            messages = factory.return_value.invoke.call_args.args[0]
        self.assertEqual(result["citations"], [citation])
        supplied = json.loads(messages[1]["content"])
        self.assertEqual(supplied["question"], "refund estimate damaged item")
        self.assertTrue(all("passage_id" in item and "text" in item for item in supplied["retrieved_passages"]))
        self.assertIn("untrusted", messages[0]["content"])

    def test_invented_citation_is_rejected(self):
        self.build()
        hits = retrieve("refund estimate damaged item", self.settings, True)
        with patch("rag.retrieve", return_value=hits), patch("rag.make_answer_model") as factory:
            factory.return_value.invoke.return_value = Answer(status="answered", answer="A claim.", citations=["invented-id"])
            with self.assertRaisesRegex(ValueError, "not supplied"):
                answer_question("refund estimate damaged item", self.settings, False)

    def test_no_citations_for_insufficient_evidence(self):
        with self.assertRaises(ValidationError):
            Answer(status="insufficient_evidence", answer="Unknown", citations=["one"])
        with self.assertRaises(ValidationError):
            Answer(status="answered", answer="A claim", citations=[])

    def test_failed_atomic_save_preserves_previous_index(self):
        self.build()
        path = index_path(self.settings, True)
        before = path.read_bytes()
        snapshot = read_snapshot(path)
        with patch("rag.os.replace", side_effect=OSError("simulated interruption")):
            with self.assertRaises(OSError):
                atomic_save(path, snapshot)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_nonfinite_vector_and_inconsistent_dimensions_are_rejected(self):
        with self.assertRaises(ValidationError):
            VectorRecord(id="a", text="text", metadata={"chunk_id": "a"}, vector=[float("nan")])
        with self.assertRaises(ValidationError):
            IndexSnapshot(fingerprint="a", mode="offline", embedding_model="offline", records={
                "a": {"id": "a", "text": "a", "metadata": {"chunk_id": "a"}, "vector": [1.0]},
                "b": {"id": "b", "text": "b", "metadata": {"chunk_id": "b"}, "vector": [1.0, 0.0]},
            })

    def test_invalid_question_is_rejected_before_retrieval(self):
        with patch("rag.open_index", side_effect=AssertionError("Unexpected retrieval")):
            with self.assertRaises(ValidationError):
                answer_question("   ", self.settings, True)

    def test_empty_corpus_has_useful_error(self):
        data_dir = Path(self.temp.name) / "empty"
        data_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            load_documents(self.settings.model_copy(update={"data_dir": data_dir}))

    def test_live_adapters_have_bounded_timeout_and_retries(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-not-a-real-key"}), patch("providers.OpenAIEmbeddings") as embeddings, patch("providers.ChatOpenAI") as chat:
            make_embeddings(self.settings, False)
            make_answer_model(self.settings)
        self.assertEqual(embeddings.call_args.kwargs["request_timeout"], 30)
        self.assertEqual(embeddings.call_args.kwargs["max_retries"], 2)
        self.assertEqual(chat.call_args.kwargs["timeout"], 30)
        self.assertEqual(chat.call_args.kwargs["max_retries"], 2)

    def test_real_adapters_parse_mocked_http_end_to_end(self):
        calls = []
        keyword_vectors = OfflineKeywordEmbeddings()

        def respond(request):
            payload = json.loads(request.content)
            calls.append(request.url.path)
            if request.url.path.endswith("/embeddings"):
                inputs = payload["input"]
                if isinstance(inputs, str):
                    inputs = [inputs]
                return httpx.Response(200, json={"object": "list", "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                    "data": [{"object": "embedding", "index": i, "embedding": keyword_vectors.embed_query(text)}
                             for i, text in enumerate(inputs)]})
            if request.url.path.endswith("/chat/completions"):
                evidence = json.loads(payload["messages"][-1]["content"])["retrieved_passages"]
                cited = next(item for item in evidence if "80 percent" in item["text"])
                result = {"status": "answered", "answer": "The estimate is 80 percent.", "citations": [cited["passage_id"]]}
                return httpx.Response(200, json={"id": "fixture-chat", "object": "chat.completion",
                    "created": 0, "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result), "refusal": None}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}})
            raise AssertionError(f"Unexpected URL {request.url.path}")

        client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)

        def embedding_client(**kwargs):
            return OpenAIEmbeddings(**kwargs, api_key="fixture-only", http_client=client)

        def chat_client(**kwargs):
            return ChatOpenAI(**kwargs, api_key="fixture-only", http_client=client)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-only"}), patch("providers.OpenAIEmbeddings", side_effect=embedding_client), patch("providers.ChatOpenAI", side_effect=chat_client):
            build_index(self.settings, False)
            result = answer_question("refund estimate damaged item", self.settings, False)
        self.assertEqual(result["mode"], "live")
        self.assertEqual(result["answer"], "The estimate is 80 percent.")
        self.assertEqual(calls.count("/v1/chat/completions"), 1)
        self.assertGreaterEqual(calls.count("/v1/embeddings"), 2)


if __name__ == "__main__":
    unittest.main()
