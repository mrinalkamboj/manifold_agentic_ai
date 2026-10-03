import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding as NativeEmbedding
from llama_index.llms.openai import OpenAI as NativeLLM
from pydantic import ValidationError

from common import Evidence, answer_from_evidence, evidence_from_hits, pack_answer
from lcel_pipeline import run_chain
from llama_comparison import run_native
from policy_tool import make_policy_tool
from rag_core.config import Settings
from rag_core.providers import OfflineKeywordEmbeddings
from rag_core.rag import build_index, index_path, retrieve
from rag_core.schemas import Answer
from weather_tool import fetch_mumbai_weather, make_weather_tool
from workflow import PathChoice, run_workflow


POLICY = "What is the refund estimate for a damaged item?"
UNKNOWN = "What is the policy for lunar mining royalties?"
POLICY_NODES = ["classify", "policy_assistant", "tools", "policy_answer"]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(index_dir=Path(self.temp.name) / "index")

    def build(self):
        return build_index(self.settings, offline=True)

    def test_tool_returns_passages_without_answer_generation(self):
        self.build()
        with patch("common.make_answer_model", side_effect=AssertionError("No generation expected")):
            output = make_policy_tool(self.settings, True).invoke({"query": POLICY})
        evidence = Evidence.model_validate(output)
        self.assertEqual(evidence.status, "found")
        self.assertTrue(any("80 percent" in p.text for p in evidence.passages))
        self.assertNotIn("answer", output)

    def test_policy_graph_executes_one_tool_round_and_checks_sources(self):
        self.build()
        with patch("rag_core.providers.OfflineKeywordEmbeddings.embed_documents",
                   side_effect=AssertionError("No corpus embedding during questions")):
            output = run_workflow(POLICY, self.settings, True)
        self.assertEqual(output["nodes"], POLICY_NODES)
        self.assertEqual(output["path"], "policy")
        self.assertEqual(output["status"], "answered")
        self.assertIn("OFFLINE REHEARSAL", output["answer"])
        supplied = {p["passage_id"] for p in output["retrieved_sources"]}
        self.assertTrue(set(output["citations"]).issubset(supplied))

    def test_successful_empty_search_is_insufficient(self):
        self.build()
        evidence = Evidence.model_validate(make_policy_tool(self.settings, True).invoke({"query": UNKNOWN}))
        self.assertEqual(evidence.status, "not_found")
        with patch("common.offline_answer", side_effect=AssertionError("No generation without evidence")):
            output = run_workflow(UNKNOWN, self.settings, True)
        self.assertEqual(output["status"], "insufficient_evidence")
        self.assertEqual(output["citations"], [])

    def test_missing_index_is_unavailable_not_empty_evidence(self):
        with patch("common.offline_answer", side_effect=AssertionError("No generation on source failure")):
            output = run_workflow(POLICY, self.settings, True)
        self.assertEqual(output["status"], "unavailable")
        self.assertEqual(output["nodes"], POLICY_NODES)
        self.assertIn("prepare.py", output["answer"])

    def test_chat_and_weather_need_no_policy_index(self):
        with patch("policy_tool.retrieve", side_effect=AssertionError("Unexpected policy access")):
            chat = run_workflow("Explain recursion in one sentence.", self.settings, True)
            weather = run_workflow("What is the current temperature in Mumbai?", self.settings, True)
        self.assertEqual(chat["nodes"], ["classify", "chat"])
        self.assertEqual(weather["nodes"], ["classify", "weather"])
        self.assertTrue(weather["weather"]["fixture"])
        self.assertEqual(weather["weather"]["observed_at"], "DEMO-FIXTURE")
        self.assertIn("OFFLINE REHEARSAL", weather["answer"])
        self.assertIn("no model-generated answer", chat["answer"])

    def test_unsupported_city_does_not_fetch_mumbai_weather(self):
        with patch("weather_tool.fetch_mumbai_weather", side_effect=AssertionError("Wrong city request")):
            output = run_workflow("What is the weather in Delhi?", self.settings, True)
        self.assertEqual(output["status"], "unavailable")
        self.assertIn("Mumbai only", output["answer"])

    def test_weather_tool_checks_units_shape_and_timeout(self):
        payload = {"current": {"time": "2026-10-02T08:15", "temperature_2m": 30.2},
                   "current_units": {"temperature_2m": "°C"}}
        response = io.BytesIO(json.dumps(payload).encode())
        response.status = 200
        with patch("weather_tool.urlopen", return_value=response) as request:
            result = make_weather_tool(self.settings).invoke({})
        self.assertEqual(request.call_args.kwargs["timeout"], 10)
        self.assertFalse(result["fixture"])
        self.assertEqual(result["observed_at"], "2026-10-02T08:15")
        self.assertEqual(result["temperature_c"], 30.2)

    def test_weather_bad_units_nonfinite_and_timeout_do_not_fabricate_value(self):
        for unit, temperature in (("°F", 80), ("°C", float("nan"))):
            with self.subTest(unit=unit, temperature=temperature):
                payload = {"current": {"time": "fixture", "temperature_2m": temperature},
                           "current_units": {"temperature_2m": unit}}
                response = io.BytesIO(json.dumps(payload).encode())
                response.status = 200
                with patch("weather_tool.urlopen", return_value=response):
                    result = make_weather_tool(self.settings).invoke({})
                self.assertEqual(result["status"], "unavailable")
                self.assertNotIn("temperature_c", result)
        with patch("weather_tool.urlopen", side_effect=TimeoutError("fixture network timeout")):
            result = make_weather_tool(self.settings).invoke({})
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("temperature_c", result)

    def test_tool_contract_rejects_inconsistent_status(self):
        self.build()
        passage = evidence_from_hits(retrieve(POLICY, self.settings, True)).passages[0]
        with self.assertRaises(ValidationError):
            Evidence(status="found", passages=[])
        with self.assertRaises(ValidationError):
            Evidence(status="unavailable", passages=[passage])

    def test_invented_citation_is_rejected(self):
        self.build()
        answer = Answer(status="answered", answer="A policy claim", citations=["invented-id"])
        with self.assertRaisesRegex(ValueError, "not supplied"):
            pack_answer(answer, retrieve(POLICY, self.settings, True), True)

    def test_blank_question_rejected_before_graph_calls(self):
        with patch("workflow.build_workflow", side_effect=AssertionError("Invalid input reached graph")):
            with self.assertRaises(ValidationError):
                run_workflow("   ", self.settings, True)

    def test_disallowed_route_rejected(self):
        with self.assertRaises(ValidationError):
            PathChoice(path="issue_refund")

    def test_live_policy_rejects_multiple_wrong_named_and_extra_argument_calls(self):
        cases = [
            [{"name": "other_tool", "args": {"query": POLICY}, "id": "bad"}],
            [{"name": "search_policies", "args": {"query": POLICY, "extra": True}, "id": "bad"}],
            [{"name": "search_policies", "args": {"query": POLICY}, "id": str(i)} for i in range(2)],
            [],
        ]
        for calls in cases:
            with self.subTest(calls=calls), patch("workflow.choose_path", return_value=PathChoice(path="policy")), \
                 patch("workflow.make_chat_model") as model, patch("policy_tool.retrieve") as retrieval:
                model.return_value.bind_tools.return_value.invoke.return_value = AIMessage(content="", tool_calls=calls)
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    run_workflow(POLICY, self.settings, False)
                retrieval.assert_not_called()

    def test_original_question_is_preserved_after_query_rewrite(self):
        self.build()
        hits = retrieve(POLICY, self.settings, True)
        answer = {"status": "answered", "answer": "fixture", "citations": []}
        with patch("workflow.choose_path", return_value=PathChoice(path="policy")), \
             patch("workflow.make_chat_model") as model, patch("policy_tool.retrieve", return_value=hits) as retrieval, \
             patch("workflow.answer_from_evidence", return_value=answer) as generation:
            model.return_value.bind_tools.return_value.invoke.return_value = AIMessage(content="", tool_calls=[
                {"name": "search_policies", "args": {"query": "damaged refund estimate"}, "id": "rewrite"}])
            run_workflow(POLICY, self.settings, False)
        self.assertEqual(retrieval.call_args.args[0], "damaged refund estimate")
        self.assertEqual(generation.call_args.args[0], POLICY)

    def test_lcel_matches_policy_evidence_and_skips_empty_generation(self):
        self.build()
        graph = run_workflow(POLICY, self.settings, True)
        chain = run_chain(POLICY, self.settings, True)
        self.assertEqual(chain["pipeline"], "lcel")
        self.assertEqual(chain["retrieved_sources"], graph["retrieved_sources"])
        self.assertEqual(chain["citations"], graph["citations"])
        with patch("lcel_pipeline.offline_answer", side_effect=AssertionError("No generation without hits")):
            self.assertEqual(run_chain(UNKNOWN, self.settings, True)["status"], "insufficient_evidence")

    def test_lcel_rejects_unknown_citation_at_final_boundary(self):
        self.build()
        bad = Answer(status="answered", answer="A claim", citations=["invented-id"])
        with patch("lcel_pipeline.offline_answer", return_value=bad):
            with self.assertRaisesRegex(ValueError, "not supplied"):
                run_chain(POLICY, self.settings, True)

    def test_native_retrieval_reuses_vectors_and_passage_ids(self):
        self.build()
        expected = {doc.metadata["chunk_id"] for doc, _ in retrieve(POLICY, self.settings, True)}
        with patch("llama_comparison.OfflineLlamaEmbedding._get_text_embedding",
                   side_effect=AssertionError("No new document embedding")), \
             patch("rag_core.providers.OfflineKeywordEmbeddings.embed_documents",
                   side_effect=AssertionError("No new document embedding")):
            result = run_native(POLICY, self.settings, True, search_only=True)
        self.assertEqual({p["passage_id"] for p in result["passages"]}, expected)

    def test_native_query_engine_labels_mock_output_and_different_contract(self):
        self.build()
        result = run_native(POLICY, self.settings, True)
        self.assertEqual(result["status"], "response_with_retrieved_sources")
        self.assertIn("native MockLLM", result["answer"])
        self.assertIn("does not apply", result["contract_note"])
        self.assertNotIn("citations", result)
        empty = run_native(UNKNOWN, self.settings, True)
        self.assertEqual(empty["status"], "insufficient_evidence")

    def test_native_offline_rehearsal_needs_no_default_tokenizer_download(self):
        self.build()
        from llama_index.core import Settings as NativeSettings
        with patch.object(NativeSettings, "_tokenizer", None), \
             patch("tiktoken.encoding_for_model", side_effect=AssertionError("No default tokenizer in offline mode")):
            result = run_native(POLICY, self.settings, True)
        self.assertEqual(result["status"], "response_with_retrieved_sources")

    def test_real_langchain_adapters_run_graph_and_lcel_with_simulated_http(self):
        payloads = []
        vectors = OfflineKeywordEmbeddings()

        def respond(request):
            body = json.loads(request.content)
            payloads.append((request.url.path, body))
            if request.url.path.endswith("/embeddings"):
                texts = body["input"]
                if isinstance(texts, str):
                    texts = [texts]
                return httpx.Response(200, json={"object": "list", "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 10, "total_tokens": 10}, "data": [
                        {"object": "embedding", "index": i, "embedding": vectors.embed_query(text)}
                        for i, text in enumerate(texts)]})
            if request.url.path.endswith("/chat/completions"):
                if body.get("tools"):
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "fixture-call", "type": "function", "function": {
                            "name": "search_policies", "arguments": json.dumps({"query": "damaged refund estimate"})}}]}
                    finish = "tool_calls"
                elif body["response_format"]["json_schema"]["name"] == "PathChoice":
                    message = {"role": "assistant", "content": json.dumps({"path": "policy"}), "refusal": None}
                    finish = "stop"
                else:
                    data = json.loads(body["messages"][-1]["content"])
                    self.assertEqual(data["question"], POLICY)
                    cited = next(p for p in data["retrieved_passages"] if "80 percent" in p["text"])
                    message = {"role": "assistant", "content": json.dumps({"status": "answered",
                        "answer": "The estimate is 80 percent.", "citations": [cited["passage_id"]]}), "refusal": None}
                    finish = "stop"
                return httpx.Response(200, json={"id": "fixture-chat", "object": "chat.completion", "created": 0,
                    "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}})
            raise AssertionError(f"Unexpected URL: {request.url}")

        client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)

        def embedding_client(**kwargs):
            return OpenAIEmbeddings(**kwargs, api_key="fixture-only", http_client=client)

        def chat_client(**kwargs):
            return ChatOpenAI(**kwargs, api_key="fixture-only", http_client=client)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-only"}), \
             patch("rag_core.providers.OpenAIEmbeddings", side_effect=embedding_client), \
             patch("rag_core.providers.ChatOpenAI", side_effect=chat_client), \
             patch("workflow.ChatOpenAI", side_effect=chat_client):
            build_index(self.settings, False)
            graph = run_workflow(POLICY, self.settings, False)
            chain = run_chain(POLICY, self.settings, False)
        self.assertEqual(graph["nodes"], POLICY_NODES)
        self.assertEqual(graph["answer"], "The estimate is 80 percent.")
        self.assertEqual(chain["answer"], graph["answer"])
        chat_calls = [p for path, p in payloads if path.endswith("/chat/completions")]
        self.assertEqual(len(chat_calls), 4)
        tool_request = next(p for p in chat_calls if p.get("tools"))
        self.assertEqual(tool_request["tool_choice"]["function"]["name"], "search_policies")
        self.assertFalse(tool_request["parallel_tool_calls"])

    def test_real_native_adapters_query_engine_with_simulated_http(self):
        self.build()
        offline_snapshot = json.loads(index_path(self.settings, True).read_text())
        offline_snapshot.update(mode="live", embedding_model=self.settings.embedding_model)
        from rag_core.rag import corpus_fingerprint, load_documents
        offline_snapshot["fingerprint"] = corpus_fingerprint(load_documents(self.settings), self.settings, False)
        index_path(self.settings, False).write_text(json.dumps(offline_snapshot))
        calls = []

        def respond(request):
            calls.append(request.url.path)
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(200, json={"object": "list", "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1}, "data": [{"object": "embedding",
                    "index": 0, "embedding": [1.0] + [0.0] * 511}]})
            body = json.loads(request.content)
            self.assertIn("context", body["messages"][-1]["content"].lower())
            return httpx.Response(200, json={"id": "fixture-native", "object": "chat.completion", "created": 0,
                "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "Simulated native engine answer.", "refusal": None}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})

        client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-only"}), \
             patch("llama_comparison.OpenAIEmbedding", side_effect=lambda **kw: NativeEmbedding(**kw, http_client=client)), \
             patch("llama_comparison.OpenAI", side_effect=lambda **kw: NativeLLM(**kw, http_client=client)):
            result = run_native(POLICY, self.settings.model_copy(update={"min_score": 0}), False)
        self.assertEqual(result["answer"], "Simulated native engine answer.")
        self.assertEqual(result["status"], "response_with_retrieved_sources")
        self.assertEqual(calls.count("/v1/embeddings"), 1)  # query only, documents already have vectors
        self.assertEqual(calls.count("/v1/chat/completions"), 1)


if __name__ == "__main__":
    unittest.main()
