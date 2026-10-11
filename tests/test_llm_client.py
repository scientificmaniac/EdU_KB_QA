"""Ollama 客户端的单元测试（全部使用 Mock，不依赖真实服务）。"""

from __future__ import annotations

import json
import unittest
from collections import deque

import requests

from career_qa.llm_client import (
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMResponseError,
    OllamaClient,
    iter_ollama_chunks,
)


class FakeResponse:
    """模拟 requests.Response 的最小实现。"""

    def __init__(self, status_code=200, payload=None, lines=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self._lines = list(lines or [])
        self.text = text
        #: 记录是否被 close()，用于验证连接有没有归还
        self.closed = False

    def json(self):
        if self._payload is None:
            raise ValueError("响应体不是 JSON")
        return self._payload

    def iter_lines(self, *args, **kwargs):
        # 真实签名带 chunk_size / decode_unicode，实现里若传参不应导致 TypeError
        del args, kwargs
        yield from self._lines

    def close(self):
        self.closed = True


class FakeSession:
    """模拟 requests.Session，可预置响应或异常。"""

    def __init__(self):
        self.post_calls: list[tuple[str, dict]] = []
        self.get_calls: list[tuple[str, dict]] = []
        self.post_queue: deque = deque()
        self.get_queue: deque = deque()
        self.default_post = FakeResponse(200, {"message": {"content": "默认回答"}})

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if self.post_queue:
            item = self.post_queue.popleft()
        else:
            item = self.default_post
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        item = self.get_queue.popleft() if self.get_queue else self.default_post
        if isinstance(item, Exception):
            raise item
        return item


def ndjson(*chunks: dict) -> list[bytes]:
    """把若干响应块编码为 Ollama 风格的 NDJSON 行。"""
    return [json.dumps(c, ensure_ascii=False).encode("utf-8") for c in chunks]


def make_client(session=None, **kwargs) -> OllamaClient:
    params = {
        "base_url": "http://localhost:11434",
        "model": "llama2",
        "max_retries": 0,
        "sleep": lambda _seconds: None,
    }
    params.update(kwargs)
    return OllamaClient(session=session if session is not None else FakeSession(), **params)


# ----------------------------------------------------------------------
# 流式解析（原实现把 NDJSON 误当成 SSE，导致流式永远为空）
# ----------------------------------------------------------------------
class IterChunksTests(unittest.TestCase):
    def test_parses_ndjson(self):
        lines = ndjson(
            {"message": {"content": "你"}, "done": False},
            {"message": {"content": "好"}, "done": False},
            {"message": {"content": ""}, "done": True},
        )
        self.assertEqual(list(iter_ollama_chunks(lines)), ["你", "好"])

    def test_tolerates_sse_prefix(self):
        """部分网关会在 NDJSON 前加 ``data: ``，解析时需兼容。"""
        lines = [
            b'data: {"message": {"content": "A"}, "done": false}',
            b'data: {"message": {"content": "B"}, "done": true}',
        ]
        self.assertEqual(list(iter_ollama_chunks(lines)), ["A", "B"])

    def test_skips_blank_and_broken_lines(self):
        lines = [b"", b"   ", b"not-json", b'{"message": {"content": "ok"}, "done": true}']
        self.assertEqual(list(iter_ollama_chunks(lines)), ["ok"])

    def test_stops_at_done(self):
        lines = ndjson(
            {"message": {"content": "A"}, "done": False},
            {"message": {"content": "B"}, "done": True},
            {"message": {"content": "C"}, "done": False},
        )
        self.assertEqual(list(iter_ollama_chunks(lines)), ["A", "B"])

    def test_accepts_str_lines(self):
        self.assertEqual(list(iter_ollama_chunks(['{"message": {"content": "x"}}'])), ["x"])

    def test_raises_on_error_field(self):
        with self.assertRaises(LLMResponseError):
            list(iter_ollama_chunks(ndjson({"error": "model not found"})))

    def test_ignores_done_marker(self):
        self.assertEqual(list(iter_ollama_chunks(["data: [DONE]"])), [])


# ----------------------------------------------------------------------
# 缓存键（原实现 `tuple(history)` 含 dict，第二轮必然 TypeError）
# ----------------------------------------------------------------------
class CacheTests(unittest.TestCase):
    def test_multi_turn_history_does_not_crash(self):
        """回归测试：带历史的多轮请求不得抛出 unhashable 异常。"""
        session = FakeSession()
        client = make_client(session)

        first = client.chat("第一个问题", [])
        history = [
            {"role": "user", "content": "第一个问题"},
            {"role": "assistant", "content": first},
        ]
        # 修复前这一行会抛 TypeError: unhashable type: 'dict'
        second = client.chat("第二个问题", history)
        self.assertIsInstance(second, str)
        self.assertEqual(len(session.post_calls), 2)

    def test_repeated_request_hits_cache(self):
        session = FakeSession()
        client = make_client(session)
        client.chat("同样的问题", [])
        client.chat("同样的问题", [])
        self.assertEqual(len(session.post_calls), 1)

    def test_different_history_is_different_key(self):
        client = make_client(FakeSession())
        self.assertNotEqual(
            client._cache_key("问题", []),
            client._cache_key("问题", [{"role": "user", "content": "别的"}]),
        )
        # 相同输入必须得到相同的键，否则缓存永远不会命中
        self.assertEqual(
            client._cache_key("问题", [{"role": "user", "content": "别的"}]),
            client._cache_key("问题", [{"role": "user", "content": "别的"}]),
        )

    def test_history_change_bypasses_cache(self):
        session = FakeSession()
        client = make_client(session)
        client.chat("问题", [])
        client.chat("问题", [{"role": "user", "content": "别的"}])
        self.assertEqual(len(session.post_calls), 2)

    def test_cache_disabled(self):
        session = FakeSession()
        client = make_client(session, cache_size=0)
        client.chat("问题", [])
        client.chat("问题", [])
        self.assertEqual(len(session.post_calls), 2)

    def test_lru_eviction(self):
        session = FakeSession()
        client = make_client(session, cache_size=2)
        client.chat("q1", [])
        client.chat("q2", [])
        client.chat("q3", [])
        self.assertEqual(client.cache_info, {"size": 2, "maxsize": 2})
        self.assertEqual(len(session.post_calls), 3)

        # 行为断言：q1 已被淘汰，再问一次必须重新访问服务端
        # （只检查缓存键字符串是不牢靠的，改错淘汰策略也可能照样通过）
        client.chat("q1", [])
        self.assertEqual(len(session.post_calls), 4)

        # q3 仍在缓存中，不应产生新的请求
        client.chat("q3", [])
        self.assertEqual(len(session.post_calls), 4)

    def test_clear_cache(self):
        session = FakeSession()
        client = make_client(session)
        client.chat("问题", [])
        client.clear_cache()
        client.chat("问题", [])
        self.assertEqual(len(session.post_calls), 2)


# ----------------------------------------------------------------------
# 请求 / 响应
# ----------------------------------------------------------------------
class ChatTests(unittest.TestCase):
    def test_returns_content(self):
        session = FakeSession()
        session.post_queue.append(
            FakeResponse(200, {"message": {"content": "你好"}})
        )
        client = make_client(session)
        self.assertEqual(client.chat("hi", []), "你好")

    def test_payload_shape(self):
        session = FakeSession()
        client = make_client(session)
        client.chat("问题", [{"role": "user", "content": "上一句"}])
        url, kwargs = session.post_calls[0]
        self.assertTrue(url.endswith("/api/chat"))
        payload = kwargs["json"]
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "llama2")
        self.assertIn("temperature", payload["options"])
        # 历史在前，新问题在后
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "问题"})
        self.assertEqual(payload["messages"][0]["content"], "上一句")

    def test_timeout_tuple_is_passed(self):
        session = FakeSession()
        client = make_client(session, connect_timeout=3.0, request_timeout=99.0)
        client.chat("问题", [])
        self.assertEqual(session.post_calls[0][1]["timeout"], (3.0, 99.0))

    def test_missing_content_raises(self):
        session = FakeSession()
        session.post_queue.append(FakeResponse(200, {"done": True}))
        client = make_client(session)
        with self.assertRaises(LLMResponseError):
            client.chat("问题", [])

    def test_error_field_raises(self):
        session = FakeSession()
        session.post_queue.append(FakeResponse(200, {"error": "boom"}))
        client = make_client(session)
        with self.assertRaises(LLMResponseError):
            client.chat("问题", [])

    def test_non_json_body_raises(self):
        session = FakeSession()
        session.post_queue.append(FakeResponse(200, None))
        client = make_client(session)
        with self.assertRaises(LLMResponseError):
            client.chat("问题", [])


class StreamTests(unittest.TestCase):
    def test_stream_chat_concatenates(self):
        session = FakeSession()
        session.post_queue.append(
            FakeResponse(
                200,
                lines=ndjson(
                    {"message": {"content": "职业"}, "done": False},
                    {"message": {"content": "规划"}, "done": True},
                ),
            )
        )
        client = make_client(session)
        self.assertEqual(list(client.stream_chat("问题", [])), ["职业", "规划"])
        self.assertTrue(session.post_calls[0][1]["stream"])

    def test_generate_response_compat_non_stream(self):
        session = FakeSession()
        session.post_queue.append(FakeResponse(200, {"message": {"content": "X"}}))
        client = make_client(session)
        self.assertEqual(client.generate_response("q"), "X")

    def test_generate_response_compat_stream(self):
        session = FakeSession()
        session.post_queue.append(
            FakeResponse(
                200,
                lines=ndjson(
                    {"message": {"content": "a"}, "done": False},
                    {"message": {"content": "b"}, "done": True},
                ),
            )
        )
        client = make_client(session)
        self.assertEqual(client.generate_response("q", stream=True), "ab")


# ----------------------------------------------------------------------
# 错误与重试
# ----------------------------------------------------------------------
class ErrorTests(unittest.TestCase):
    def test_connection_error_wrapped(self):
        session = FakeSession()
        session.post_queue.append(requests.ConnectionError("refused"))
        client = make_client(session)
        with self.assertRaises(LLMConnectionError) as ctx:
            client.chat("问题", [])
        self.assertIn("Ollama", str(ctx.exception))

    def test_retries_then_succeeds(self):
        session = FakeSession()
        session.post_queue.append(requests.ConnectionError("refused"))
        session.post_queue.append(FakeResponse(200, {"message": {"content": "恢复"}}))
        client = make_client(session, max_retries=1)
        self.assertEqual(client.chat("问题", []), "恢复")
        self.assertEqual(len(session.post_calls), 2)

    def test_retry_exhausted(self):
        session = FakeSession()
        for _ in range(3):
            session.post_queue.append(requests.ConnectionError("refused"))
        client = make_client(session, max_retries=2)
        with self.assertRaises(LLMConnectionError):
            client.chat("问题", [])
        self.assertEqual(len(session.post_calls), 3)

    def test_server_error_is_retried(self):
        session = FakeSession()
        session.post_queue.append(FakeResponse(503, None, text="unavailable"))
        session.post_queue.append(FakeResponse(200, {"message": {"content": "ok"}}))
        client = make_client(session, max_retries=1)
        self.assertEqual(client.chat("问题", []), "ok")

    def test_model_not_found(self):
        session = FakeSession()
        session.post_queue.append(FakeResponse(404, {"error": "model 'x' not found"}))
        client = make_client(session)
        with self.assertRaises(LLMModelNotFoundError):
            client.chat("问题", [])

    def test_client_error_not_retried(self):
        session = FakeSession()
        session.post_queue.append(FakeResponse(400, {"error": "bad request"}))
        client = make_client(session, max_retries=3)
        with self.assertRaises(LLMResponseError):
            client.chat("问题", [])
        self.assertEqual(len(session.post_calls), 1)


class AvailabilityTests(unittest.TestCase):
    def test_list_models(self):
        session = FakeSession()
        session.get_queue.append(
            FakeResponse(200, {"models": [{"name": "llama2"}, {"name": "qwen2:7b"}]})
        )
        client = make_client(session)
        self.assertEqual(client.list_models(), ["llama2", "qwen2:7b"])

    def test_is_available_false_on_failure(self):
        session = FakeSession()
        session.get_queue.append(requests.ConnectionError("refused"))
        client = make_client(session)
        self.assertFalse(client.is_available())

    def test_is_available_true(self):
        session = FakeSession()
        session.get_queue.append(FakeResponse(200, {"models": []}))
        client = make_client(session)
        self.assertTrue(client.is_available())


class DirtyLineTests(unittest.TestCase):
    """脏行不能静默吞掉，否则回答「缺字」且无从排查。"""

    def test_dirty_line_emits_warning(self):
        lines = [b"not-json", b'{"message": {"content": "ok"}, "done": true}']
        with self.assertLogs("career_qa.llm_client", level="WARNING") as captured:
            result = list(iter_ollama_chunks(lines))
        self.assertEqual(result, ["ok"])
        self.assertTrue(any("跳过无法解析" in msg for msg in captured.output))


class ResourceReleaseTests(unittest.TestCase):
    """回归测试：响应必须关闭，否则 stream=True 会持续泄漏连接。"""

    def test_chat_closes_response(self):
        session = FakeSession()
        response = FakeResponse(200, {"message": {"content": "ok"}})
        session.post_queue.append(response)
        make_client(session).chat("问题", [])
        self.assertTrue(response.closed)

    def test_stream_chat_closes_response(self):
        session = FakeSession()
        response = FakeResponse(
            200,
            lines=ndjson(
                {"message": {"content": "A"}, "done": False},
                {"message": {"content": "B"}, "done": True},
            ),
        )
        session.post_queue.append(response)
        self.assertEqual(list(make_client(session).stream_chat("问题", [])), ["A", "B"])
        self.assertTrue(response.closed)

    def test_stream_chat_closes_on_early_abort(self):
        session = FakeSession()
        response = FakeResponse(
            200,
            lines=ndjson(
                {"message": {"content": "A"}, "done": False},
                {"message": {"content": "B"}, "done": False},
            ),
        )
        session.post_queue.append(response)
        stream = make_client(session).stream_chat("问题", [])
        self.assertEqual(next(stream), "A")
        stream.close()  # 消费方提前放弃，例如用户关闭窗口
        self.assertTrue(response.closed)

    def test_stream_chat_closes_on_parse_error(self):
        session = FakeSession()
        response = FakeResponse(200, lines=ndjson({"error": "boom"}))
        session.post_queue.append(response)
        with self.assertRaises(LLMResponseError):
            list(make_client(session).stream_chat("问题", []))
        self.assertTrue(response.closed)

    def test_discarded_5xx_response_is_closed(self):
        session = FakeSession()
        dropped = FakeResponse(503, None, text="unavailable")
        session.post_queue.append(dropped)
        session.post_queue.append(FakeResponse(200, {"message": {"content": "ok"}}))
        self.assertEqual(make_client(session, max_retries=1).chat("问题", []), "ok")
        self.assertTrue(dropped.closed)

    def test_error_status_response_is_closed(self):
        session = FakeSession()
        response = FakeResponse(404, {"error": "model not found"})
        session.post_queue.append(response)
        with self.assertRaises(LLMModelNotFoundError):
            make_client(session).chat("问题", [])
        self.assertTrue(response.closed)

    def test_list_models_closes_response(self):
        session = FakeSession()
        response = FakeResponse(200, {"models": [{"name": "llama2"}]})
        session.get_queue.append(response)
        self.assertEqual(make_client(session).list_models(), ["llama2"])
        self.assertTrue(response.closed)


class ListModelsRetryTests(unittest.TestCase):
    def test_retries_then_succeeds(self):
        session = FakeSession()
        session.get_queue.append(requests.ConnectionError("refused"))
        session.get_queue.append(FakeResponse(200, {"models": [{"name": "llama2"}]}))
        client = make_client(session, max_retries=1)
        self.assertEqual(client.list_models(), ["llama2"])
        self.assertEqual(len(session.get_calls), 2)

    def test_is_available_uses_no_retry(self):
        session = FakeSession()
        session.get_queue.append(requests.ConnectionError("refused"))
        client = make_client(session, max_retries=3)
        self.assertFalse(client.is_available())
        # 快速探测只尝试一次，不应卡在退避等待上
        self.assertEqual(len(session.get_calls), 1)


if __name__ == "__main__":
    unittest.main()
