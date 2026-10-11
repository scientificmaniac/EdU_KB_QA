"""Ollama HTTP 客户端。

相比原实现修复了以下问题：

1. **缓存键不可哈希**：原代码用 ``(prompt, tuple(history))`` 作缓存键，
   而 ``history`` 是字典列表，``tuple(history)`` 仍含 dict，
   第二次对话必然抛 ``TypeError: unhashable type: 'dict'``。
   现在改为 JSON 稳定序列化后再做键。
2. **流式解析格式错误**：Ollama ``/api/chat`` 返回的是 **NDJSON**
   （每行一个完整 JSON），并不存在 ``data: `` 前缀，原代码因此永远
   解析不出内容。现在按行 ``json.loads``，同时兼容网关可能加上的
   SSE 前缀。
3. **无超时、无重试、无异常分类**：网络故障直接抛出原始
   ``requests`` 异常。现在统一包装为 :class:`LLMError` 子类。
4. **可变默认参数** ``history=[]`` 与无上限缓存均已修正。
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Iterable, Iterator, Sequence

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "LLMError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMModelNotFoundError",
    "OllamaClient",
    "iter_ollama_chunks",
]

#: 默认系统提示词在 dialogue 层注入，这里只负责传输
Message = dict[str, str]

#: 探测模型列表时的读取超时（这类请求应当快速返回）
_TAGS_READ_TIMEOUT = 10.0


class LLMError(Exception):
    """大模型调用相关错误的基类。"""


class LLMConnectionError(LLMError):
    """无法连接 Ollama 服务（未启动、地址错误、超时）。"""


class LLMResponseError(LLMError):
    """服务返回了错误响应或无法解析的内容。"""


class LLMModelNotFoundError(LLMResponseError):
    """请求的模型未在 Ollama 中安装。"""


def _decode(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _close(response: Any) -> None:
    """尽力关闭响应，把底层连接归还连接池。

    ``requests`` 在 ``stream=True`` 时必须显式 ``close()``（或读尽 body）
    才会释放连接；异常路径与重试时被丢弃的响应同样需要关闭。
    """
    closer = getattr(response, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:  # noqa: BLE001 - 关闭失败不应影响主流程
        logger.debug("关闭响应时出现异常", exc_info=True)


def iter_ollama_chunks(lines: Iterable[Any]) -> Iterator[str]:
    """把 Ollama 流式响应逐行解析为文本片段。

    :param lines: 可迭代对象，元素为 ``bytes`` 或 ``str``（通常是
        ``response.iter_lines()`` 的结果）
    :raises LLMResponseError: 服务端在流中返回了 ``error`` 字段

    解析规则：

    * 空行跳过；
    * 兼容可选的 ``data:`` 前缀（Ollama 原生为纯 NDJSON，不带前缀）；
    * 非法 JSON 行静默跳过，避免个别脏行中断整个回答；
    * 读到 ``done: true`` 立即结束。
    """
    for raw in lines:
        text = _decode(raw).strip()
        if not text:
            continue
        if text.startswith("data:"):
            text = text[5:].strip()
            if not text:
                continue
        if text == "[DONE]":
            return

        try:
            chunk = json.loads(text)
        except json.JSONDecodeError:
            # 个别脏行不该中断整个回答，但也不能无声吞掉：那会让回答「缺字」
            # 且无法区分是模型本身如此还是解析器丢内容
            logger.warning("跳过无法解析的流式响应行：%r", text[:120])
            continue
        if not isinstance(chunk, dict):
            continue

        error = chunk.get("error")
        if error:
            raise LLMResponseError(f"Ollama 返回错误：{error}")

        message = chunk.get("message")
        piece = ""
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                piece = content
        if piece:
            yield piece

        if chunk.get("done"):
            return


class OllamaClient:
    """与本地 Ollama 服务交互的轻量客户端。

    :param base_url: 服务地址，例如 ``http://localhost:11434``
    :param model: 模型名称
    :param connect_timeout: 建连超时（秒）
    :param request_timeout: 读取超时（秒）
    :param max_retries: 网络类失败的重试次数（不含首次）
    :param cache_size: 非流式响应缓存条数，``0`` 表示禁用
    :param temperature: 采样温度
    :param session: 依赖注入用的 ``requests.Session`` 替身
    :param sleep: 依赖注入用的退避函数
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        connect_timeout: float = 5.0,
        request_timeout: float = 120.0,
        max_retries: int = 2,
        cache_size: int = 128,
        temperature: float = 0.7,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.max_retries = max(0, int(max_retries))
        self.cache_size = max(0, int(cache_size))
        self.temperature = temperature
        self.session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._cache: OrderedDict[str, str] = OrderedDict()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @property
    def timeout(self) -> tuple[float, float]:
        return (self.connect_timeout, self.request_timeout)

    def _cache_key(self, prompt: str, history: Sequence[Message] | None) -> str:
        """构造稳定、可哈希的缓存键（JSON 序列化替代原来的 tuple(dict)）。"""
        return json.dumps(
            [self.model, self.temperature, prompt, list(history or [])],
            ensure_ascii=False,
            sort_keys=True,
        )

    def _cache_get(self, key: str) -> str | None:
        if not self.cache_size or key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value  # 命中后移到队尾（LRU）
        return value

    def _cache_put(self, key: str, value: str) -> None:
        if not self.cache_size:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """清空响应缓存。"""
        self._cache.clear()

    @property
    def cache_info(self) -> dict[str, int]:
        return {"size": len(self._cache), "maxsize": self.cache_size}

    def _build_payload(
        self, prompt: str, history: Sequence[Message] | None, stream: bool
    ) -> dict[str, Any]:
        messages: list[Message] = [dict(item) for item in (history or [])]
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self.temperature},
        }

    def _post(self, path: str, payload: dict[str, Any], stream: bool):
        """带超时与指数退避重试的 POST。"""
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(min(0.5 * (2**attempt), 4.0))
                    continue
                raise LLMConnectionError(
                    f"无法连接 Ollama 服务（{url}）：{exc}。"
                    "请确认 Ollama 已启动，且地址配置正确。"
                ) from exc
            except requests.RequestException as exc:  # 其他 requests 异常
                raise LLMError(f"请求 Ollama 失败：{exc}") from exc

            status = getattr(response, "status_code", 200)
            if status >= 500 and attempt < self.max_retries:
                last_error = LLMResponseError(f"服务端返回 {status}")
                # 被重试丢弃的响应也要归还连接，否则会慢慢耗尽连接池
                _close(response)
                self._sleep(min(0.5 * (2**attempt), 4.0))
                continue

            try:
                self._raise_for_status(response)
            except LLMError:
                # 抛错时调用方拿不到这个 response，只能在这里关闭
                _close(response)
                raise
            return response

        # 理论不可达，仅作兜底
        raise LLMConnectionError(f"请求 Ollama 失败：{last_error}")

    def _raise_for_status(self, response: Any) -> None:
        status = getattr(response, "status_code", 200)
        if status < 400:
            return

        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("error"):
                detail = str(body["error"])
        except Exception:  # noqa: BLE001 - 错误体解析失败不影响主流程
            detail = ""
        if not detail:
            detail = (getattr(response, "text", "") or "")[:200]

        if status == 404:
            raise LLMModelNotFoundError(
                f"模型 {self.model!r} 不存在：{detail}。"
                f"可用 `ollama pull {self.model}` 安装。"
            )
        raise LLMResponseError(f"Ollama 返回 HTTP {status}：{detail}")

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def chat(self, prompt: str, history: Sequence[Message] | None = None) -> str:
        """非流式请求，返回完整回答文本（带 LRU 缓存）。"""
        key = self._cache_key(prompt, history)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        payload = self._build_payload(prompt, history, stream=False)
        response = self._post("/api/chat", payload, stream=False)

        try:
            try:
                data = response.json()
            except ValueError as exc:
                raise LLMResponseError(f"响应不是合法 JSON：{exc}") from exc
            if not isinstance(data, dict):
                raise LLMResponseError("响应 JSON 顶层不是对象")

            if data.get("error"):
                raise LLMResponseError(f"Ollama 返回错误：{data['error']}")

            message = data.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise LLMResponseError("响应中缺少 message.content 字段")
        finally:
            _close(response)

        self._cache_put(key, content)
        return content

    def stream_chat(
        self, prompt: str, history: Sequence[Message] | None = None
    ) -> Iterator[str]:
        """流式请求，逐段产出回答文本。

        无论正常读完、消费方中途放弃（触发 ``GeneratorExit``）
        还是解析报错，都会关闭响应，把连接归还连接池。
        """
        payload = self._build_payload(prompt, history, stream=True)
        response = self._post("/api/chat", payload, stream=True)
        try:
            yield from iter_ollama_chunks(response.iter_lines())
        finally:
            _close(response)

    def generate_response(
        self,
        prompt: str,
        history: Sequence[Message] | None = None,
        stream: bool = False,
    ) -> str:
        """兼容旧版调用方式的薄封装。

        ``stream=True`` 时逐段拼接流式结果后整体返回。
        """
        if not stream:
            return self.chat(prompt, history)
        return "".join(self.stream_chat(prompt, history))

    # ------------------------------------------------------------------
    # 可用性探测
    # ------------------------------------------------------------------
    def list_models(self, max_retries: int | None = None) -> list[str]:
        """列出服务端已安装的模型名称。

        :param max_retries: 覆盖实例默认的重试次数；可用 ``0`` 做快速探测
        """
        url = f"{self.base_url}/api/tags"
        retries = self.max_retries if max_retries is None else max(0, int(max_retries))
        last_error: Exception | None = None
        response: Any = None

        for attempt in range(retries + 1):
            try:
                response = self.session.get(
                    url, timeout=(self.connect_timeout, _TAGS_READ_TIMEOUT)
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt < retries:
                    self._sleep(min(0.5 * (2**attempt), 4.0))
                    continue
                raise LLMConnectionError(
                    f"无法连接 Ollama 服务（{url}）：{exc}。"
                    "请确认 Ollama 已启动，且地址配置正确。"
                ) from exc
            except requests.RequestException as exc:
                raise LLMError(f"请求 Ollama 失败：{exc}") from exc

            status = getattr(response, "status_code", 200)
            if status >= 500 and attempt < retries:
                last_error = LLMResponseError(f"服务端返回 {status}")
                _close(response)
                response = None
                self._sleep(min(0.5 * (2**attempt), 4.0))
                continue
            break

        if response is None:  # pragma: no cover - 兜底，理论上不可达
            raise LLMConnectionError(f"请求 Ollama 失败：{last_error}")

        try:
            self._raise_for_status(response)
            try:
                data = response.json()
            except ValueError as exc:
                raise LLMResponseError(f"响应不是合法 JSON：{exc}") from exc
        finally:
            _close(response)

        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        names: list[str] = []
        for item in models:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str):
                    names.append(name)
        return names

    def is_available(self) -> bool:
        """服务可达时返回 ``True``，不抛异常。"""
        try:
            # 用 0 重试做快速探测，避免启动自检卡在退避等待上
            self.list_models(max_retries=0)
        except LLMError:
            return False
        return True
