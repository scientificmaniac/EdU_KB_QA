"""对话编排：把知识库检索、提示词构造和多轮历史串起来。

原实现把这三件事揉在 ``DialogueManager.process_question`` 里，且
历史裁剪依赖魔法数字 ``6``、提示词硬编码。这里做了拆分：

* :meth:`DialogueManager.build_prompt` 只负责检索 + 提示词构造，可单独测试；
* :meth:`DialogueManager.ask` / :meth:`stream_ask` 负责调用模型与记录历史；
* 历史轮数、检索条数、提示词模板全部来自 :class:`~career_qa.config.AppConfig`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from .config import AppConfig
from .knowledge_base import KnowledgeBase, SearchHit
from .llm_client import OllamaClient

logger = logging.getLogger(__name__)

#: 没有命中参考资料时注入的提示，避免模型误以为知识库为空是错误
_NO_CONTEXT_HINT = "（知识库中没有匹配的参考资料，请基于常识谨慎回答。）"
#: 已限定标签但没命中时的提示模板
_NO_CONTEXT_IN_TAGS = (
    "（已在标签「{tags}」范围内检索，但没有找到与问题匹配的参考资料，"
    "请基于常识谨慎回答。）"
)


@dataclass(frozen=True)
class Answer:
    """一次完整问答的结果。"""

    question: str
    text: str
    hits: tuple[SearchHit, ...] = ()

    @property
    def has_reference(self) -> bool:
        return bool(self.hits)


@dataclass
class DialogueManager:
    """串联知识库与大模型的对话管理器。"""

    kb: KnowledgeBase
    llm: OllamaClient
    config: AppConfig = field(default_factory=AppConfig)
    _history: list[dict[str, str]] = field(default_factory=list, repr=False)
    #: 最近一次检索命中的参考资料（流式输出时供界面展示）
    last_hits: tuple[SearchHit, ...] = ()

    # ------------------------------------------------------------------
    # 历史管理
    # ------------------------------------------------------------------
    @property
    def history(self) -> tuple[dict[str, str], ...]:
        """当前保留的对话历史（不含系统提示词）。"""
        return tuple(self._history)

    @property
    def history_turns(self) -> int:
        return len(self._history) // 2

    def reset(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    def _remember(self, question: str, answer: str) -> None:
        """记录一轮问答，并按配置裁剪历史。"""
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})

        limit = max(self.config.max_history_turns, 0) * 2
        if limit == 0:
            self._history.clear()
        elif len(self._history) > limit:
            # 按「轮」裁剪，避免把一问一答拆散
            del self._history[: len(self._history) - limit]

    def _messages(self) -> list[dict[str, str]]:
        """构造发送给模型的消息列表（含系统提示词）。

        返回的是**副本**：调用方修改结果不会污染对话历史。
        """
        messages: list[dict[str, str]] = []
        system_prompt = (self.config.system_prompt or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(dict(item) for item in self._history)
        return messages

    # ------------------------------------------------------------------
    # 提示词构造
    # ------------------------------------------------------------------
    def build_prompt(
        self, question: str, tags: Sequence[str] | None = None
    ) -> tuple[str, tuple[SearchHit, ...]]:
        """检索知识库并渲染提示词，返回 ``(提示词, 命中的参考资料)``。"""
        hits = tuple(
            self.kb.search(
                question,
                tags=tags,
                top_n=self.config.top_n,
                min_score=self.config.min_score,
            )
        )

        if hits:
            lines = []
            for index, hit in enumerate(hits, start=1):
                suffix = f"（标签：{'、'.join(hit.tags)}）" if hit.tags else ""
                lines.append(f"[{index}] {hit.answer}{suffix}")
            context = "参考资料：\n" + "\n".join(lines)
        elif tags:
            # 标签是检索的硬过滤条件，未命中时要说清原因，
            # 否则用户会以为知识库里根本没有相关内容
            context = _NO_CONTEXT_IN_TAGS.format(tags="、".join(tags))
        else:
            context = _NO_CONTEXT_HINT

        try:
            prompt = self.config.prompt_template.format(
                question=question,
                context=context,
                tags="、".join(tags or ()),
            )
        except (KeyError, IndexError, ValueError) as exc:
            # 自定义模板里出现了非法占位符时降级，保证仍能提问
            logger.warning("提示词模板渲染失败（%s），已使用兜底模板", exc)
            prompt = (
                f"这是一个职业生涯规划相关的问题：{question}\n"
                f"{context}\n"
                "请用简洁语言回答，并提供职业生涯规划方面的相关建议。"
            )

        return prompt, hits

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------
    def ask(self, question: str, tags: Sequence[str] | None = None) -> Answer:
        """阻塞式提问，返回完整回答。"""
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        prompt, hits = self.build_prompt(question, tags)
        self.last_hits = hits
        text = self.llm.chat(prompt, self._messages())
        self._remember(question, text)
        return Answer(question=question, text=text, hits=hits)

    def stream_ask(
        self, question: str, tags: Sequence[str] | None = None
    ) -> Iterator[str]:
        """流式提问，逐段产出回答文本。

        只有在流正常结束后才会把这一轮写入历史；中途异常不会污染上下文。
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        prompt, hits = self.build_prompt(question, tags)
        self.last_hits = hits

        pieces: list[str] = []
        for piece in self.llm.stream_chat(prompt, self._messages()):
            pieces.append(piece)
            yield piece

        self._remember(question, "".join(pieces))
