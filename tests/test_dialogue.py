"""对话管理层的单元测试（使用假的 LLM 客户端）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from career_qa.config import AppConfig
from career_qa.dialogue import DialogueManager
from career_qa.knowledge_base import KnowledgeBase
from career_qa.llm_client import LLMError

SAMPLE = [
    {"问题": "如何进行职业规划?", "答案": "明确优势与目标。", "标签": ["职业规划"]},
    {"问题": "如何准备面试?", "答案": "梳理项目经历。", "标签": ["求职"]},
]


class FakeLLM:
    """记录调用的假客户端。"""

    def __init__(self, reply="这是回答", chunks=None):
        self.reply = reply
        self.chunks = list(chunks) if chunks is not None else None
        self.chat_calls: list[tuple[str, list]] = []
        self.stream_calls: list[tuple[str, list]] = []

    def chat(self, prompt, history=None):
        self.chat_calls.append((prompt, list(history or [])))
        return self.reply

    def stream_chat(self, prompt, history=None):
        self.stream_calls.append((prompt, list(history or [])))
        for piece in (self.chunks if self.chunks is not None else [self.reply]):
            yield piece


class BoomLLM:
    def stream_chat(self, prompt, history=None):
        yield "部分内容"
        raise LLMError("连接中断")


class EmptyQuestionLLM:
    def chat(self, prompt, history=None):  # pragma: no cover - 不应被调用
        raise AssertionError("空问题不应调用模型")


class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.kb_path = Path(self.tmp.name) / "kb.json"
        self.kb_path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
        self.kb = KnowledgeBase(self.kb_path)

    def make_manager(self, llm=None, **config_kwargs) -> DialogueManager:
        defaults = {"kb_path": self.kb_path, "top_n": 2, "min_score": 0.1}
        defaults.update(config_kwargs)
        return DialogueManager(
            kb=self.kb, llm=llm or FakeLLM(), config=AppConfig(**defaults)
        )

    # ---------------- 提示词构造 ----------------
    def test_prompt_contains_reference(self):
        manager = self.make_manager()
        prompt, hits = manager.build_prompt("如何进行职业规划?")
        self.assertTrue(hits)
        self.assertIn("参考资料", prompt)
        self.assertIn("明确优势与目标。", prompt)
        self.assertIn("如何进行职业规划?", prompt)

    def test_prompt_includes_tags_hint(self):
        manager = self.make_manager()
        prompt, hits = manager.build_prompt("职业规划")
        self.assertTrue(hits)
        self.assertIn("标签：职业规划", prompt)

    def test_prompt_without_reference(self):
        manager = self.make_manager()
        prompt, hits = manager.build_prompt("今天天气怎么样")
        self.assertEqual(hits, ())
        self.assertIn("没有匹配的参考资料", prompt)

    def test_prompt_respects_top_n(self):
        manager = self.make_manager(top_n=1, min_score=0.0)
        _, hits = manager.build_prompt("如何")
        self.assertLessEqual(len(hits), 1)

    def test_tags_are_forwarded_to_search(self):
        manager = self.make_manager(min_score=0.0, top_n=5)
        _, hits = manager.build_prompt("如何", tags=["求职"])
        self.assertTrue(hits)
        self.assertTrue(all("求职" in hit.tags for hit in hits))

    def test_broken_template_falls_back(self):
        manager = self.make_manager(prompt_template="只有 {不存在} 占位符")
        prompt, _ = manager.build_prompt("问题")
        self.assertIn("职业生涯规划", prompt)

    # ---------------- 非流式问答 ----------------
    def test_ask_returns_answer(self):
        llm = FakeLLM(reply="建议一")
        manager = self.make_manager(llm=llm)
        answer = manager.ask("如何进行职业规划?")
        self.assertEqual(answer.text, "建议一")
        self.assertEqual(answer.question, "如何进行职业规划?")
        self.assertTrue(answer.has_reference)
        self.assertEqual(len(llm.chat_calls), 1)

    def test_ask_records_history(self):
        manager = self.make_manager()
        manager.ask("问题一")
        self.assertEqual(manager.history_turns, 1)
        self.assertEqual(manager.history[0], {"role": "user", "content": "问题一"})
        self.assertEqual(manager.history[1]["role"], "assistant")

    def test_history_sent_to_model(self):
        llm = FakeLLM()
        manager = self.make_manager(llm=llm)
        manager.ask("问题一")
        manager.ask("问题二")
        _, messages = llm.chat_calls[1]
        # 第 0 条是系统提示词，其后为历史，最后是新提问
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual([m["content"] for m in messages[1:]], ["问题一", "这是回答"])
        self.assertEqual(llm.chat_calls[1][0].count("问题二"), 1)

    def test_system_prompt_prepended(self):
        llm = FakeLLM()
        manager = self.make_manager(llm=llm, system_prompt="你是顾问")
        manager.ask("问题")
        _, messages = llm.chat_calls[0]
        self.assertEqual(messages[0], {"role": "system", "content": "你是顾问"})

    def test_empty_system_prompt_skipped(self):
        llm = FakeLLM()
        manager = self.make_manager(llm=llm, system_prompt="   ")
        manager.ask("问题")
        _, messages = llm.chat_calls[0]
        self.assertTrue(all(m["role"] != "system" for m in messages))

    def test_history_truncated_to_configured_turns(self):
        manager = self.make_manager(max_history_turns=2)
        for index in range(5):
            manager.ask(f"问题{index}")
        self.assertEqual(manager.history_turns, 2)
        self.assertEqual(manager.history[0]["content"], "问题3")
        self.assertEqual(manager.history[-2]["content"], "问题4")

    def test_history_disabled(self):
        manager = self.make_manager(max_history_turns=0)
        manager.ask("问题一")
        self.assertEqual(manager.history_turns, 0)

    def test_reset_clears_history(self):
        manager = self.make_manager()
        manager.ask("问题")
        manager.reset()
        self.assertEqual(manager.history_turns, 0)

    def test_empty_question_rejected(self):
        manager = DialogueManager(
            kb=self.kb, llm=EmptyQuestionLLM(), config=AppConfig(kb_path=self.kb_path)
        )
        with self.assertRaises(ValueError):
            manager.ask("   ")

    # ---------------- 流式问答 ----------------
    def test_stream_ask_yields_chunks(self):
        llm = FakeLLM(chunks=["职业", "规划", "建议"])
        manager = self.make_manager(llm=llm)
        pieces = list(manager.stream_ask("如何进行职业规划?"))
        self.assertEqual(pieces, ["职业", "规划", "建议"])

    def test_stream_ask_records_history_once_done(self):
        llm = FakeLLM(chunks=["甲", "乙"])
        manager = self.make_manager(llm=llm)
        list(manager.stream_ask("问题"))
        self.assertEqual(manager.history_turns, 1)
        self.assertEqual(manager.history[1]["content"], "甲乙")

    def test_stream_failure_does_not_pollute_history(self):
        """流式中断时不得把半截回答写进上下文。"""
        manager = DialogueManager(
            kb=self.kb, llm=BoomLLM(), config=AppConfig(kb_path=self.kb_path)
        )
        with self.assertRaises(LLMError):
            list(manager.stream_ask("问题"))
        self.assertEqual(manager.history_turns, 0)

    def test_stream_passes_history(self):
        llm = FakeLLM(chunks=["x"])
        manager = self.make_manager(llm=llm)
        list(manager.stream_ask("第一问"))
        list(manager.stream_ask("第二问"))
        _, messages = llm.stream_calls[1]
        roles = [m["role"] for m in messages]
        self.assertEqual(roles[0], "system")
        self.assertIn({"role": "user", "content": "第一问"}, messages)
        self.assertIn({"role": "assistant", "content": "x"}, messages)

    def test_last_hits_exposed(self):
        manager = self.make_manager()
        list(manager.stream_ask("如何进行职业规划?"))
        self.assertTrue(manager.last_hits)

    # ---------------- 上下文隔离与提示词 ----------------
    def test_messages_is_a_copy(self):
        """回归测试：_messages() 不得返回内部历史的别名。"""
        manager = self.make_manager()
        manager.ask("问题一")

        messages = manager._messages()
        messages[1]["content"] = "被外部改写"
        messages.append({"role": "user", "content": "多出来的"})

        self.assertEqual(manager.history[0]["content"], "问题一")
        self.assertEqual(manager.history_turns, 1)

    def test_prompt_mentions_tags_when_filtered_and_empty(self):
        """限定了标签却没命中时，提示词要说明原因而不是含糊其辞。"""
        manager = self.make_manager(min_score=0.99, top_n=2)
        prompt, hits = manager.build_prompt("完全无关的问题", tags=["求职"])
        self.assertEqual(hits, ())
        self.assertIn("求职", prompt)
        self.assertIn("没有找到", prompt)
        self.assertNotIn("参考资料：", prompt)

    def test_prompt_without_tags_uses_generic_hint(self):
        manager = self.make_manager(min_score=0.99, top_n=2)
        prompt, hits = manager.build_prompt("完全无关的问题")
        self.assertEqual(hits, ())
        self.assertIn("没有匹配的参考资料", prompt)
        self.assertNotIn("参考资料：", prompt)


if __name__ == "__main__":
    unittest.main()
