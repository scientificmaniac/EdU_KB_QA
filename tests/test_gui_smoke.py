"""GUI 冒烟测试。

覆盖单元测试无法触及的部分：后台线程 → 队列 → 主线程渲染的完整链路。

在没有可用显示环境（如 CI 容器）时会自动跳过。
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import tkinter as tk

    from career_qa.gui import ChatGUI
except ImportError:  # pragma: no cover - 无 tkinter 的环境（如精简版 Linux）
    tk = None  # type: ignore[assignment]
    ChatGUI = None  # type: ignore[assignment]

from career_qa.config import AppConfig
from career_qa.llm_client import LLMConnectionError, LLMError

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base.json"


class FakeStreamLLM:
    """假的流式客户端。"""

    def __init__(self, chunks=("建议：", "先明确优势。")):
        self.chunks = list(chunks)
        self.calls = []

    def chat(self, prompt, history=None):
        self.calls.append((prompt, list(history or [])))
        return "".join(self.chunks)

    def stream_chat(self, prompt, history=None):
        self.calls.append((prompt, list(history or [])))
        for piece in self.chunks:
            yield piece


class FakeErrorLLM:
    def chat(self, prompt, history=None):
        raise LLMConnectionError("Ollama 未启动")

    def stream_chat(self, prompt, history=None):
        raise LLMConnectionError("Ollama 未启动")
        yield  # pragma: no cover - 使其成为生成器


@unittest.skipIf(tk is None, "tkinter 不可用")
class GuiSmokeTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:  # pragma: no cover
            self.skipTest(f"无可用显示环境：{exc}")
        self.root.withdraw()
        self.gui = None

    def tearDown(self):
        if self.gui is not None:
            try:
                self.gui._on_close()
            except tk.TclError:
                pass
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    def make_gui(self, **config_kwargs) -> ChatGUI:
        defaults = {"kb_path": KB_PATH, "min_score": 0.1, "top_n": 2}
        defaults.update(config_kwargs)
        with mock.patch("career_qa.gui.messagebox"):
            gui = ChatGUI(self.root, AppConfig(**defaults))
        self.gui = gui
        return gui

    def wait_idle(self, gui: ChatGUI, timeout: float = 10.0) -> None:
        """驱动 Tk 事件循环直到后台请求完成。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.root.update()
            if not gui._busy:
                break
            time.sleep(0.01)
        else:
            self.fail("GUI 未在超时内完成请求")
        self.root.update()

    def chat_text(self) -> str:
        return self.gui.chat_history.get("1.0", tk.END)

    # ------------------------------------------------------------------
    def test_window_builds(self):
        gui = self.make_gui()
        self.assertEqual(len(gui.knowledge_tree.get_children()), 2)
        self.assertIn("knowledge_base.json", gui.status_var.get())

    def test_streaming_message_renders(self):
        gui = self.make_gui(stream=True)
        gui.manager.llm = FakeStreamLLM()

        gui.input_entry.insert(0, "如何进行职业规划?")
        gui.send_message()
        self.wait_idle(gui)

        text = self.chat_text()
        self.assertIn("你：如何进行职业规划?", text)
        self.assertIn("建议：先明确优势。", text)
        self.assertEqual(gui.input_entry.get(), "")

    def test_non_streaming_message_renders(self):
        gui = self.make_gui(stream=False)
        gui.manager.llm = FakeStreamLLM(chunks=["完整回答"])

        gui.input_entry.insert(0, "数据科学家的职业前景如何?")
        gui.send_message()
        self.wait_idle(gui)

        self.assertIn("完整回答", self.chat_text())

    def test_history_survives_across_messages(self):
        """回归测试：第二轮对话在旧实现中会因缓存键不可哈希而崩溃。"""
        gui = self.make_gui(stream=True)
        llm = FakeStreamLLM()
        gui.manager.llm = llm

        for question in ("如何进行职业规划?", "如何准备面试?"):
            gui.input_entry.insert(0, question)
            gui.send_message()
            self.wait_idle(gui)

        self.assertEqual(len(llm.calls), 2)
        # 第二次请求应带上第一轮的历史
        self.assertGreater(len(llm.calls[1][1]), 0)
        self.assertEqual(self.gui.manager.history_turns, 2)

    def test_error_is_reported_not_crashing(self):
        gui = self.make_gui(stream=True)
        gui.manager.llm = FakeErrorLLM()

        with mock.patch("career_qa.gui.messagebox") as box:
            gui.input_entry.insert(0, "会失败的问题")
            gui.send_message()
            self.wait_idle(gui)

        box.showerror.assert_called_once()
        self.assertIn("Ollama 未启动", box.showerror.call_args[0][1])
        self.assertFalse(gui._busy)

    def test_empty_input_does_nothing(self):
        gui = self.make_gui()
        gui.manager.llm = FakeStreamLLM()
        gui.input_entry.delete(0, tk.END)
        gui.send_message()
        self.assertFalse(gui._busy)
        self.assertEqual(self.chat_text().strip(), "")

    def test_tag_filter_updates_tree(self):
        gui = self.make_gui()
        gui.selected_tags = ["数据科学"]
        gui._refresh_knowledge_display()
        self.assertEqual(len(gui.knowledge_tree.get_children()), 1)

    def test_clear_conversation(self):
        gui = self.make_gui(stream=True)
        gui.manager.llm = FakeStreamLLM(chunks=["回答"])
        gui.input_entry.insert(0, "问题")
        gui.send_message()
        self.wait_idle(gui)

        gui.clear_conversation()
        self.assertEqual(self.chat_text().strip(), "")
        self.assertEqual(gui.manager.history_turns, 0)

    def test_double_click_fills_input(self):
        gui = self.make_gui()
        first = gui.knowledge_tree.get_children()[0]
        gui.knowledge_tree.selection_set(first)
        gui._on_knowledge_selected()
        self.assertEqual(gui.input_entry.get(), gui._visible_entries[0].question)

    def test_busy_state_disables_send_button(self):
        gui = self.make_gui(stream=True)
        gui.manager.llm = FakeStreamLLM()
        gui.input_entry.insert(0, "问题")
        gui.send_message()
        self.assertTrue(gui._busy)
        self.assertTrue(gui.send_button.instate(["disabled"]))
        self.wait_idle(gui)
        self.assertFalse(gui.send_button.instate(["disabled"]))

    def test_model_name_in_title(self):
        gui = self.make_gui(model="qwen2")
        self.assertIn("qwen2", gui.root.title())

    def test_status_keeps_tag_filter_after_answer(self):
        """回归测试：回答结束后状态栏不应丢失标签筛选上下文。"""
        gui = self.make_gui(stream=True)
        gui.manager.llm = FakeStreamLLM()

        gui.selected_tags = ["数据科学"]
        gui._refresh_knowledge_display()
        self.assertIn("已筛选", gui.status_var.get())

        gui.input_entry.insert(0, "问题")
        gui.send_message()
        self.wait_idle(gui)

        self.assertIn("已筛选", gui.status_var.get())


if __name__ == "__main__":
    unittest.main()
