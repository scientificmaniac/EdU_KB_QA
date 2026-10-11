"""tkinter 图形界面。

在原界面布局基础上修复以下问题：

* **线程安全**：原代码在后台线程里直接读取 ``self.selected_tags`` 等
  Tk 变量。现在发起请求前先做快照，工作线程只操作纯 Python 数据，
  所有界面更新都通过 :class:`queue.Queue` 回到主线程执行。
* **并发请求**：请求进行中禁用「发送」按钮，避免连点造成多路请求
  同时改写同一个对话历史。
* **流式输出**：接上修好的 NDJSON 流式解析，回答逐字出现，
  首字延迟明显降低。
* **错误提示**：区分「Ollama 未启动」「模型不存在」等情形，
  给出可操作的提示而不是裸异常文本。
"""

from __future__ import annotations

import logging
import queue
import sys
import tkinter as tk
from pathlib import Path
from threading import Thread
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .config import AppConfig
from .dialogue import DialogueManager
from .knowledge_base import DuplicateEntryError, KnowledgeBase, KnowledgeBaseError
from .llm_client import LLMConnectionError, LLMError, LLMModelNotFoundError, OllamaClient

logger = logging.getLogger(__name__)

#: 主线程轮询后台消息的间隔（毫秒）
_PUMP_INTERVAL_MS = 40


def default_font_family() -> str:
    """按平台挑选一个对中文显示友好的字体族。"""
    if sys.platform == "win32":
        return "Microsoft YaHei UI"
    if sys.platform == "darwin":
        return "PingFang SC"
    return "Noto Sans CJK SC"


class ChatGUI:
    """职业生涯规划助手主窗口。"""

    def __init__(self, root: tk.Tk, config: AppConfig | None = None) -> None:
        self.root = root
        self.config = config or AppConfig()
        self.kb_path = Path(self.config.kb_path)
        self.manager = DialogueManager(
            kb=KnowledgeBase(self.kb_path),
            llm=OllamaClient(
                base_url=self.config.ollama_base_url,
                model=self.config.model,
                connect_timeout=self.config.connect_timeout,
                request_timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
                cache_size=self.config.cache_size,
                temperature=self.config.temperature,
            ),
            config=self.config,
        )

        # 标签筛选状态
        self.selected_tags: list[str] = []
        # 当前表格中可见的条目（用于双击定位，避免下标错位）
        self._visible_entries = list(self.manager.kb.entries)

        # 后台线程 -> 主线程的消息队列
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._busy = False
        self._stream_open = False
        self._pump_job: str | None = None

        self.font_family = default_font_family()
        self.root.title(f"职业生涯规划助手 — {self.config.model}")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_main_frame()
        self._update_tags_display()
        self._update_tag_combobox()
        self._refresh_knowledge_display()
        self._report_load_warnings()
        self._pump()

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _font(self, size: int = 12) -> tuple[str, int]:
        return (self.font_family, size)

    def _create_main_frame(self) -> None:
        style = ttk.Style()
        # LabelFrame 标题使用 TLabelframe.Label 样式名
        style.configure("TLabelframe.Label", font=self._font(14))
        style.configure("TButton", font=self._font(12))
        style.configure("Treeview", font=self._font(12), rowheight=24)
        style.configure("Treeview.Heading", font=self._font(12))
        style.configure("Status.TLabel", font=self._font(10), foreground="#555555")

        # ---------------- 顶部：标签筛选 ----------------
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=20, pady=(16, 8))

        ttk.Label(top_frame, text="标签筛选:", font=self._font(12)).pack(
            side=tk.LEFT, padx=(0, 10)
        )

        self.tag_combobox = ttk.Combobox(
            top_frame, state="readonly", width=20, font=self._font(12)
        )
        self.tag_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.tag_combobox.bind("<<ComboboxSelected>>", self._on_tag_selected)

        self.selected_tags_frame = ttk.Frame(top_frame)
        self.selected_tags_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ---------------- 中部：聊天 + 知识库 ----------------
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)
        middle_frame.columnconfigure(0, weight=3, uniform="cols")
        middle_frame.columnconfigure(1, weight=2, uniform="cols")
        middle_frame.rowconfigure(0, weight=1)

        chat_frame = ttk.LabelFrame(middle_frame, text="智能对话")
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.chat_history = scrolledtext.ScrolledText(
            chat_frame, width=60, height=20, wrap=tk.WORD, font=self._font(12)
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_history.tag_config("user", foreground="#007BFF")
        self.chat_history.tag_config("ai", foreground="#28A745")
        self.chat_history.tag_config("ref", foreground="#888888")
        self.chat_history.configure(state=tk.DISABLED)

        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.input_entry = ttk.Entry(input_frame, width=70, font=self._font(12))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", self._on_return_key)

        self.send_button = ttk.Button(
            input_frame, text="发送", command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        kb_frame = ttk.LabelFrame(middle_frame, text="知识库")
        kb_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        tree_wrap = ttk.Frame(kb_frame)
        tree_wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.knowledge_tree = ttk.Treeview(
            tree_wrap, columns=("问题", "标签"), show="headings"
        )
        self.knowledge_tree.heading("问题", text="问题", anchor=tk.W)
        self.knowledge_tree.heading("标签", text="标签", anchor=tk.W)
        self.knowledge_tree.column("问题", width=280, anchor=tk.W)
        self.knowledge_tree.column("标签", width=130, anchor=tk.W)

        scrollbar = ttk.Scrollbar(
            tree_wrap, orient=tk.VERTICAL, command=self.knowledge_tree.yview
        )
        self.knowledge_tree.configure(yscrollcommand=scrollbar.set)
        self.knowledge_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.knowledge_tree.bind("<Double-1>", self._on_knowledge_selected)

        # ---------------- 底部：操作按钮 + 状态栏 ----------------
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=20, pady=(8, 16))

        ttk.Button(
            bottom_frame, text="导入知识库", command=self.import_kb
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            bottom_frame, text="添加知识", command=self.add_knowledge
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            bottom_frame, text="导出知识库", command=self.export_kb
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            bottom_frame, text="刷新", command=self.refresh
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            bottom_frame, text="清空对话", command=self.clear_conversation
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.status_var = tk.StringVar()
        ttk.Label(
            bottom_frame, textvariable=self.status_var, style="Status.TLabel"
        ).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # 状态栏与提示
    # ------------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _status_text(self, shown: int | None = None) -> str:
        """生成状态栏文本。

        回答结束后 :meth:`_set_busy` 也会调用它，所以标签筛选的上下文
        不会被「模型 X · 知识库 Y · N 条」这样的文本覆盖丢失。
        """
        text = (
            f"模型 {self.config.model} · 知识库 {self.kb_path.name} · "
            f"共 {len(self.manager.kb)} 条"
        )
        if shown is not None:
            text += f"，显示 {shown} 条"
        if self.selected_tags:
            text += f"（已筛选：{'、'.join(self.selected_tags)}）"
        return text

    def _report_load_warnings(self) -> None:
        warnings = list(self.manager.kb.warnings)
        if not warnings:
            return
        self._set_status("知识库加载有告警")
        messagebox.showwarning("知识库告警", "\n".join(warnings))

    def _show_error(self, message: str) -> None:
        self._set_status("出错")
        messagebox.showerror("错误", message)

    # ------------------------------------------------------------------
    # 主线程消息泵
    # ------------------------------------------------------------------
    def _pump(self) -> None:
        """在主线程中消费后台线程投递的消息。"""
        try:
            while True:
                try:
                    kind, payload = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle_message(kind, payload)
                except tk.TclError:
                    # 单条消息渲染失败（窗口可能已被销毁）。这里不能直接
                    # return，否则消息泵会静默停摆，界面永久停在「正在生成回答…」
                    if not self._window_alive():
                        return
                    logger.exception("渲染界面消息失败：kind=%s", kind)
        finally:
            self._schedule_pump()

    def _handle_message(self, kind: str, payload: str) -> None:
        if kind == "chunk":
            self._append_ai_chunk(payload)
        elif kind == "done":
            self._finish_stream()
        elif kind == "error":
            self._finish_stream()
            self._show_error(payload)

    def _window_alive(self) -> bool:
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False

    def _schedule_pump(self) -> None:
        if not self._window_alive():
            self._pump_job = None
            return
        try:
            self._pump_job = self.root.after(_PUMP_INTERVAL_MS, self._pump)
        except tk.TclError:
            self._pump_job = None

    def _on_close(self) -> None:
        if self._pump_job is not None:
            try:
                self.root.after_cancel(self._pump_job)
            except tk.TclError:
                pass
            self._pump_job = None
        self.root.destroy()

    # ------------------------------------------------------------------
    # 标签筛选
    # ------------------------------------------------------------------
    def _update_tag_combobox(self) -> None:
        tags = self.manager.kb.all_tags()
        available = [tag for tag in tags if tag not in self.selected_tags]
        self.tag_combobox["values"] = available
        if available:
            self.tag_combobox.current(0)
        else:
            self.tag_combobox.set("")

    def _on_tag_selected(self, _event: object = None) -> None:
        tag = self.tag_combobox.get().strip()
        if tag and tag not in self.selected_tags:
            self.selected_tags.append(tag)
            self._update_tags_display()
            self._update_tag_combobox()
            self._refresh_knowledge_display()

    def _update_tags_display(self) -> None:
        for widget in self.selected_tags_frame.winfo_children():
            widget.destroy()

        for tag in self.selected_tags:
            tag_frame = ttk.Frame(self.selected_tags_frame)
            tag_frame.pack(side=tk.LEFT, padx=(0, 5))
            ttk.Label(
                tag_frame,
                text=tag,
                background="#e2e8f0",
                padding=(5, 2),
                borderwidth=1,
                relief="solid",
                font=self._font(10),
            ).pack(side=tk.LEFT)
            ttk.Button(
                tag_frame,
                text="×",
                width=2,
                command=lambda t=tag: self._remove_tag(t),
            ).pack(side=tk.RIGHT)

        if self.selected_tags:
            ttk.Button(
                self.selected_tags_frame,
                text="清除所有",
                command=self._clear_all_tags,
            ).pack(side=tk.LEFT, padx=(5, 0))

    def _remove_tag(self, tag: str) -> None:
        if tag in self.selected_tags:
            self.selected_tags.remove(tag)
            self._update_tags_display()
            self._update_tag_combobox()
            self._refresh_knowledge_display()

    def _clear_all_tags(self) -> None:
        self.selected_tags = []
        self._update_tags_display()
        self._update_tag_combobox()
        self._refresh_knowledge_display()

    # ------------------------------------------------------------------
    # 知识库展示
    # ------------------------------------------------------------------
    def _refresh_knowledge_display(self) -> None:
        for item in self.knowledge_tree.get_children():
            self.knowledge_tree.delete(item)

        entries = self.manager.kb.filter_by_tags(self.selected_tags)
        self._visible_entries = entries

        for index, entry in enumerate(entries):
            self.knowledge_tree.insert(
                "", tk.END, iid=str(index), values=(entry.question, "、".join(entry.tags))
            )

        self._set_status(self._status_text(shown=len(entries)))

    def refresh(self) -> None:
        """重新从磁盘读取知识库并刷新界面。"""
        try:
            self.manager.kb.reload()
        except KnowledgeBaseError as exc:
            self._show_error(str(exc))
            return
        self._update_tag_combobox()
        self._refresh_knowledge_display()
        if self.manager.kb.warnings:
            messagebox.showwarning("知识库告警", "\n".join(self.manager.kb.warnings))

    def _on_knowledge_selected(self, _event: object = None) -> None:
        selection = self.knowledge_tree.selection()
        if not selection:
            return
        try:
            entry = self._visible_entries[int(selection[0])]
        except (ValueError, IndexError):
            return
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, entry.question)

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------
    def _on_return_key(self, _event: object = None) -> str:
        self.send_message()
        return "break"

    def _append_to_chat(self, text: str, tag: str) -> None:
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.insert(tk.END, text, tag)
        self.chat_history.see(tk.END)
        self.chat_history.configure(state=tk.DISABLED)

    def _append_ai_chunk(self, text: str) -> None:
        self._append_to_chat(text, "ai")

    def _finish_stream(self) -> None:
        if self._stream_open:
            self._append_to_chat("\n\n", "ai")
            self._stream_open = False
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        try:
            self.send_button.configure(state=tk.DISABLED if busy else tk.NORMAL)
            self.input_entry.configure(state=tk.DISABLED if busy else tk.NORMAL)
        except tk.TclError:
            pass
        if not busy:
            # 复用统一的文本生成，避免把「已筛选」等上下文覆盖掉
            self._set_status(self._status_text())

    def clear_conversation(self) -> None:
        """清空界面与模型上下文。"""
        self.manager.reset()
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.delete("1.0", tk.END)
        self.chat_history.configure(state=tk.DISABLED)
        self._set_status("对话已清空")

    def send_message(self) -> None:
        if self._busy:
            return

        question = self.input_entry.get().strip()
        if not question:
            return

        if question.lower() in {"exit", "quit", "退出"}:
            self._on_close()
            return

        # 关键修复：在主线程中快照标签与输入，后台线程不再触碰 Tk 控件
        tags = list(self.selected_tags) or None

        self._append_to_chat(f"你：{question}\n", "user")
        self.input_entry.delete(0, tk.END)
        self._append_to_chat("AI：", "ai")
        self._stream_open = True
        self._set_busy(True)
        self._set_status("正在生成回答…")

        Thread(
            target=self._worker, args=(question, tags), daemon=True
        ).start()

    def _worker(self, question: str, tags: list[str] | None) -> None:
        """后台线程：只产出文本，通过队列交给主线程渲染。"""
        try:
            if self.config.stream:
                for piece in self.manager.stream_ask(question, tags):
                    self._queue.put(("chunk", piece))
                self._queue.put(("done", ""))
            else:
                answer = self.manager.ask(question, tags)
                self._queue.put(("chunk", answer.text))
                self._queue.put(("done", ""))
        except LLMModelNotFoundError as exc:
            self._queue.put(("error", str(exc)))
        except LLMConnectionError as exc:
            self._queue.put(("error", str(exc)))
        except LLMError as exc:
            self._queue.put(("error", f"模型调用失败：{exc}"))
        except Exception as exc:  # noqa: BLE001 - 兜底，避免线程静默死亡
            logger.exception("处理请求时出现未预期的错误")
            self._queue.put(("error", f"处理请求时出现错误：{exc}"))

    # ------------------------------------------------------------------
    # 知识库操作
    # ------------------------------------------------------------------
    def import_kb(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择知识库 JSON 文件",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not file_path:
            return

        kb = KnowledgeBase(Path(file_path))
        if kb.warnings:
            preview = "\n".join(kb.warnings[:5])
            if not messagebox.askyesno(
                "导入告警", f"该文件存在以下问题：\n{preview}\n\n仍要导入吗？"
            ):
                return

        # 明确切换「当前知识库」，避免用户误以为后续编辑仍写回原文件
        self.manager.kb = kb
        self.kb_path = Path(file_path)
        self.selected_tags = []
        self._update_tags_display()
        self._update_tag_combobox()
        self._refresh_knowledge_display()
        messagebox.showinfo(
            "导入成功",
            f"已载入 {len(kb)} 条知识。\n当前知识库：{self.kb_path}\n\n"
            "提示：此后「添加知识」将写入该文件。",
        )

    def export_kb(self) -> None:
        file_path = filedialog.asksaveasfilename(
            title="导出知识库",
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not file_path:
            return
        try:
            self.manager.kb.save(file_path)
        except KnowledgeBaseError as exc:
            self._show_error(str(exc))
            return
        messagebox.showinfo("导出成功", f"已导出到：{file_path}")

    def add_knowledge(self) -> None:
        add_window = tk.Toplevel(self.root)
        add_window.title("添加知识")
        add_window.geometry("560x340")
        add_window.transient(self.root)
        add_window.grab_set()

        ttk.Label(add_window, text="问题:", font=self._font(12)).pack(
            pady=(12, 4), padx=12, anchor=tk.W
        )
        question_entry = ttk.Entry(add_window, width=60, font=self._font(12))
        question_entry.pack(fill=tk.X, padx=12)

        ttk.Label(add_window, text="答案:", font=self._font(12)).pack(
            pady=(10, 4), padx=12, anchor=tk.W
        )
        answer_entry = scrolledtext.ScrolledText(
            add_window, width=60, height=6, font=self._font(12)
        )
        answer_entry.pack(fill=tk.BOTH, expand=True, padx=12)

        ttk.Label(add_window, text="标签（用逗号分隔）:", font=self._font(12)).pack(
            pady=(10, 4), padx=12, anchor=tk.W
        )
        tags_entry = ttk.Entry(add_window, width=60, font=self._font(12))
        tags_entry.pack(fill=tk.X, padx=12)

        def save_knowledge() -> None:
            question = question_entry.get().strip()
            answer = answer_entry.get("1.0", tk.END).strip()
            tags_text = tags_entry.get().strip()

            if not question or not answer:
                messagebox.showerror("错误", "问题和答案不能为空", parent=add_window)
                return

            try:
                self.manager.kb.add_entry(question, answer, tags_text)
            except DuplicateEntryError as exc:
                messagebox.showwarning("重复条目", str(exc), parent=add_window)
                return
            except KnowledgeBaseError as exc:
                messagebox.showerror("保存失败", str(exc), parent=add_window)
                return

            self._update_tag_combobox()
            self._refresh_knowledge_display()
            messagebox.showinfo("成功", "知识添加成功", parent=add_window)
            add_window.destroy()

        ttk.Button(add_window, text="保存", command=save_knowledge).pack(pady=12)
        question_entry.focus_set()


def run(config: AppConfig | None = None) -> int:
    """创建主窗口并进入事件循环。"""
    root = tk.Tk()
    ChatGUI(root, config)
    root.mainloop()
    return 0
