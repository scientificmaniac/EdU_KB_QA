import pandas as pd
import requests
import json
import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk, messagebox
from threading import Thread
from collections import defaultdict

class KnowledgeBase:
    """支持标签化的知识库"""

    def __init__(self, file_path):
        self.file_path = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = []  # 格式: [{问题, 答案, 标签: [标签1, 标签2]}]

    def search_answers(self, question, tags=None, top_n=3):
        """根据问题和标签搜索答案"""
        # 先根据标签筛选
        if tags:
            filtered_data = [item for item in self.data if any(tag in item.get('标签', []) for tag in tags)]
        else:
            filtered_data = self.data

        # 再根据问题匹配
        matches = [item for item in filtered_data if question in item['问题']]
        return matches[:top_n]

    def get_tags_stats(self):
        """获取标签统计信息"""
        tag_counts = defaultdict(int)
        for item in self.data:
            for tag in item.get('标签', []):
                tag_counts[tag] += 1
        return tag_counts

    def save_to_file(self):
        """保存知识库到文件"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def add_entry(self, question, answer, tags=None):
        """添加带标签的知识条目"""
        entry = {"问题": question, "答案": answer}
        if tags:
            entry["标签"] = tags
        self.data.append(entry)
        self.save_to_file()

    def get_all_tags(self):
        """获取所有标签的集合"""
        tags = set()
        for item in self.data:
            tags.update(item.get('标签', []))
        return sorted(tags)


class OllamaClient:
    """与Ollama模型API交互的客户端"""

    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.cache = {}

    def generate_response(self, prompt, history=[], stream=False):
        cache_key = (prompt, tuple(history))
        if cache_key in self.cache:
            return self.cache[cache_key]

        messages = history + [{"role": "user", "content": prompt}]
        payload = {
            "model": "llama2",
            "messages": messages,
            "stream": stream
        }

        if not stream:
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            result = response.json()["message"]["content"]
            self.cache[cache_key] = result
            return result

        # 流式响应处理
        response = requests.post(
            f"{self.base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    chunk = json.loads(decoded_line[6:])
                    if "content" in chunk.get("message", {}):
                        full_response += chunk["message"]["content"]

        self.cache[cache_key] = full_response
        return full_response


class DialogueManager:
    """对话管理类"""

    def __init__(self, kb_path):
        self.kb = KnowledgeBase(kb_path)
        self.llm = OllamaClient()
        self.history = []

    def process_question(self, question, selected_tags=None):
        """处理用户问题，结合知识库和大模型"""
        kb_answers = self.kb.search_answers(question, selected_tags)
        if kb_answers:
            context = "\n".join(
                [f"参考资料：{ans['答案']}（标签：{', '.join(ans.get('标签', []))}）" for ans in kb_answers])
            prompt = f"这是一个职业生涯规划相关的问题：{question}\n{context}\n请用简洁语言回答，并提供职业生涯规划方面的相关建议"
        else:
            prompt = f"这是一个职业生涯规划相关的问题：{question}\n请用简洁语言回答，并提供职业生涯规划方面的相关建议"

        response = self.llm.generate_response(prompt, self.history)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        return response


class ChatGUI:
    """聊天界面类"""

    def __init__(self, root, kb_path):
        self.root = root
        self.root.title("职业生涯规划助手")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        self.manager = DialogueManager(kb_path)
        self.kb_path = kb_path

        # 标签管理相关变量
        self.selected_tags = []
        self.tag_filters = []

        # 创建主界面
        self._create_main_frame()

        # 初始化知识库和标签
        self._update_tags_display()

    def _create_main_frame(self):
        """创建主界面布局"""
        # 配置样式
        style = ttk.Style()
        style.configure("TLabelframe.Label", font=("Arial", 14))  # 设置LabelFrame标题字体（注意大小写）
        style.configure("TButton", font=("Arial", 12))
        style.configure("Treeview", font=("Arial", 12))
        style.configure("Treeview.Heading", font=("Arial", 12))

        # 顶部框架 - 标签筛选
        top_frame = ttk.Frame(self.root, style="TFrame")
        top_frame.pack(fill=tk.X, padx=20, pady=20)

        ttk.Label(top_frame, text="标签筛选:", font=("Arial", 12)).pack(side=tk.LEFT, padx=(0, 10))

        # 标签选择下拉框
        self.tag_combobox = ttk.Combobox(top_frame, state="readonly", width=20, font=("Arial", 12))
        self.tag_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.tag_combobox.bind("<<ComboboxSelected>>", self._on_tag_selected)

        # 已选标签显示
        self.selected_tags_frame = ttk.Frame(top_frame)
        self.selected_tags_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 中间框架 - 聊天和知识库
        middle_frame = ttk.Frame(self.root, style="TFrame")
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 左侧聊天区域 - 使用默认样式或正确的自定义样式
        chat_frame = ttk.LabelFrame(middle_frame, text="智能对话")  # 使用默认样式
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 聊天历史
        self.chat_history = scrolledtext.ScrolledText(chat_frame, width=60, height=20, wrap=tk.WORD, font=("Arial", 12))
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_history.tag_config("user", foreground="#007BFF")
        self.chat_history.tag_config("ai", foreground="#28A745")

        # 输入区域
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.input_entry = ttk.Entry(input_frame, width=70, font=("Arial", 12))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", lambda event: self.send_message())

        self.send_button = ttk.Button(input_frame, text="发送", command=self.send_message, style="TButton")
        self.send_button.pack(side=tk.RIGHT)

        # 右侧知识库区域 - 使用默认样式
        kb_frame = ttk.LabelFrame(middle_frame, text="知识库")  # 使用默认样式
        kb_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # 知识列表
        self.knowledge_tree = ttk.Treeview(kb_frame, columns=("问题", "标签"), show="headings", style="Treeview")
        self.knowledge_tree.heading("问题", text="问题", anchor=tk.W)
        self.knowledge_tree.heading("标签", text="标签", anchor=tk.W)
        self.knowledge_tree.column("问题", width=300)
        self.knowledge_tree.column("标签", width=150)
        self.knowledge_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.knowledge_tree.bind("<Double-1>", self._on_knowledge_selected)

        # 底部框架 - 功能按钮
        bottom_frame = ttk.Frame(self.root, style="TFrame")
        bottom_frame.pack(fill=tk.X, padx=20, pady=20)

        self.import_button = ttk.Button(bottom_frame, text="导入知识库", command=self.import_kb, style="TButton")
        self.import_button.pack(side=tk.LEFT, padx=(0, 10))

        self.add_knowledge_button = ttk.Button(bottom_frame, text="添加知识", command=self.add_knowledge, style="TButton")
        self.add_knowledge_button.pack(side=tk.LEFT, padx=(0, 10))

        self.export_button = ttk.Button(bottom_frame, text="导出知识库", command=self.export_kb, style="TButton")
        self.export_button.pack(side=tk.LEFT, padx=(0, 10))

        self.refresh_button = ttk.Button(bottom_frame, text="刷新", command=self._refresh_knowledge_display, style="TButton")
        self.refresh_button.pack(side=tk.LEFT, padx=(0, 10))

        # 加载标签
        self._update_tag_combobox()

    def _update_tag_combobox(self):
        """更新标签下拉框"""
        tags = self.manager.kb.get_all_tags()
        self.tag_combobox['values'] = tags
        if tags:
            self.tag_combobox.current(0)

    def _on_tag_selected(self, event):
        """处理标签选择事件"""
        selected_tag = self.tag_combobox.get()
        if selected_tag and selected_tag not in self.selected_tags:
            self.selected_tags.append(selected_tag)
            self._update_tags_display()
            self._refresh_knowledge_display()

    def _update_tags_display(self):
        """更新已选标签显示"""
        # 清除现有标签
        for widget in self.selected_tags_frame.winfo_children():
            widget.destroy()

        # 添加新标签
        for tag in self.selected_tags:
            tag_frame = ttk.Frame(self.selected_tags_frame)
            tag_frame.pack(side=tk.LEFT, padx=(0, 5))

            ttk.Label(tag_frame, text=tag, background="#e2e8f0", padding=(5, 2),
                      borderwidth=1, relief="solid",
                      font=("Arial", 10)).pack(side=tk.LEFT)

            remove_btn = ttk.Button(tag_frame, text="×", width=2,
                                    command=lambda t=tag: self._remove_tag(t))
            remove_btn.pack(side=tk.RIGHT)

        # 添加清除所有按钮
        if self.selected_tags:
            clear_btn = ttk.Button(self.selected_tags_frame, text="清除所有",
                                   command=self._clear_all_tags)
            clear_btn.pack(side=tk.LEFT, padx=(5, 0))

    def _remove_tag(self, tag):
        """移除选中的标签"""
        if tag in self.selected_tags:
            self.selected_tags.remove(tag)
            self._update_tags_display()
            self._refresh_knowledge_display()

    def _clear_all_tags(self):
        """清除所有选中的标签"""
        self.selected_tags = []
        self._update_tags_display()
        self._refresh_knowledge_display()

    def _refresh_knowledge_display(self):
        """刷新知识库显示"""
        # 清空现有内容
        for item in self.knowledge_tree.get_children():
            self.knowledge_tree.delete(item)

        # 获取筛选后的知识条目
        if self.selected_tags:
            knowledge_items = self.manager.kb.search_answers("", self.selected_tags)
        else:
            knowledge_items = self.manager.kb.data

        # 添加到列表
        for item in knowledge_items:
            tags_text = ", ".join(item.get('标签', []))
            self.knowledge_tree.insert("", tk.END, values=(item['问题'], tags_text))

    def _on_knowledge_selected(self, event):
        """处理知识库项双击事件"""
        selection = self.knowledge_tree.selection()
        if selection:
            item = self.knowledge_tree.item(selection[0])
            question = item['values'][0]
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, question)

    def send_message(self):
        """发送消息处理"""
        question = self.input_entry.get().strip()
        if not question:
            return

        if question.lower() == "exit":
            self.root.destroy()
        else:
            # 显示用户问题
            self.chat_history.insert(tk.END, f"你：{question}\n", "user")
            self.chat_history.see(tk.END)

            # 清空输入框
            self.input_entry.delete(0, tk.END)

            # 处理问题
            Thread(target=self._process_question, args=(question,)).start()

    def _process_question(self, question):
        """在后台线程中处理问题"""
        try:
            # 获取当前选中的标签
            tags = self.selected_tags if self.selected_tags else None

            # 处理问题
            answer = self.manager.process_question(question, tags)

            # 在主线程中更新UI
            self.root.after(0, self._update_chat_history, question, answer)
        except Exception as e:
            self.root.after(0, self._show_error, f"处理请求时出现错误：{str(e)}")

    def _update_chat_history(self, question, answer):
        """更新聊天历史"""
        self.chat_history.insert(tk.END, f"AI：{answer}\n\n", "ai")
        self.chat_history.see(tk.END)

    def _show_error(self, error_msg):
        """显示错误消息"""
        messagebox.showerror("错误", error_msg)

    def import_kb(self):
        """导入知识库"""
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            self.manager.kb = KnowledgeBase(file_path)
            self.kb_path = file_path
            self._update_tag_combobox()
            self._refresh_knowledge_display()
            messagebox.showinfo("成功", "知识库导入成功")

    def export_kb(self):
        """导出知识库"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if file_path:
            self.manager.kb.save_to_file(file_path)
            messagebox.showinfo("成功", "知识库导出成功")

    def add_knowledge(self):
        """添加新知识"""
        add_window = tk.Toplevel(self.root)
        add_window.title("添加知识")
        add_window.geometry("500x300")

        # 问题
        ttk.Label(add_window, text="问题:", font=("Arial", 12)).pack(pady=5, padx=10, anchor=tk.W)
        question_entry = ttk.Entry(add_window, width=60, font=("Arial", 12))
        question_entry.pack(fill=tk.X, padx=10)

        # 答案
        ttk.Label(add_window, text="答案:", font=("Arial", 12)).pack(pady=5, padx=10, anchor=tk.W)
        answer_entry = scrolledtext.ScrolledText(add_window, width=60, height=5, font=("Arial", 12))
        answer_entry.pack(fill=tk.X, padx=10)

        # 标签
        ttk.Label(add_window, text="标签 (用逗号分隔):", font=("Arial", 12)).pack(pady=5, padx=10, anchor=tk.W)
        tags_entry = ttk.Entry(add_window, width=60, font=("Arial", 12))
        tags_entry.pack(fill=tk.X, padx=10)

        def save_knowledge():
            question = question_entry.get().strip()
            answer = answer_entry.get("1.0", tk.END).strip()
            tags_text = tags_entry.get().strip()

            if not question or not answer:
                messagebox.showerror("错误", "问题和答案不能为空")
                return

            # 处理标签
            tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()] if tags_text else []

            # 添加到知识库
            self.manager.kb.add_entry(question, answer, tags)

            # 更新UI
            self._update_tag_combobox()
            self._refresh_knowledge_display()

            messagebox.showinfo("成功", "知识添加成功")
            add_window.destroy()

        ttk.Button(add_window, text="保存", command=save_knowledge, style="TButton").pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatGUI(root, "knowledge_base.json")
    root.mainloop()