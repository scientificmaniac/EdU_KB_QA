"""职业生涯规划智能问答系统。

一个基于本地知识库 + Ollama 大模型的桌面问答应用。

模块划分::

    career_qa.config          配置加载（默认值 < config.json < 环境变量 < 命令行）
    career_qa.knowledge_base  知识库读写与中文相似度检索
    career_qa.llm_client      Ollama HTTP 客户端（支持流式）
    career_qa.dialogue        对话编排（检索 + 提示词 + 多轮历史）
    career_qa.gui             tkinter 图形界面
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]
