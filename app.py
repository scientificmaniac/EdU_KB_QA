"""职业生涯规划智能问答系统 —— 兼容入口。

v2 起核心逻辑拆分到 :mod:`career_qa` 包中，本文件保留为薄入口，
因此 ``python app.py`` 以及 PyInstaller 的 ``['app.py']`` 配置都可以
继续使用。

推荐用法::

    python app.py                 # 启动图形界面
    python app.py --check         # 环境自检（配置 / 知识库 / Ollama）
    python app.py --list-models   # 列出已安装的 Ollama 模型
    python -m career_qa           # 等价的模块启动方式
"""

from __future__ import annotations

from typing import Any

from career_qa import __version__
from career_qa.cli import main
from career_qa.config import AppConfig, ConfigError, load_config
from career_qa.dialogue import Answer, DialogueManager
from career_qa.knowledge_base import (
    Entry,
    KnowledgeBase,
    KnowledgeBaseError,
    SearchHit,
)
from career_qa.llm_client import LLMError, OllamaClient

__all__ = [
    "__version__",
    "main",
    "AppConfig",
    "ConfigError",
    "load_config",
    "Answer",
    "DialogueManager",
    "Entry",
    "KnowledgeBase",
    "KnowledgeBaseError",
    "SearchHit",
    "LLMError",
    "OllamaClient",
    "ChatGUI",
]


def __getattr__(name: str) -> Any:
    """延迟导出 :class:`career_qa.gui.ChatGUI`。

    这样在没有图形环境（未安装 tkinter）时，``--check`` 等命令行
    功能依然可用，同时保留 ``from app import ChatGUI`` 的老写法。
    """
    if name == "ChatGUI":
        from career_qa.gui import ChatGUI

        return ChatGUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())
