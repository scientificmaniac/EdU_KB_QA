"""命令行入口：自检、模型列举与 GUI 启动。

用法::

    python app.py                          # 启动图形界面
    python app.py --check                  # 环境自检
    python app.py --list-models            # 列出 Ollama 已安装模型
    python app.py --model qwen2 --no-stream
"""

from __future__ import annotations

import logging
import shutil
import sys
from typing import Sequence

from .config import (
    PROJECT_ROOT,
    AppConfig,
    ConfigError,
    bundled_root,
    load_config,
)
from .knowledge_base import KnowledgeBase
from .llm_client import LLMError, OllamaClient

#: 打包后需要从临时解包目录复制到 exe 旁边的数据文件
_BUNDLED_DATA_FILES = ("knowledge_base.json", "config.example.json")


def bootstrap_bundled_data() -> list[str]:
    """把 PyInstaller 内嵌的数据文件释放到 exe 同级目录（首次运行时）。

    返回实际复制出来的文件名列表。源码运行时始终返回空列表。
    """
    source_root = bundled_root()
    if source_root is None:
        return []

    copied: list[str] = []
    for name in _BUNDLED_DATA_FILES:
        source = source_root / name
        target = PROJECT_ROOT / name
        if target.exists() or not source.is_file():
            continue
        try:
            shutil.copy2(source, target)
        except OSError as exc:  # pragma: no cover - 依赖文件系统状态
            logging.getLogger(__name__).warning("释放 %s 失败：%s", name, exc)
            continue
        copied.append(name)
    return copied


def _make_client(config: AppConfig) -> OllamaClient:
    return OllamaClient(
        base_url=config.ollama_base_url,
        model=config.model,
        connect_timeout=config.connect_timeout,
        request_timeout=config.request_timeout,
        max_retries=config.max_retries,
        cache_size=config.cache_size,
        temperature=config.temperature,
    )


def run_check(config: AppConfig) -> int:
    """打印环境自检报告，返回进程退出码（0 表示全部通过）。"""
    problems: list[str] = []

    print("=" * 62)
    print("职业生涯规划智能问答系统 · 环境自检")
    print("=" * 62)

    print("\n[配置]")
    print(f"  知识库路径   : {config.kb_path}")
    print(f"  Ollama 地址  : {config.ollama_base_url}")
    print(f"  模型         : {config.model}")
    print(f"  检索条数/阈值: top_n={config.top_n}, min_score={config.min_score}")
    print(f"  历史轮数     : {config.max_history_turns}")
    print(f"  流式输出     : {'开启' if config.stream else '关闭'}")

    print("\n[知识库]")
    kb = KnowledgeBase(config.kb_path)
    if kb.entries:
        print(f"  条目数量     : {len(kb)}")
        tags = kb.all_tags()
        print(f"  标签数量     : {len(tags)}")
        if tags:
            preview = "、".join(tags[:12])
            print(f"  标签预览     : {preview}{' …' if len(tags) > 12 else ''}")
    else:
        print("  条目数量     : 0（知识库为空或尚未创建）")
    for warning in kb.warnings:
        print(f"  ! 告警       : {warning}")
        problems.append(warning)

    print("\n[Ollama]")
    client = _make_client(config)
    try:
        models = client.list_models()
    except LLMError as exc:
        print(f"  ! 无法连接   : {exc}")
        problems.append(str(exc))
    else:
        print(f"  服务状态     : 在线")
        print(f"  已安装模型   : {len(models)} 个")
        for name in models[:10]:
            print(f"    - {name}")
        if models and config.model not in {m.split(":")[0] for m in models} and (
            config.model not in models
        ):
            print(f"  ! 目标模型 {config.model!r} 不在已安装列表中，")
            print(f"    可执行 `ollama pull {config.model}` 安装。")
            problems.append(f"模型 {config.model} 未安装")

    print("\n" + "=" * 62)
    if problems:
        print(f"自检完成：发现 {len(problems)} 个问题，请按上面的提示处理。")
        return 1
    print("自检完成：一切正常，可以运行 `python app.py` 启动界面。")
    return 0


def run_list_models(config: AppConfig) -> int:
    client = _make_client(config)
    try:
        models = client.list_models()
    except LLMError as exc:
        print(f"无法获取模型列表：{exc}", file=sys.stderr)
        return 1

    if not models:
        print("Ollama 中尚未安装任何模型。")
        return 0
    print(f"已安装 {len(models)} 个模型：")
    for name in models:
        marker = "  <- 当前配置" if name == config.model else ""
        print(f"  - {name}{marker}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """程序主入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config, args = load_config(argv=argv)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    # 打包运行时，先把内嵌的示例知识库释放到 exe 同级目录
    bootstrap_bundled_data()

    if args.list_models:
        return run_list_models(config)

    if args.check:
        return run_check(config)

    # 延迟导入：无图形环境下 --check / --list-models 也能正常工作
    from .gui import run as run_gui

    try:
        return run_gui(config)
    except Exception as exc:  # noqa: BLE001 - 顶层兜底，给出可读提示
        logging.getLogger(__name__).exception("界面启动失败")
        print(f"界面启动失败：{exc}", file=sys.stderr)
        return 1


__all__ = ["main", "run_check", "run_list_models", "bootstrap_bundled_data"]
