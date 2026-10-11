"""应用配置。

配置优先级（后者覆盖前者）::

    内置默认值  <  config.json  <  环境变量  <  命令行参数

环境变量统一使用 ``CAREERQA_`` 前缀，例如 ``CAREERQA_MODEL=qwen2``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from . import __version__


def detect_project_root() -> Path:
    """确定「数据根目录」。

    * 源码运行时：仓库根目录（``career_qa/`` 的上一级）；
    * PyInstaller 打包后：``exe`` 所在目录，这样用户可以就地编辑
      知识库与配置文件，而不是面对只读的临时解包目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_root() -> Path | None:
    """返回 PyInstaller 的临时解包目录；非打包环境返回 ``None``。"""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else None


# 项目根目录（career_qa/ 的上一级）
PROJECT_ROOT = detect_project_root()
DEFAULT_KB_PATH = PROJECT_ROOT / "knowledge_base.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

ENV_PREFIX = "CAREERQA_"

#: 环境变量名 -> 配置字段名
ENV_FIELDS: dict[str, str] = {
    "KB_PATH": "kb_path",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "MODEL": "model",
    "CONNECT_TIMEOUT": "connect_timeout",
    "REQUEST_TIMEOUT": "request_timeout",
    "MAX_RETRIES": "max_retries",
    "MAX_HISTORY_TURNS": "max_history_turns",
    "TOP_N": "top_n",
    "MIN_SCORE": "min_score",
    "STREAM": "stream",
    "CACHE_SIZE": "cache_size",
    "TEMPERATURE": "temperature",
}

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "n"}


class ConfigError(ValueError):
    """配置文件格式错误或取值非法。"""


@dataclass(frozen=True)
class AppConfig:
    """不可变的应用配置对象。

    使用 :meth:`with_overrides` 派生新配置，避免运行期被意外修改。
    """

    #: 知识库 JSON 文件路径
    kb_path: Path = DEFAULT_KB_PATH
    #: Ollama 服务地址
    ollama_base_url: str = "http://localhost:11434"
    #: 使用的模型名称
    model: str = "llama2"
    #: 建立连接的超时（秒）
    connect_timeout: float = 5.0
    #: 读取响应的超时（秒）；本地大模型首字延迟较高，默认给足
    request_timeout: float = 120.0
    #: 网络类失败的重试次数（不含首次请求）
    max_retries: int = 2
    #: 保留的历史对话轮数（1 轮 = 1 问 + 1 答）
    max_history_turns: int = 3
    #: 知识库检索最多返回的条目数
    top_n: int = 3
    #: 检索相似度阈值，低于该值的条目视为不相关
    min_score: float = 0.25
    #: 是否使用流式输出
    stream: bool = True
    #: LLM 响应缓存条数上限，0 表示禁用缓存
    cache_size: int = 128
    #: 采样温度
    temperature: float = 0.7
    #: 系统提示词
    system_prompt: str = (
        "你是一名严谨的职业生涯规划顾问。"
        "请结合参考资料，用简洁的简体中文回答，并给出可执行的建议。"
    )
    #: 用户提示词模板，可用占位符：{question} {context} {tags}
    prompt_template: str = (
        "这是一个职业生涯规划相关的问题：{question}\n"
        "{context}\n"
        "请用简洁语言回答，并提供职业生涯规划方面的相关建议。"
    )

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # dataclass 为 frozen，这里用 object.__setattr__ 做类型归一化
        object.__setattr__(self, "kb_path", Path(self.kb_path))

        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ConfigError(
                f"ollama_base_url 必须是 http(s) 地址，当前为 {self.ollama_base_url!r}"
            )
        if not self.model.strip():
            raise ConfigError("model 不能为空")
        if self.max_retries < 0:
            raise ConfigError("max_retries 不能为负数")
        if self.max_history_turns < 0:
            raise ConfigError("max_history_turns 不能为负数")
        if self.top_n < 1:
            raise ConfigError("top_n 必须 >= 1")
        if not 0.0 <= self.min_score <= 1.0:
            raise ConfigError("min_score 必须位于 [0, 1] 区间")
        if self.cache_size < 0:
            raise ConfigError("cache_size 不能为负数")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("temperature 必须位于 [0, 2] 区间")
        if self.connect_timeout <= 0 or self.request_timeout <= 0:
            raise ConfigError("超时时间必须为正数")

        # 归一化 base_url：去掉尾部斜杠
        object.__setattr__(self, "ollama_base_url", self.ollama_base_url.rstrip("/"))

    # ------------------------------------------------------------------
    # 派生与序列化
    # ------------------------------------------------------------------
    def with_overrides(self, **kwargs: Any) -> "AppConfig":
        """基于当前配置派生一份新配置，``None`` 值表示不覆盖。"""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        unknown = set(clean) - {f.name for f in fields(self)}
        if unknown:
            raise ConfigError(f"未知配置项：{', '.join(sorted(unknown))}")
        return replace(self, **clean)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kb_path"] = str(self.kb_path)
        return data

    def save(self, path: str | Path) -> None:
        """把当前配置写出为 JSON（用于生成 config.json）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")


# ----------------------------------------------------------------------
# 取值转换
# ----------------------------------------------------------------------
def _coerce(field_name: str, raw: Any) -> Any:
    """把 JSON / 环境变量里的原始值转换成配置字段所需的类型。"""
    if field_name == "kb_path":
        return Path(str(raw)).expanduser()
    if field_name in {"stream"}:
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE_VALUES:
            return True
        if text in _FALSE_VALUES:
            return False
        raise ConfigError(f"无法把 {raw!r} 解析为布尔值")
    if field_name in {
        "connect_timeout",
        "request_timeout",
        "min_score",
        "temperature",
    }:
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"配置项 {field_name} 需要数字，收到 {raw!r}") from exc
    if field_name in {"max_retries", "max_history_turns", "top_n", "cache_size"}:
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"配置项 {field_name} 需要整数，收到 {raw!r}") from exc
    return raw if isinstance(raw, str) else str(raw)


def load_config_file(path: str | Path) -> dict[str, Any]:
    """读取 config.json；文件不存在时返回空字典。"""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置文件 {path} 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件 {path} 的顶层必须是 JSON 对象")
    return data


def load_env(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """从环境变量读取配置覆盖项。"""
    environ = os.environ if environ is None else environ
    result: dict[str, Any] = {}
    for env_name, field_name in ENV_FIELDS.items():
        raw = environ.get(ENV_PREFIX + env_name)
        if raw is None or raw == "":
            continue
        result[field_name] = _coerce(field_name, raw)
    return result


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="career-qa",
        description="职业生涯规划智能问答系统（本地知识库 + Ollama）",
    )
    parser.add_argument("--config", metavar="PATH", help="配置文件路径，默认 config.json")
    parser.add_argument("--kb", metavar="PATH", help="知识库 JSON 路径")
    parser.add_argument("--ollama-url", metavar="URL", help="Ollama 服务地址")
    parser.add_argument("--model", metavar="NAME", help="Ollama 模型名称")
    parser.add_argument("--top-n", type=int, metavar="N", help="检索返回条数")
    parser.add_argument(
        "--min-score", type=float, metavar="S", help="检索相似度阈值，取值 0~1"
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="关闭流式输出，等待完整回答"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅做环境自检（配置、知识库、Ollama 连通性）后退出",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="列出 Ollama 已安装模型后退出"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def load_config(
    config_path: str | Path | None = None,
    argv: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
    require_file: bool = False,
) -> tuple[AppConfig, argparse.Namespace]:
    """按优先级合并配置，返回 ``(AppConfig, 命令行命名空间)``。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if config_path is None:
        config_path = args.config or DEFAULT_CONFIG_PATH
    config_path = Path(config_path)

    # 默认值
    data: dict[str, Any] = {}

    # 1) 配置文件
    file_data = load_config_file(config_path)
    if require_file and not file_data:
        raise ConfigError(f"配置文件不存在或为空：{config_path}")
    for key, value in file_data.items():
        if key not in {f.name for f in fields(AppConfig)}:
            raise ConfigError(f"配置文件 {config_path} 含未知配置项：{key}")
        data[key] = _coerce(key, value)

    # 2) 环境变量
    data.update(load_env(environ))

    # 3) 命令行
    cli_data = {
        "kb_path": Path(args.kb) if args.kb else None,
        "ollama_base_url": args.ollama_url,
        "model": args.model,
        "top_n": args.top_n,
        "min_score": args.min_score,
        "stream": False if args.no_stream else None,
    }
    data.update({k: v for k, v in cli_data.items() if v is not None})

    config = AppConfig(**data)

    # 相对路径统一以项目根目录为基准解析
    if not config.kb_path.is_absolute():
        config = config.with_overrides(kb_path=(PROJECT_ROOT / config.kb_path).resolve())

    return config, args
