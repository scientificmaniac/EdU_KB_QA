"""知识库：JSON 读写与中文相似度检索。

相比原实现（``question in item['问题']`` 的纯子串匹配），这里采用
**字符 bigram Dice 系数 + 序列相似度** 的混合打分，使「职业规划怎么做」
这类与条目表述不同的提问也能命中，同时用阈值挡住无关条目，
避免污染大模型的参考资料。

设计要点：

* 读写全部容错，JSON 损坏时备份坏文件并降级为空库，而不是让程序崩溃。
* 保存使用「临时文件 + ``os.replace``」的原子写入，避免中途失败写坏知识库。
* 仅依赖标准库，不引入分词器，保持零额外依赖。
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

# 打分权重（两者之和为 1）
_WEIGHT_DICE = 0.6
_WEIGHT_SEQUENCE = 0.4
#: 「提问被条目标题完整包含」时的分数下限
_CONTAINMENT_BASE = 0.70
#: 覆盖度（提问长度 / 标题长度）带来的分数增量上限
_CONTAINMENT_SPAN = 0.25
#: 提问命中条目某个标签时的加分
_TAG_BONUS = 0.15
#: 标签加分上限
_MAX_TAG_BONUS = 0.3

_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


class KnowledgeBaseError(Exception):
    """知识库操作失败的基类。"""


class DuplicateEntryError(KnowledgeBaseError):
    """已存在完全相同的问题。"""


def normalize(text: str) -> str:
    """归一化文本：NFKC 折叠全半角、转小写、去除空白与标点。

    >>> normalize("如何进行  职业规划？")
    '如何进行职业规划'
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return _NON_WORD_RE.sub("", text)


def _bigrams(text: str) -> set[str]:
    """返回字符二元组集合；长度不足 2 时退化为单字符集合。"""
    if not text:
        return set()
    if len(text) == 1:
        return {text}
    return {text[i : i + 2] for i in range(len(text) - 1)}


def dice_coefficient(left: str, right: str) -> float:
    """字符 bigram 的 Dice 系数，取值 [0, 1]。"""
    a, b = _bigrams(left), _bigrams(right)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def similarity(query: str, target: str) -> float:
    """计算两个已归一化字符串的相似度，取值 [0, 1]。"""
    if not query or not target:
        return 0.0
    if query == target:
        return 1.0

    score = _WEIGHT_DICE * dice_coefficient(query, target) + _WEIGHT_SEQUENCE * (
        difflib.SequenceMatcher(None, query, target).ratio()
    )
    if query in target:
        # 用「提问在标题中的占比」给出连续分数。若固定给一个高分，
        # 所有包含关系叠加标签加分后都会饱和到 1.0，
        # 排序与 min_score 阈值会同时失去区分度。
        coverage = len(query) / len(target)
        score = max(score, _CONTAINMENT_BASE + _CONTAINMENT_SPAN * coverage)
    return min(score, 1.0)


def parse_tags(raw: Any) -> tuple[str, ...]:
    """把任意形式的标签输入规范化成去重后的标签元组。"""
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: Iterable[Any] = raw.replace("，", ",").replace("、", ",").split(",")
    elif isinstance(raw, (list, tuple, set)):
        items = raw
    else:
        items = [raw]

    seen: list[str] = []
    for item in items:
        tag = str(item).strip()
        if tag and tag not in seen:
            seen.append(tag)
    return tuple(seen)


@dataclass(frozen=True)
class Entry:
    """一条知识条目。"""

    question: str
    answer: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"问题": self.question, "答案": self.answer, "标签": list(self.tags)}

    @classmethod
    def from_dict(cls, raw: Any, position: int = 0) -> "Entry":
        """从 JSON 字典构造条目，字段缺失或类型异常时抛出 ValueError。"""
        if not isinstance(raw, dict):
            raise ValueError(f"第 {position} 条不是 JSON 对象：{type(raw).__name__}")

        question = raw.get("问题", raw.get("question"))
        answer = raw.get("答案", raw.get("answer"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"第 {position} 条缺少有效的「问题」字段")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"第 {position} 条缺少有效的「答案」字段")

        return cls(
            question=question.strip(),
            answer=answer.strip(),
            tags=parse_tags(raw.get("标签", raw.get("tags"))),
        )


@dataclass(frozen=True)
class SearchHit:
    """一次检索结果。"""

    entry: Entry
    score: float

    @property
    def question(self) -> str:
        return self.entry.question

    @property
    def answer(self) -> str:
        return self.entry.answer

    @property
    def tags(self) -> tuple[str, ...]:
        return self.entry.tags


@dataclass
class KnowledgeBase:
    """标签化知识库。

    :param file_path: 知识库 JSON 路径
    :param strict: 为 ``True`` 时，文件损坏直接抛异常；默认降级为空库
    """

    file_path: Path
    strict: bool = False
    _entries: list[Entry] = field(default_factory=list, repr=False)
    #: 加载过程中产生的非致命问题（坏文件备份、被跳过的条目等）
    warnings: list[str] = field(default_factory=list, repr=False)
    #: 内存副本是否已完成一次加载（无论成功与否）
    _loaded: bool = field(default=False, repr=False)
    #: 上一次加载是否因 I/O 故障失败；为 True 时禁止写回原文件，
    #: 以免用不完整的内存副本覆盖真实数据
    _read_failed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    @property
    def entries(self) -> tuple[Entry, ...]:
        if not self._loaded:
            self.load()
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self.entries)

    def load(self) -> None:
        """从磁盘加载知识库；失败时按 ``strict`` 决定抛错还是降级。

        两类失败被区别对待：

        * **JSON 语法损坏** —— 文件内容确实已不可用，备份为
          ``*.corrupt-<时间戳>`` 后以空库启动；
        * **I/O 故障**（``OSError``：文件被占用、权限不足、网络盘掉线）——
          文件内容很可能完好，**绝不移动或重命名原文件**，只记录告警、
          以空库启动，并置 :attr:`_read_failed` 禁止后续写回原文件，
          避免一次读取抖动演变成数据丢失。
        """
        self.warnings = []
        self._entries = []
        self._loaded = True
        self._read_failed = False

        if not self.file_path.is_file():
            # 文件不存在是正常情况（首次运行），后续保存会自动创建
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            if self.strict:
                raise KnowledgeBaseError(
                    f"知识库 {self.file_path} 不是合法 JSON：{exc}"
                ) from exc
            backup = self._backup_corrupt_file()
            self.warnings.append(
                f"知识库 {self.file_path} 无法解析（{exc}），"
                f"已备份为 {backup.name} 并从空库开始。"
            )
            return
        except OSError as exc:
            # 读取失败属于环境问题，绝不能当作数据损坏去改名文件
            self._read_failed = True
            message = f"无法读取知识库 {self.file_path}：{exc}"
            if self.strict:
                raise KnowledgeBaseError(message) from exc
            self.warnings.append(
                message + "。原文件未被改动；在排除占用/权限问题之前，"
                "本程序不会把数据写回该文件。"
            )
            return

        if isinstance(raw, dict):
            # 兼容 {"entries": [...]} / {"data": [...]} 形式的包装
            for key in ("entries", "data", "items", "知识库"):
                if isinstance(raw.get(key), list):
                    raw = raw[key]
                    break
            else:
                raw = [raw]

        if not isinstance(raw, list):
            message = f"知识库顶层应为 JSON 数组，实际为 {type(raw).__name__}"
            if self.strict:
                raise KnowledgeBaseError(message)
            self.warnings.append(message + "，已按空库处理。")
            return

        skipped = 0
        for index, item in enumerate(raw):
            try:
                self._entries.append(Entry.from_dict(item, index))
            except ValueError as exc:
                skipped += 1
                if self.strict:
                    raise KnowledgeBaseError(str(exc)) from exc
                self.warnings.append(str(exc))

        if skipped:
            self.warnings.append(f"共跳过 {skipped} 条无法解析的知识条目。")

    def reload(self) -> None:
        """丢弃内存副本，重新从磁盘加载。"""
        self.load()

    def _backup_corrupt_file(self) -> Path:
        """把损坏的知识库改名备份，返回实际备份路径。

        同一秒内发生二次损坏时追加序号，避免覆盖上一次的备份。
        """
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        base = self.file_path.with_name(f"{self.file_path.name}.corrupt-{stamp}")
        backup = base
        counter = 1
        while backup.exists():
            backup = base.with_name(f"{base.name}-{counter}")
            counter += 1
        try:
            os.replace(self.file_path, backup)
        except OSError:
            return self.file_path
        return backup

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """确保内存副本已从磁盘加载，避免未加载时误写空文件。"""
        if not self._loaded:
            self.load()

    def _to_list(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return [entry.to_dict() for entry in self._entries]

    def _is_same_file(self, other: str | Path) -> bool:
        """判断 ``other`` 是否指向本知识库文件（目标不存在时也安全）。"""
        try:
            return Path(other).resolve() == self.file_path.resolve()
        except OSError:  # pragma: no cover - resolve 极少失败
            return Path(other).absolute() == self.file_path.absolute()

    def save(self, path: str | Path | None = None) -> Path:
        """原子写入知识库，返回实际写入路径。

        :raises KnowledgeBaseError: 写入失败；或此前读取该文件失败时
            拒绝写回（防止用不完整的内存副本覆盖真实数据）
        """
        self._ensure_loaded()
        target = Path(path) if path is not None else self.file_path

        if self._read_failed and self._is_same_file(target):
            raise KnowledgeBaseError(
                f"知识库 {self.file_path} 此前读取失败，为防止覆盖原数据已拒绝写入。"
                "请确认文件未被其他程序占用、且当前用户具有读取权限后重试。"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._to_list(), ensure_ascii=False, indent=4)

        handle = None
        temp_name: str | None = None
        try:
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".tmp",
                prefix=f".{target.name}.",
                dir=str(target.parent),
                delete=False,
            )
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_name = handle.name
            handle.close()
            handle = None
            os.replace(temp_name, target)
            temp_name = None
        except OSError as exc:
            if handle is not None:
                handle.close()
            if temp_name is not None:
                # 清理写了一半的临时文件，避免在用户目录里越积越多
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
            raise KnowledgeBaseError(f"保存知识库到 {target} 失败：{exc}") from exc

        if self._is_same_file(target):
            self._loaded = True
            self._read_failed = False
        return target

    # ------------------------------------------------------------------
    # 增删改
    # ------------------------------------------------------------------
    def add_entry(
        self,
        question: str,
        answer: str,
        tags: Sequence[str] | str | None = None,
        *,
        allow_duplicate: bool = False,
    ) -> Entry:
        """新增条目并落盘。默认拒绝重复问题。"""
        # 必须在 append 之前完成加载：save() 会触发惰性加载，
        # 若先 append 再 load，刚加入的条目会被磁盘内容覆盖掉。
        self._ensure_loaded()

        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question:
            raise KnowledgeBaseError("问题不能为空")
        if not answer:
            raise KnowledgeBaseError("答案不能为空")

        if not allow_duplicate:
            key = normalize(question)
            for existing in self.entries:
                if normalize(existing.question) == key:
                    raise DuplicateEntryError(f"知识库中已存在相同的问题：{existing.question}")

        entry = Entry(question=question, answer=answer, tags=parse_tags(tags))
        self._entries.append(entry)
        try:
            self.save()
        except KnowledgeBaseError:
            # 落盘失败必须回滚内存，否则用户重试同一条会被误判为「重复条目」，
            # 内存与磁盘就此长期不一致
            self._entries.pop()
            raise
        return entry

    def update_entry(
        self,
        index: int,
        question: str | None = None,
        answer: str | None = None,
        tags: Sequence[str] | str | None = None,
    ) -> Entry:
        """按下标更新条目，``None`` 表示保持原值。"""
        self._check_index(index)
        old = self._entries[index]
        new = Entry(
            question=(question or old.question).strip(),
            answer=(answer or old.answer).strip(),
            tags=parse_tags(tags) if tags is not None else old.tags,
        )
        if not new.question or not new.answer:
            raise KnowledgeBaseError("问题和答案不能为空")

        key = normalize(new.question)
        for position, existing in enumerate(self._entries):
            if position != index and normalize(existing.question) == key:
                raise DuplicateEntryError(
                    f"知识库中已存在相同的问题：{existing.question}"
                )

        self._entries[index] = new
        try:
            self.save()
        except KnowledgeBaseError:
            self._entries[index] = old
            raise
        return new

    def remove_entry(self, index: int) -> Entry:
        """按下标删除条目并落盘，返回被删除的条目。"""
        self._check_index(index)
        entry = self._entries.pop(index)
        try:
            self.save()
        except KnowledgeBaseError:
            self._entries.insert(index, entry)
            raise
        return entry

    def _check_index(self, index: int) -> None:
        self._ensure_loaded()
        # bool 是 int 的子类，若不排除，remove_entry(True) 会误删第 2 条
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self._entries)
        ):
            raise KnowledgeBaseError(f"知识条目下标越界：{index}")

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def filter_by_tags(self, tags: Sequence[str] | None = None) -> list[Entry]:
        """按标签筛选（任一命中即保留）；``tags`` 为空则返回全部。"""
        normalized = {tag.strip() for tag in (tags or ()) if str(tag).strip()}
        if not normalized:
            return list(self.entries)
        return [
            entry for entry in self.entries if normalized & set(entry.tags)
        ]

    def score(self, question: str, entry: Entry) -> float:
        """计算提问与条目的综合得分，取值 [0, 1]。

        空提问一律返回 ``1.0``，语义是「不做相关性筛选」。
        :meth:`search` 对空提问走单独的「按标签浏览」分支，不依赖这个返回值。
        """
        q = normalize(question)
        if not q:
            return 1.0

        best = similarity(q, normalize(entry.question))

        # 标签命中加分：提问中出现了该条目的标签文本
        bonus = 0.0
        for tag in entry.tags:
            tag_norm = normalize(tag)
            if tag_norm and tag_norm in q:
                bonus += _TAG_BONUS
        best = min(best + min(bonus, _MAX_TAG_BONUS), 1.0)
        return best

    def search(
        self,
        question: str,
        tags: Sequence[str] | None = None,
        top_n: int = 3,
        min_score: float = 0.25,
    ) -> list[SearchHit]:
        """检索知识条目，按得分降序返回至多 ``top_n`` 条。

        ``question`` 为空时视为「仅按标签浏览」，返回命中的前 ``top_n`` 条。
        """
        candidates = self.filter_by_tags(tags)

        if not normalize(question):
            return [SearchHit(entry, 1.0) for entry in candidates[: max(top_n, 0)]]

        hits: list[SearchHit] = []
        for entry in candidates:
            score = self.score(question, entry)
            if score >= min_score:
                hits.append(SearchHit(entry, score))

        # 稳定排序：得分相同时保持知识库中的原始顺序
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(top_n, 0)]

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def all_tags(self) -> list[str]:
        """返回排序后的全部标签。"""
        tags: set[str] = set()
        for entry in self.entries:
            tags.update(entry.tags)
        return sorted(tags)

    def tag_counts(self) -> dict[str, int]:
        """返回每个标签的条目数，按数量降序、名称升序。"""
        counter: Counter[str] = Counter()
        for entry in self.entries:
            counter.update(entry.tags)
        return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))

    def to_list(self) -> list[dict[str, Any]]:
        """导出为可 JSON 序列化的列表。"""
        return self._to_list()
