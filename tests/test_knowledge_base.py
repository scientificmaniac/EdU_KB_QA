"""知识库模块的单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from career_qa.knowledge_base import (
    DuplicateEntryError,
    Entry,
    KnowledgeBase,
    KnowledgeBaseError,
    dice_coefficient,
    normalize,
    parse_tags,
    similarity,
)

SAMPLE = [
    {"问题": "如何进行职业规划?", "答案": "明确优势与目标。", "标签": ["职业规划", "目标设定"]},
    {"问题": "数据科学家的职业前景如何?", "答案": "需求旺盛。", "标签": ["职业前景", "数据科学"]},
    {"问题": "如何准备面试?", "答案": "梳理项目经历。", "标签": ["求职", "面试"]},
]


class NormalizeTests(unittest.TestCase):
    def test_strips_punctuation_and_space(self):
        self.assertEqual(normalize("如何进行  职业规划？"), "如何进行职业规划")

    def test_fullwidth_folding(self):
        self.assertEqual(normalize("ＡＢＣ１２３"), "abc123")

    def test_empty(self):
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize(None), "")  # type: ignore[arg-type]

    def test_dice_identical(self):
        self.assertAlmostEqual(dice_coefficient("职业规划", "职业规划"), 1.0)

    def test_dice_disjoint(self):
        self.assertEqual(dice_coefficient("职业规划", "天气预报"), 0.0)

    def test_dice_partial(self):
        value = dice_coefficient("职业规划", "职业发展")
        self.assertTrue(0.0 < value < 1.0)

    def test_similarity_containment(self):
        score = similarity("职业规划", "如何进行职业规划")
        self.assertGreater(score, 0.70)
        self.assertLess(score, 1.0)

    def test_containment_score_scales_with_coverage(self):
        """回归测试：包含关系不能一律饱和到同一分值。

        旧实现给固定 0.95 分，叠加标签加分后普遍顶到 1.0，
        排序与阈值同时失去区分度。
        """
        short = similarity("职业", "如何做好长期职业规划")
        long = similarity("如何做好长期职业规划", "如何做好长期职业规划呢")
        self.assertGreater(long, short)
        self.assertLess(short, 0.95)

    def test_similarity_identical(self):
        self.assertEqual(similarity("abc", "abc"), 1.0)

    def test_similarity_empty(self):
        self.assertEqual(similarity("", "abc"), 0.0)
        self.assertEqual(similarity("abc", ""), 0.0)


class ParseTagsTests(unittest.TestCase):
    def test_none(self):
        self.assertEqual(parse_tags(None), ())

    def test_chinese_comma_and_dedup(self):
        self.assertEqual(parse_tags("a，b、a, c"), ("a", "b", "c"))

    def test_list_input(self):
        self.assertEqual(parse_tags(["x", " y ", ""]), ("x", "y"))

    def test_plain_string_without_separator(self):
        self.assertEqual(parse_tags("单独标签"), ("单独标签",))


class EntryTests(unittest.TestCase):
    def test_round_trip(self):
        entry = Entry("问题", "答案", ("标签",))
        self.assertEqual(Entry.from_dict(entry.to_dict()).question, "问题")

    def test_english_keys_compat(self):
        entry = Entry.from_dict({"question": "q", "answer": "a", "tags": ["t"]})
        self.assertEqual((entry.question, entry.answer, entry.tags), ("q", "a", ("t",)))

    def test_missing_question_raises(self):
        with self.assertRaises(ValueError):
            Entry.from_dict({"答案": "a"})

    def test_blank_answer_raises(self):
        with self.assertRaises(ValueError):
            Entry.from_dict({"问题": "q", "答案": "   "})

    def test_non_dict_raises(self):
        with self.assertRaises(ValueError):
            Entry.from_dict("字符串")


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "kb.json"

    def write(self, payload) -> None:
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_missing_file_is_empty(self):
        kb = KnowledgeBase(self.path)
        self.assertEqual(len(kb), 0)
        self.assertEqual(kb.warnings, [])

    def test_loads_entries(self):
        self.write(SAMPLE)
        kb = KnowledgeBase(self.path)
        self.assertEqual(len(kb), 3)
        self.assertEqual(kb.entries[0].tags, ("职业规划", "目标设定"))

    def test_corrupt_json_backs_up_and_degrades(self):
        self.path.write_text("{ 这不是 JSON", encoding="utf-8")
        kb = KnowledgeBase(self.path)
        self.assertEqual(len(kb), 0)
        self.assertTrue(kb.warnings)
        backups = list(self.path.parent.glob("kb.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertFalse(self.path.exists())

    def test_corrupt_json_strict_raises(self):
        self.path.write_text("{ 坏", encoding="utf-8")
        kb = KnowledgeBase(self.path, strict=True)
        with self.assertRaises(KnowledgeBaseError):
            kb.load()

    def test_non_list_toplevel_degrades(self):
        self.write({"foo": 1})
        kb = KnowledgeBase(self.path)
        # 字典会被尝试当作单条条目解析；缺少问题字段 → 跳过
        self.assertEqual(len(kb), 0)
        self.assertTrue(kb.warnings)

    def test_wrapped_entries_key(self):
        self.write({"entries": SAMPLE})
        kb = KnowledgeBase(self.path)
        self.assertEqual(len(kb), 3)

    def test_skips_invalid_items(self):
        self.write([SAMPLE[0], {"答案": "无问题"}, "字符串", SAMPLE[1]])
        kb = KnowledgeBase(self.path)
        self.assertEqual(len(kb), 2)
        self.assertTrue(any("跳过" in w for w in kb.warnings))

    def test_reload_picks_up_changes(self):
        self.write(SAMPLE)
        kb = KnowledgeBase(self.path)
        self.assertEqual(len(kb), 3)
        self.write(SAMPLE[:1])
        kb.reload()
        self.assertEqual(len(kb), 1)


class SaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "kb.json"

    def test_save_creates_file(self):
        kb = KnowledgeBase(self.path)
        kb.add_entry("问题", "答案", ["标签"])
        self.assertTrue(self.path.exists())
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data[0]["问题"], "问题")
        self.assertEqual(data[0]["标签"], ["标签"])

    def test_save_does_not_wipe_unloaded_kb(self):
        """回归测试：未先读取就保存，不得把磁盘上的条目清空。"""
        self.path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
        kb = KnowledgeBase(self.path)
        kb.save()  # 不应覆盖为空数组
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))), 3)

    def test_save_to_other_path_does_not_touch_source(self):
        self.path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
        kb = KnowledgeBase(self.path)
        target = Path(self.tmp.name) / "out" / "export.json"
        kb.save(target)
        self.assertTrue(target.exists())
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))), 3)

    def test_atomic_write_leaves_no_temp_files(self):
        kb = KnowledgeBase(self.path)
        kb.add_entry("问题", "答案")
        leftovers = [p.name for p in self.path.parent.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_no_bom_and_utf8(self):
        kb = KnowledgeBase(self.path)
        kb.add_entry("中文问题", "中文答案")
        raw = self.path.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn("中文问题", raw.decode("utf-8"))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "kb.json"
        self.path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
        self.kb = KnowledgeBase(self.path)

    def test_exact_match_first(self):
        hits = self.kb.search("如何进行职业规划?", min_score=0.1)
        self.assertEqual(hits[0].question, "如何进行职业规划?")

    def test_paraphrase_still_matches(self):
        """原实现的纯子串匹配无法命中这类改写提问。"""
        hits = self.kb.search("职业规划怎么做", min_score=0.2)
        self.assertTrue(hits)
        self.assertIn("职业规划", hits[0].question)

    def test_unrelated_question_filtered(self):
        hits = self.kb.search("今天天气怎么样", min_score=0.25)
        self.assertEqual(hits, [])

    def test_scores_are_descending(self):
        hits = self.kb.search("职业规划", min_score=0.05)
        scores = [hit.score for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_n_limit(self):
        hits = self.kb.search("职业", min_score=0.0, top_n=2)
        self.assertLessEqual(len(hits), 2)

    def test_tag_filter_restricts_candidates(self):
        hits = self.kb.search("如何", tags=["求职"], min_score=0.0)
        self.assertTrue(all("求职" in hit.tags for hit in hits))

    def test_empty_question_returns_tagged_entries(self):
        hits = self.kb.search("", tags=["数据科学"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].question, "数据科学家的职业前景如何?")

    def test_empty_question_without_tags_returns_top_n(self):
        hits = self.kb.search("", top_n=2)
        self.assertEqual(len(hits), 2)

    def test_score_bounds(self):
        for entry in self.kb.entries:
            self.assertGreaterEqual(self.kb.score("职业规划", entry), 0.0)
            self.assertLessEqual(self.kb.score("如何进行职业规划?", entry), 1.0)

    def test_tag_bonus_applies(self):
        entry = self.kb.entries[0]  # 标签含「职业规划」
        with_tag = self.kb.score("职业规划相关的问题", entry)
        without_tag = similarity(normalize("职业规划相关的问题"), normalize(entry.question))
        self.assertGreaterEqual(with_tag, without_tag)


class MutationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "kb.json"
        self.path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
        self.kb = KnowledgeBase(self.path)

    def test_add_entry(self):
        entry = self.kb.add_entry("新问题", "新答案", "标签A,标签B")
        self.assertEqual(entry.tags, ("标签A", "标签B"))
        self.assertEqual(len(self.kb), 4)

    def test_add_duplicate_rejected(self):
        with self.assertRaises(DuplicateEntryError):
            self.kb.add_entry("如何进行职业规划?", "另一个答案")

    def test_add_duplicate_case_insensitive_after_normalize(self):
        with self.assertRaises(DuplicateEntryError):
            self.kb.add_entry("如何进行  职业规划", "另一个答案")

    def test_add_duplicate_allowed_when_flagged(self):
        self.kb.add_entry("如何进行职业规划?", "另一个答案", allow_duplicate=True)
        self.assertEqual(len(self.kb), 4)

    def test_add_empty_question_rejected(self):
        with self.assertRaises(KnowledgeBaseError):
            self.kb.add_entry("   ", "答案")

    def test_add_empty_answer_rejected(self):
        with self.assertRaises(KnowledgeBaseError):
            self.kb.add_entry("问题", "  ")

    def test_update_entry(self):
        self.kb.update_entry(0, answer="新答案")
        self.assertEqual(self.kb.entries[0].answer, "新答案")
        self.assertEqual(self.kb.entries[0].question, "如何进行职业规划?")

    def test_update_tags(self):
        self.kb.update_entry(0, tags=["新标签"])
        self.assertEqual(self.kb.entries[0].tags, ("新标签",))

    def test_remove_entry(self):
        removed = self.kb.remove_entry(0)
        self.assertEqual(removed.question, "如何进行职业规划?")
        self.assertEqual(len(self.kb), 2)

    def test_index_out_of_range(self):
        with self.assertRaises(KnowledgeBaseError):
            self.kb.remove_entry(99)

    def test_persisted_after_mutation(self):
        self.kb.add_entry("持久化测试", "答案")
        reloaded = KnowledgeBase(self.path)
        self.assertEqual(len(reloaded), 4)

    def test_all_tags_sorted(self):
        tags = self.kb.all_tags()
        self.assertEqual(tags, sorted(tags))
        self.assertIn("面试", tags)

    def test_tag_counts(self):
        counts = self.kb.tag_counts()
        self.assertEqual(sum(counts.values()), 6)
        self.assertEqual(counts["职业规划"], 1)

    def test_to_list(self):
        data = self.kb.to_list()
        self.assertEqual(len(data), 3)
        self.assertIn("问题", data[0])


class ReadFailureTests(unittest.TestCase):
    """回归测试：I/O 故障绝不能被当成「文件损坏」处理。

    旧实现把 OSError 与 JSONDecodeError 合并处理，一次瞬时读取失败
    （文件被占用、权限变化、网络盘掉线）就会把好文件改名备份、
    内存降级为空库，用户随后添加知识即造成数据丢失。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "kb.json"
        self.path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")

    def _failing_open(self):
        return mock.patch("builtins.open", side_effect=PermissionError("文件被占用"))

    def test_read_error_does_not_move_or_blank_file(self):
        kb = KnowledgeBase(self.path)
        with self._failing_open():
            kb.load()

        # 原文件必须原地保留，内容不变
        self.assertTrue(self.path.exists())
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))), 3)
        # 不得产生 .corrupt-* 备份
        self.assertEqual(list(self.dir.glob("*.corrupt-*")), [])
        self.assertTrue(kb.warnings)
        self.assertEqual(len(kb), 0)
        self.assertTrue(kb._read_failed)

    def test_save_refuses_to_overwrite_after_read_failure(self):
        kb = KnowledgeBase(self.path)
        with self._failing_open():
            kb.load()
        with self.assertRaises(KnowledgeBaseError):
            kb.save()
        # 原文件仍是 3 条，没有被空库覆盖
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))), 3)

    def test_add_entry_refuses_to_overwrite_after_read_failure(self):
        kb = KnowledgeBase(self.path)
        with self._failing_open():
            kb.load()
        with self.assertRaises(KnowledgeBaseError):
            kb.add_entry("新问题", "新答案")
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))), 3)
        # 内存也要回滚
        self.assertEqual(len(kb), 0)

    def test_save_to_other_path_still_allowed(self):
        kb = KnowledgeBase(self.path)
        with self._failing_open():
            kb.load()
        target = self.dir / "export.json"
        kb.save(target)
        self.assertTrue(target.exists())

    def test_read_error_strict_raises(self):
        kb = KnowledgeBase(self.path, strict=True)
        with self._failing_open():
            with self.assertRaises(KnowledgeBaseError):
                kb.load()

    def test_reload_recovers_after_transient_failure(self):
        kb = KnowledgeBase(self.path)
        with self._failing_open():
            kb.load()
        kb.reload()  # 故障解除
        self.assertEqual(len(kb), 3)
        self.assertFalse(kb._read_failed)
        # 恢复后可以正常写回
        kb.save()
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))), 3)


class WriteFailureTests(unittest.TestCase):
    """回归测试：落盘失败时的状态一致性。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "kb.json"
        self.path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
        self.kb = KnowledgeBase(self.path)

    def _failing_replace(self):
        return mock.patch(
            "career_qa.knowledge_base.os.replace", side_effect=OSError("磁盘已满")
        )

    def test_failed_save_cleans_temp_file(self):
        with self._failing_replace():
            with self.assertRaises(KnowledgeBaseError):
                self.kb.save()
        leftovers = [p.name for p in self.dir.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))), 3)

    def test_add_entry_rolls_back_on_write_failure(self):
        with self._failing_replace():
            with self.assertRaises(KnowledgeBaseError):
                self.kb.add_entry("新问题", "新答案")

        # 内存必须回滚，否则用户重试同一条会被误判为「重复条目」
        self.assertEqual(len(self.kb), 3)
        self.kb.add_entry("新问题", "新答案")
        self.assertEqual(len(self.kb), 4)
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))), 4)

    def test_update_entry_rolls_back_on_write_failure(self):
        original = self.kb.entries[0].answer
        with self._failing_replace():
            with self.assertRaises(KnowledgeBaseError):
                self.kb.update_entry(0, answer="新答案")
        self.assertEqual(self.kb.entries[0].answer, original)

    def test_remove_entry_rolls_back_on_write_failure(self):
        with self._failing_replace():
            with self.assertRaises(KnowledgeBaseError):
                self.kb.remove_entry(0)
        self.assertEqual(len(self.kb), 3)
        self.assertEqual(self.kb.entries[0].question, "如何进行职业规划?")


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "kb.json"
        self.path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")

    def test_second_backup_in_same_second_does_not_overwrite_first(self):
        kb = KnowledgeBase(self.path)
        first = kb._backup_corrupt_file()
        self.path.write_text("坏文件", encoding="utf-8")
        second = kb._backup_corrupt_file()

        self.assertNotEqual(first, second)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_corrupt_backup_keeps_original_bytes(self):
        kb = KnowledgeBase(self.path)
        original = self.path.read_bytes()
        self.path.write_text("{ 坏", encoding="utf-8")
        broken = self.path.read_bytes()
        backup = kb._backup_corrupt_file()
        self.assertEqual(backup.read_bytes(), broken)
        self.assertNotEqual(backup.read_bytes(), original)


class UpdateValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "kb.json"
        self.path.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
        self.kb = KnowledgeBase(self.path)

    def test_update_to_existing_question_rejected(self):
        with self.assertRaises(DuplicateEntryError):
            self.kb.update_entry(0, question="如何准备面试?")

    def test_update_keeping_own_question_allowed(self):
        entry = self.kb.update_entry(0, answer="改过的答案")
        self.assertEqual(entry.answer, "改过的答案")
        self.assertEqual(entry.question, "如何进行职业规划?")

    def test_bool_index_rejected(self):
        # bool 是 int 子类，remove_entry(True) 不应删掉第 2 条
        with self.assertRaises(KnowledgeBaseError):
            self.kb.remove_entry(True)
        self.assertEqual(len(self.kb), 3)


if __name__ == "__main__":
    unittest.main()
