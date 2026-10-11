"""配置加载的单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from career_qa.config import (
    DEFAULT_KB_PATH,
    PROJECT_ROOT,
    AppConfig,
    ConfigError,
    load_config,
    load_config_file,
    load_env,
)


class DefaultsTests(unittest.TestCase):
    def test_default_values(self):
        config, _ = load_config(config_path="__missing__.json", argv=[], environ={})
        self.assertEqual(config.model, "llama2")
        self.assertEqual(config.ollama_base_url, "http://localhost:11434")
        self.assertEqual(config.top_n, 3)
        self.assertAlmostEqual(config.min_score, 0.25)
        self.assertTrue(config.stream)
        self.assertEqual(config.max_history_turns, 3)
        self.assertEqual(config.kb_path, DEFAULT_KB_PATH)

    def test_base_url_trailing_slash_stripped(self):
        config = AppConfig(ollama_base_url="http://localhost:11434/")
        self.assertEqual(config.ollama_base_url, "http://localhost:11434")

    def test_kb_path_is_path_object(self):
        config = AppConfig(kb_path="knowledge_base.json")
        self.assertIsInstance(config.kb_path, Path)

    def test_missing_config_file_is_fine(self):
        self.assertEqual(load_config_file("__definitely_missing__.json"), {})

    def test_require_file_raises(self):
        with self.assertRaises(ConfigError):
            load_config(config_path="__missing__.json", argv=[], environ={}, require_file=True)


class ValidationTests(unittest.TestCase):
    def test_invalid_base_url(self):
        with self.assertRaises(ConfigError):
            AppConfig(ollama_base_url="localhost:11434")

    def test_empty_model(self):
        with self.assertRaises(ConfigError):
            AppConfig(model="  ")

    def test_min_score_out_of_range(self):
        with self.assertRaises(ConfigError):
            AppConfig(min_score=1.5)
        with self.assertRaises(ConfigError):
            AppConfig(min_score=-0.1)

    def test_temperature_out_of_range(self):
        with self.assertRaises(ConfigError):
            AppConfig(temperature=3.0)

    def test_negative_top_n(self):
        with self.assertRaises(ConfigError):
            AppConfig(top_n=0)

    def test_negative_timeouts(self):
        with self.assertRaises(ConfigError):
            AppConfig(connect_timeout=0)
        with self.assertRaises(ConfigError):
            AppConfig(request_timeout=-1)

    def test_negative_retries(self):
        with self.assertRaises(ConfigError):
            AppConfig(max_retries=-1)

    def test_unknown_override_key(self):
        with self.assertRaises(ConfigError):
            AppConfig().with_overrides(nonexistent=1)

    def test_immutable(self):
        config = AppConfig()
        with self.assertRaises(Exception):
            config.model = "other"  # type: ignore[misc]


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write_config(self, payload) -> Path:
        path = self.dir / "config.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_file_overrides_defaults(self):
        path = self.write_config({"model": "qwen2", "top_n": 7, "stream": False})
        config, _ = load_config(config_path=path, argv=[], environ={})
        self.assertEqual(config.model, "qwen2")
        self.assertEqual(config.top_n, 7)
        self.assertFalse(config.stream)

    def test_numeric_strings_coerced(self):
        path = self.write_config({"top_n": "5", "min_score": "0.4", "stream": "no"})
        config, _ = load_config(config_path=path, argv=[], environ={})
        self.assertEqual(config.top_n, 5)
        self.assertAlmostEqual(config.min_score, 0.4)
        self.assertFalse(config.stream)

    def test_relative_kb_path_resolved_to_project_root(self):
        path = self.write_config({"kb_path": "data/kb.json"})
        config, _ = load_config(config_path=path, argv=[], environ={})
        self.assertTrue(config.kb_path.is_absolute())
        self.assertEqual(config.kb_path, (PROJECT_ROOT / "data" / "kb.json").resolve())

    def test_absolute_kb_path_kept(self):
        absolute = self.dir / "kb.json"
        path = self.write_config({"kb_path": str(absolute)})
        config, _ = load_config(config_path=path, argv=[], environ={})
        self.assertEqual(config.kb_path, absolute)

    def test_unknown_key_rejected(self):
        path = self.write_config({"unknown_option": 1})
        with self.assertRaises(ConfigError):
            load_config(config_path=path, argv=[], environ={})

    def test_broken_json_rejected(self):
        path = self.dir / "bad.json"
        path.write_text("{ 坏文件", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(config_path=path, argv=[], environ={})

    def test_non_object_rejected(self):
        path = self.dir / "list.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(config_path=path, argv=[], environ={})

    def test_invalid_type_rejected(self):
        path = self.write_config({"top_n": "很多"})
        with self.assertRaises(ConfigError):
            load_config(config_path=path, argv=[], environ={})

    def test_invalid_bool_rejected(self):
        path = self.write_config({"stream": "也许"})
        with self.assertRaises(ConfigError):
            load_config(config_path=path, argv=[], environ={})


class EnvTests(unittest.TestCase):
    def test_env_overrides_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "config.json"
        path.write_text(json.dumps({"model": "from-file"}), encoding="utf-8")

        config, _ = load_config(
            config_path=path,
            argv=[],
            environ={"CAREERQA_MODEL": "from-env", "CAREERQA_TOP_N": "9"},
        )
        self.assertEqual(config.model, "from-env")
        self.assertEqual(config.top_n, 9)

    def test_empty_env_ignored(self):
        config, _ = load_config(
            config_path="__missing__.json", argv=[], environ={"CAREERQA_MODEL": ""}
        )
        self.assertEqual(config.model, "llama2")

    def test_env_helpers(self):
        data = load_env({"CAREERQA_STREAM": "0", "CAREERQA_MIN_SCORE": "0.5"})
        self.assertEqual(data, {"stream": False, "min_score": 0.5})

    def test_env_invalid_bool(self):
        with self.assertRaises(ConfigError):
            load_env({"CAREERQA_STREAM": "maybe"})


class CliTests(unittest.TestCase):
    def test_cli_overrides_env_and_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "config.json"
        path.write_text(json.dumps({"model": "from-file"}), encoding="utf-8")

        config, args = load_config(
            config_path=path,
            argv=["--model", "from-cli", "--top-n", "4"],
            environ={"CAREERQA_MODEL": "from-env"},
        )
        self.assertEqual(config.model, "from-cli")
        self.assertEqual(config.top_n, 4)
        self.assertFalse(args.check)

    def test_no_stream_flag(self):
        config, _ = load_config(
            config_path="__missing__.json", argv=["--no-stream"], environ={}
        )
        self.assertFalse(config.stream)

    def test_flags_parsed(self):
        _, args = load_config(
            config_path="__missing__.json",
            argv=["--check", "--list-models"],
            environ={},
        )
        self.assertTrue(args.check)
        self.assertTrue(args.list_models)

    def test_kb_flag(self):
        config, _ = load_config(
            config_path="__missing__.json", argv=["--kb", "custom.json"], environ={}
        )
        self.assertEqual(config.kb_path.name, "custom.json")
        self.assertTrue(config.kb_path.is_absolute())


class SerializationTests(unittest.TestCase):
    def test_to_dict_uses_string_path(self):
        config = AppConfig(kb_path="knowledge_base.json")
        self.assertIsInstance(config.to_dict()["kb_path"], str)

    def test_save_and_reload(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        target = Path(tmp.name) / "nested" / "config.json"

        original = AppConfig(model="qwen2", top_n=5, stream=False)
        original.save(target)
        self.assertTrue(target.exists())

        reloaded, _ = load_config(config_path=target, argv=[], environ={})
        self.assertEqual(reloaded.model, "qwen2")
        self.assertEqual(reloaded.top_n, 5)
        self.assertFalse(reloaded.stream)

    def test_example_config_is_valid(self):
        """config.example.json 必须能被正常加载。"""
        example = PROJECT_ROOT / "config.example.json"
        if not example.exists():
            self.skipTest("示例配置不存在")
        config, _ = load_config(config_path=example, argv=[], environ={})
        self.assertEqual(config.model, "llama2")


if __name__ == "__main__":
    unittest.main()
