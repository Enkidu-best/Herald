"""Проверки llm_config без моделей: 4 уровня приоритета, файла нет, файл битый, поиск вверх.

    python3 -m unittest discover -s "_общие модули/llm_config/tests"     (или pytest)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "llm_config.py"
_spec = importlib.util.spec_from_file_location("llm_config_under_test", _SRC)
lc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lc
_spec.loader.exec_module(lc)


class LlmConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.proj = self.root / "Проект" / "common" / "llm_config"      # как копия в проекте
        self.proj.mkdir(parents=True)
        env = {k: v for k, v in os.environ.items() if k not in (lc.FILE_ENV, "T_MODEL", "T_MODEL2", "T_URL")}
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()
        lc._warned.clear()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def write(self, text: str, where: Path | None = None) -> Path:
        f = (where or self.root) / lc.FILE_NAME
        f.write_text(text, encoding="utf-8")
        return f

    # 4 уровня приоритета
    def test_level4_default_without_file(self):
        self.assertIsNone(lc.find_file(self.proj))
        self.assertEqual(lc.resolve("main", start=self.proj), ("qwen3.5:9b", "default"))
        self.assertEqual(lc.model("fallback", start=self.proj), "qwen3.5:4b")
        self.assertEqual(lc.model("embed", start=self.proj), "bge-m3")
        self.assertEqual(lc.ollama_url(start=self.proj), "http://127.0.0.1:11434")
        self.assertEqual(lc.model("main", default="своя:1b", start=self.proj), "своя:1b")

    def test_level3_file_found_upwards(self):
        f = self.write(json.dumps({"main": "m-file", "fallback": "f-file", "ollama_url": "http://h:1/"}))
        self.assertEqual(lc.find_file(self.proj), f)
        self.assertEqual(lc.resolve("main", start=self.proj), ("m-file", f"file {f}"))
        self.assertEqual(lc.model("fallback", start=self.proj), "f-file")
        self.assertEqual(lc.model("embed", start=self.proj), "bge-m3")          # роли нет в файле - встроенное
        self.assertEqual(lc.ollama_url(start=self.proj), "http://h:1")          # без «/» в конце

    def test_nearest_file_wins(self):
        self.write(json.dumps({"main": "far"}))
        self.write(json.dumps({"main": "near"}), self.proj.parent.parent)
        self.assertEqual(lc.model(start=self.proj), "near")

    def test_level2_user_over_file(self):
        self.write(json.dumps({"main": "m-file"}))
        self.assertEqual(lc.resolve("main", user="m-user", start=self.proj), ("m-user", "user"))
        self.assertEqual(lc.model("main", user="  ", start=self.proj), "m-file")    # пустой выбор - не выбор
        self.assertEqual(lc.model("main", user=None, start=self.proj), "m-file")

    def test_level1_env_over_all(self):
        self.write(json.dumps({"main": "m-file", "ollama_url": "http://file:1"}))
        os.environ["T_MODEL2"] = "m-env2"
        self.assertEqual(lc.resolve("main", env="T_MODEL2", user="m-user", start=self.proj), ("m-env2", "env T_MODEL2"))
        self.assertEqual(lc.model("main", env=("T_MODEL", "T_MODEL2"), start=self.proj), "m-env2")   # первая непустая
        os.environ["T_MODEL"] = ""                                                          # пустая - как нет
        self.assertEqual(lc.model("main", env="T_MODEL", user="m-user", start=self.proj), "m-user")
        os.environ["T_URL"] = "http://env:2/"
        self.assertEqual(lc.ollama_url(env="T_URL", start=self.proj), "http://env:2")

    # битый файл
    def test_broken_file(self):
        for text in ("{не json", "[1, 2]", '"строка"', "", json.dumps({"main": 5, "fallback": ""})):
            with self.subTest(text=text):
                lc._warned.clear()
                self.write(text)
                with self.assertLogs(lc.log, "WARNING") if text in ("{не json", "[1, 2]", '"строка"', "") else _nolog():
                    self.assertEqual(lc.resolve("main", start=self.proj), ("qwen3.5:9b", "default"))
                self.assertEqual(lc.model("fallback", start=self.proj), "qwen3.5:4b")

    def test_unknown_role(self):
        with self.assertRaises(KeyError):
            lc.model("vision", start=self.proj)
        self.write(json.dumps({"vision": "qwen3-vl"}))
        self.assertEqual(lc.model("vision", start=self.proj), "qwen3-vl")

    # явный путь
    def test_file_env(self):
        other = self.root / "другое"
        other.mkdir()
        f = self.write(json.dumps({"main": "m-explicit"}), other)
        self.write(json.dumps({"main": "m-walk"}))
        os.environ[lc.FILE_ENV] = str(f)
        self.assertEqual(lc.model(start=self.proj), "m-explicit")
        os.environ[lc.FILE_ENV] = ""                                    # пусто - файл не читать
        self.assertEqual(lc.model(start=self.proj), "qwen3.5:9b")
        os.environ[lc.FILE_ENV] = str(self.root / "нет.json")          # путь без файла - встроенное
        self.assertEqual(lc.model(start=self.proj), "qwen3.5:9b")

    def test_change_in_file_applies_without_restart(self):
        self.write(json.dumps({"main": "a"}))
        self.assertEqual(lc.model(start=self.proj), "a")
        self.write(json.dumps({"main": "b"}))
        self.assertEqual(lc.model(start=self.proj), "b")


class _nolog:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    unittest.main()
