"""Одна настройка модели Ollama для всех программ - общий модуль (`_общие модули/llm_config`, описание - README.md)."""
from .llm_config import DEFAULTS, DEFAULT_URL, FILE_ENV, FILE_NAME, find_file, load, model, ollama_url, resolve

__all__ = ["DEFAULTS", "DEFAULT_URL", "FILE_ENV", "FILE_NAME", "find_file", "load", "model", "ollama_url", "resolve"]
