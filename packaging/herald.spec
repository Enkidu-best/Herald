# -*- mode: python ; coding: utf-8 -*-
# Сборка Herald для Windows (по образцу Scribe 2.4): папка с Herald.exe (окно) и
# herald-cli.exe (консоль, для проверок). ffmpeg кладётся рядом шагом CI.
# Модели в сборку НЕ входят: качаются при первом запуске (Hugging Face) или
# берутся из офлайн-пакета. Образцов голосов в сборке нет и быть не должно.
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
datas, binaries, hidden = [], [], []
for pkg in ("customtkinter", "tkinterdnd2", "faster_whisper", "ctranslate2", "onnxruntime",
            "_sounddevice_data", "docx", "ebooklib", "fitz", "pymupdf", "tokenizers", "av",
            "ruaccent", "silero_stress", "ru_normalizr", "pymorphy3", "pymorphy3_dicts_ru",
            "eng_to_ipa", "num2words", "razdel", "vocos", "encodec", "x_transformers", "einx",
            "torchaudio", "rjieba", "pypinyin", "transformers", "sentencepiece", "pycrfsuite",
            "vosk_tts"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hidden += h
    except Exception as e:
        print("пропущено:", pkg, e)
for dist in ("torch", "torchaudio", "transformers", "tokenizers", "huggingface-hub", "safetensors",
             "faster-whisper", "ctranslate2", "customtkinter", "numpy", "onnxruntime", "av",
             "python-docx", "ruaccent", "silero-stress", "ru-normalizr", "vocos", "x-transformers",
             "einops", "einx", "regex", "requests", "tqdm", "filelock", "packaging", "pyyaml"):
    try:
        datas += copy_metadata(dist)
    except Exception as e:
        print("без метаданных:", dist, e)
datas += [(os.path.join(ROOT, "assets"), "assets"),
          (os.path.join(ROOT, "core", "engines", "f5net", "LICENSE"), os.path.join("core", "engines", "f5net")),
          # справочник ударений (Викисловарь, CC BY-SA) - 11а
          (os.path.join(ROOT, "core", "data"), os.path.join("core", "data"))]
hidden += ["theme", "player", "core", "core.osutil", "core.asr", "core.align", "core.stream", "core.speak_server", "core.voice_store", "core.engines.fast", "core.cuda_pack",
           "core.engines.f5_torch", "core.engines.f5net.dit", "core.engines.f5net.modules",
           "sounddevice", "scipy.signal"]
excludes = ["piper", "mlx", "mlx_whisper", "f5_tts", "f5_tts_mlx", "vocos_mlx", "AVFoundation", "Foundation",
            "objc", "torchcodec", "gradio", "wandb", "datasets", "IPython", "jupyter", "notebook",
            "pytest", "matplotlib", "tensorboard"]

common = dict(pathex=[ROOT], binaries=binaries, datas=datas, hiddenimports=hidden, excludes=excludes,
              hookspath=[], runtime_hooks=[], noarchive=False,
              # torch лежит обычными файлами, а не в архиве: тогда докачанная сборка
              # с CUDA (core/cuda_pack.py) встаёт в путь поиска раньше и заменяет его
              module_collection_mode={"torch": "py"})
a_gui = Analysis([os.path.join(ROOT, "tts_gui.py")], **common)
a_cli = Analysis([os.path.join(ROOT, "cli.py")], **common)
pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)
icon = os.path.join(SPECPATH, "herald.ico")
# Режим UTF-8 у самого Python: без него в сборке для Windows open() без явной
# кодировки писал cp1252, и русская расшифровка образца падала (поймано на раннере)
utf8 = [("X utf8", None, "OPTION")]
exe_gui = EXE(pyz_gui, a_gui.scripts, utf8, exclude_binaries=True, name="Herald", console=False,
              icon=icon, upx=False)
exe_cli = EXE(pyz_cli, a_cli.scripts, utf8, exclude_binaries=True, name="herald-cli", console=True,
              icon=icon, upx=False)
coll = COLLECT(exe_gui, exe_cli, a_gui.binaries, a_gui.datas, a_cli.binaries, a_cli.datas,
               strip=False, upx=False, name="Herald")
