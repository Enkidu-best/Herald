"""F5 на torch - для Windows и Linux: видеокарта NVIDIA (CUDA) или процессор.

Тот же голос и тот же наш алгоритм, что у MLX-движка на Mac: та же модель
(Misha24-10 accent_tune), расписание шагов EPSS, якорь тишины в конце окна и
хвост куска 3.1, длительность по байтам, поправка темпа под образец. Отличается
только то, ЧЕМ считается сеть: здесь torch. Поэтому класс - наследник
F5MLXEngine: вся подготовка (образец, токены, длительность, нарезка, хвост)
общая, переопределены только загрузка и решение ОДУ.

Сеть и мел-спектр - копия из F5-TTS (core/engines/f5net, MIT): пакет f5-tts
целиком тянет gradio, datasets, wandb, и в установщик они не нужны.

Устройство: CUDA, если есть видеокарта NVIDIA и torch с её поддержкой; иначе
процессор. MPS на Mac не используется: там есть MLX, а F5 на MPS переполняет
память (проверено в 2.x).
"""

from __future__ import annotations

import os

import numpy as np

from . import f5_mlx as _m
from .f5_mlx import (F5MLXEngine, LANG_MODELS, SAMPLE_RATE, HOP_LENGTH, CFG_STRENGTH,
                     TAIL_ANCHOR_SEC, TARGET_RMS, preprocess_ref_audio,
                     _ref_text_for_model, convert_char_to_tokens, timesteps)

VOCOS_REPO = "charactr/vocos-mel-24khz"
# архитектура F5TTS_v1_Base (configs/F5TTS_v1_Base.yaml в F5-TTS)
DIT_ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                text_mask_padding=True, qk_norm=None, conv_layers=4, pe_attn_head=None,
                attn_backend="torch", attn_mask_enabled=False, mel_dim=100)


def torch_device() -> str:
    """cuda - если видеокарта NVIDIA и torch с CUDA, иначе cpu."""
    want = os.environ.get("HERALD_DEVICE")
    if want:
        return want
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass                      # нет torch с CUDA - считаем на процессоре
    return "cpu"


class F5TorchEngine(F5MLXEngine):
    name = "f5torch"

    def __init__(self, *a, device: str | None = None, **kw) -> None:
        kw.pop("quant_bits", None)            # сжатие весов - только для MLX
        kw.pop("dtype", None)
        super().__init__(*a, **kw)
        self.device = device or torch_device()
        self.dtype = "float32"

    # --- модель -----------------------------------------------------------
    def _weights_and_vocab(self) -> tuple[str, str]:
        from huggingface_hub import hf_hub_download
        # только для проб (11б, шаг 3б): папка с дообученными весами model_v1.safetensors + vocab.txt
        own = os.environ.get("HERALD_F5_WEIGHTS")
        if own:
            return os.path.join(own, "model_v1.safetensors"), os.path.join(own, "vocab.txt")
        if self.variant:
            mdir = self._prepare_variant_dir()
            return (os.path.join(mdir, "model_v1.safetensors"),
                    os.path.join(mdir, "vocab.txt"))
        cfg = LANG_MODELS[self.language]
        return hf_hub_download(cfg["repo"], cfg["ckpt"]), hf_hub_download(cfg["repo"], cfg["vocab"])

    def _load_in_thread(self) -> None:
        import torch
        from safetensors.torch import load_file
        from .f5net.dit import DiT
        ckpt, vocab_path = self._weights_and_vocab()
        sd = load_file(ckpt)
        pre = "ema_model.transformer."
        tsd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        if not tsd:                            # веса без префикса EMA
            tsd = {k[len("transformer."):]: v for k, v in sd.items()
                   if k.startswith("transformer.")}
        n_text = tsd["text_embed.text_embed.weight"].shape[0] - 1
        net = DiT(**DIT_ARCH, text_num_embeds=n_text)
        missing, unexpected = net.load_state_dict(tsd, strict=False)
        if missing:
            raise RuntimeError(f"в весах F5 не хватает слоёв: {missing[:5]}")
        self._net = net.to(self.device).eval()
        import torchaudio
        # мел - ровно как у MLX-движка на Mac (f5_tts_mlx.audio.log_mel_spectrogram):
        # края дополняются нулями, последний кадр отбрасывается. Исходный F5 на
        # torch дополняет отражением и кадр оставляет - сеть та же, но из-за краёв
        # образца звук выходил бы другим (поймано test_torch_parity)
        self._fb = torchaudio.functional.melscale_fbanks(
            1024 // 2 + 1, 0.0, SAMPLE_RATE / 2, 100, SAMPLE_RATE, norm=None,
            mel_scale="htk").to(self.device)
        self._win = torch.hann_window(1024, periodic=True, device=self.device)
        with open(vocab_path, encoding="utf-8") as f:
            chars = [line[:-1] if line.endswith("\n") else line for line in f]
        self._vocab = {c: i for i, c in enumerate(chars)}
        from vocos import Vocos
        self._vocoder = Vocos.from_pretrained(VOCOS_REPO).to(self.device).eval()
        self._torch = torch
        self._loaded = True

    def _mel(self, wav):
        """(1, T) -> (1, 100, кадры): лог-мел как у MLX-движка."""
        torch = self._torch
        x = torch.nn.functional.pad(wav, (512, 512))
        spec = torch.stft(x, 1024, hop_length=HOP_LENGTH, win_length=1024, window=self._win,
                          center=False, return_complex=True).abs()      # (1, 513, кадры)
        mel = torch.matmul(self._fb.T, spec[..., :-1])
        return mel.clamp(min=1e-5).log()

    def _make_ref(self, vid: str, wav: str, ref_text: str):
        self._check_ref_text(vid, ref_text)
        audio = preprocess_ref_audio(wav)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        gain = 1.0
        if rms < TARGET_RMS:
            gain = TARGET_RMS / rms
            audio = audio * gain
        rt = _ref_text_for_model(ref_text)
        self._audio_cache[vid] = (np.asarray(audio, dtype=np.float32), rt, 1.0 / gain,
                                  self._pace_for(audio, rt))
        return self._audio_cache[vid]

    # --- синтез -----------------------------------------------------------
    def _synth_in_thread(self, audio, ref_frames: int, ref_text: str,
                         gen_texts: list[str], speed_muls: list[float],
                         headroom_muls: list[float]) -> list[tuple[np.ndarray, bool]]:
        torch = self._torch
        tail = int(TAIL_ANCHOR_SEC * SAMPLE_RATE / HOP_LENGTH)
        out = []
        with torch.inference_mode():
            for g, sm, hm in zip(gen_texts, speed_muls, headroom_muls):
                dur = self._gen_frames(ref_frames, ref_text, g, sm, hm)
                tokens = convert_char_to_tokens([ref_text + g])[0]
                mel, n = self._sample_mel_torch(audio, tokens, dur, tail)
                m = mel[:, ref_frames:n, :]
                wave = self._vocoder.decode(m.permute(0, 2, 1).float())
                # через модуль, а не по имени: тест окончаний подменяет _finish_tail
                out.append(_m._finish_tail(wave.detach().cpu().numpy().astype(np.float32).reshape(-1)))
        return out

    def _sample_mel_torch(self, audio, tokens: list[str], duration: int, tail: int,
                          noise=None):
        """Наш Эйлер с CFG и якорем тишины - как F5MLXEngine._sample_mel.

        noise - начальный шум (для сверки с MLX тем же шумом, тест).
        Без inference_mode torch копил бы граф для обучения на всех шагах -
        десятки гигабайт (процесс убивался по памяти)."""
        with self._torch.inference_mode():
            return self._sample_mel_inner(audio, tokens, duration, tail, noise)

    def _sample_mel_inner(self, audio, tokens, duration, tail, noise):
        torch = self._torch
        dev = self.device
        wav = torch.as_tensor(np.asarray(audio, dtype=np.float32), device=dev)[None, :]
        cond = self._mel(wav).permute(0, 2, 1)                 # (1, n_ref, 100)
        ref_len = cond.shape[1]
        gen_end = max(ref_len + 1, int(duration), len(tokens))
        n = gen_end + tail
        step_cond = torch.zeros(1, n, 100, device=dev)
        step_cond[:, :ref_len] = cond
        if tail:
            sil = self._mel(torch.zeros(1, (tail + 2) * HOP_LENGTH, device=dev)).permute(0, 2, 1)
            step_cond[:, gen_end:n] = sil[:, :tail]
        text = torch.tensor([[self._vocab.get(c, 0) for c in tokens]], device=dev)
        if noise is None:
            x = torch.randn(1, n, 100, device=dev)
        else:
            x = torch.as_tensor(np.asarray(noise, dtype=np.float32), device=dev)[None, :n]
        t = torch.as_tensor(timesteps(self.steps, self.schedule), device=dev)
        use_half = dev == "cuda"
        n_steps = t.shape[0] - 1
        # подсказка CFG только на первых cfg_steps шагах (3.6) - как в MLX
        k_cfg = n_steps if self.cfg_steps is None else max(0, min(n_steps, int(self.cfg_steps)))
        for i in range(n_steps):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_half):
                if i < k_cfg:
                    v = self._net(x=x, cond=step_cond, text=text, time=t[i], mask=None,
                                  cfg_infer=True, cache=True)
                    pred, null = v.float().chunk(2, dim=0)
                    d = pred + (pred - null) * CFG_STRENGTH
                else:
                    d = self._net(x=x, cond=step_cond, text=text, time=t[i], mask=None,
                                  drop_audio_cond=False, drop_text=False, cache=True).float()
            x = x + (t[i + 1] - t[i]) * d
        self._net.clear_cache()
        if tail:
            x[:, gen_end:n] = step_cond[:, gen_end:n]
        return x, n


__all__ = ["F5TorchEngine", "torch_device"]
