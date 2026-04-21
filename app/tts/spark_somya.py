from __future__ import annotations

import logging
from pathlib import Path
import warnings

import numpy as np

from app.types import SynthesizedAudio

logger = logging.getLogger(__name__)


class SparkSomyaTTSBackend:
    backend_name = "spark_somya_tts"

    def __init__(
        self,
        *,
        model_dir: Path,
        repo_id: str = "somyalab/Spark_somya_TTS",
        device: str = "cpu",
        audio_prompt_path: Path | None = None,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        self.model_dir = Path(model_dir)
        self.repo_id = repo_id
        self.device = device
        self.audio_prompt_path = audio_prompt_path
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._model = None
        self._tokenizer = None
        self._audio_tokenizer = None
        self._torch = None
        self._device = None
        self._sample_rate = 16000
        self._global_token_ids = None
        self._audio_prompt_mtime: float | None = None

    def warmup(self):
        if self._model is not None:
            self._refresh_reference_tokens()
            return

        torch = self._import_torch()
        self._torch = torch
        self._device = self._resolve_device(torch)
        model_dir = self._ensure_model_dir()

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as transformers_logging
        from sparktts.models.audio_tokenizer import BiCodecTokenizer

        logger.info("Loading Spark Somya TTS model from %s", model_dir)
        previous_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                trust_remote_code=False,
                fix_mistral_regex=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                trust_remote_code=False,
                dtype=self._resolve_dtype(torch),
            )
        finally:
            transformers_logging.set_verbosity(previous_verbosity)
        model.to(self._device)
        model.eval()
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`torch.nn.utils.weight_norm` is deprecated",
                category=FutureWarning,
            )
            audio_tokenizer = BiCodecTokenizer(model_dir, device=self._device)
        self._tokenizer = tokenizer
        self._model = model
        self._audio_tokenizer = audio_tokenizer
        self._refresh_reference_tokens()
        logger.info("Spark Somya TTS ready (%s)", self.describe())

    def synthesize(self, text: str) -> SynthesizedAudio:
        self.warmup()
        text = text.strip()
        if not text:
            return SynthesizedAudio(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                duration_ms=0.0,
            )

        if self._global_token_ids is None:
            logger.error("Spark Somya TTS requires a reference audio prompt at %s", self.audio_prompt_path)
            return SynthesizedAudio(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                duration_ms=0.0,
            )

        prompt = self._build_prompt(text)
        inputs = self._tokenizer([prompt], return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        try:
            from transformers.utils import logging as transformers_logging

            previous_verbosity = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
            try:
                with self._torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        max_length=None,
                        do_sample=True,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        pad_token_id=self._tokenizer.pad_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                    )
            finally:
                transformers_logging.set_verbosity(previous_verbosity)
            generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
            semantic_ids = self._extract_semantic_ids(generated_ids[0].tolist())
            if semantic_ids.size == 0:
                raise RuntimeError("Model generation did not contain any semantic audio tokens.")
            wav = self._audio_tokenizer.detokenize(
                self._global_token_ids.to(self._device).squeeze(0),
                self._torch.from_numpy(semantic_ids).to(self._device),
            )
            samples = np.asarray(wav, dtype=np.float32).reshape(-1)
        except Exception as exc:
            logger.error("Spark Somya TTS synthesis failed: %s", exc)
            return SynthesizedAudio(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                duration_ms=0.0,
            )

        duration_ms = (samples.size / self._sample_rate) * 1000.0
        return SynthesizedAudio(
            samples=samples,
            sample_rate=self._sample_rate,
            duration_ms=duration_ms,
        )

    def describe(self) -> str:
        return f"{self.backend_name}/{self.model_dir.name}"

    def _build_prompt(self, text: str) -> str:
        global_tokens = "".join(
            f"<|bicodec_global_{int(token_id)}|>"
            for token_id in self._global_token_ids.squeeze().tolist()
        )
        return "".join(
            [
                "<|task_tts|>",
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
            ]
        )

    def _extract_semantic_ids(self, token_ids: list[int]) -> np.ndarray:
        semantic_ids: list[int] = []
        for token in self._tokenizer.convert_ids_to_tokens(token_ids):
            if token.startswith("<|bicodec_semantic_") and token.endswith("|>"):
                semantic_ids.append(int(token[19:-2]))
        return np.asarray(semantic_ids, dtype=np.int64)[None, :]

    def _ensure_model_dir(self) -> Path:
        config_path = self.model_dir / "config.json"
        weights_path = self.model_dir / "model.safetensors"
        if config_path.exists() and weights_path.exists():
            return self.model_dir

        from huggingface_hub import snapshot_download

        logger.info("Downloading Spark Somya TTS model: %s", self.repo_id)
        snapshot_download(
            repo_id=self.repo_id,
            local_dir=str(self.model_dir),
        )
        return self.model_dir

    def _resolve_device(self, torch):
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self, torch):
        if self._device is not None and self._device.type == "cuda":
            return torch.bfloat16
        return torch.float32

    def _refresh_reference_tokens(self):
        if not self.audio_prompt_path or not self.audio_prompt_path.exists():
            self._global_token_ids = None
            self._audio_prompt_mtime = None
            return

        current_mtime = self.audio_prompt_path.stat().st_mtime
        if self._global_token_ids is not None and current_mtime == self._audio_prompt_mtime:
            return

        logger.info("Encoding Spark Somya voice prompt from %s", self.audio_prompt_path)
        global_token_ids, _ = self._audio_tokenizer.tokenize(str(self.audio_prompt_path))
        self._global_token_ids = global_token_ids
        self._audio_prompt_mtime = current_mtime

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Spark Somya TTS requires torch, transformers, torchaudio, soundfile, and the vendored sparktts runtime."
            ) from exc
        return torch
