from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re
import unicodedata

import numpy as np


logger = logging.getLogger(__name__)
DEFAULT_RUNTIME_CONFIG = "runtime_config.json"
DEFAULT_TOKEN_ID_OFFSET = 4
DEFAULT_TOKENIZER_SPECIAL_IDS = {
    "blank": 0,
    "sos": 1,
    "eos": 2,
    "pad": 3,
}
_WS_RE = re.compile(r"\s+")


@dataclass
class DecodeResult:
    token_ids: list[int]
    score: float = 0.0
    confidence: float = 0.0


def _clean_control_chars(text: str) -> str:
    out_chars: list[str] = []
    for ch in text:
        if ch in {"\t", "\n", "\r"}:
            out_chars.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat in {"Cf", "Cc"}:
            continue
        out_chars.append(ch)
    return "".join(out_chars)


def normalize_text(text: str) -> str:
    out = unicodedata.normalize("NFC", text)
    out = _clean_control_chars(out)
    return _WS_RE.sub(" ", out).strip()


def decode_token_ids(
    token_ids: list[int],
    sp,
    *,
    token_id_offset: int,
    special_ids: dict[str, int],
) -> str:
    special_values = {int(value) for value in special_ids.values()}
    sp_ids = [
        int(token_id) - int(token_id_offset)
        for token_id in token_ids
        if int(token_id) >= int(token_id_offset) and int(token_id) not in special_values
    ]
    if not sp_ids:
        return ""
    return str(sp.decode(sp_ids))


def _static_onnx_dim(session, *, input_index: int, axis: int, label: str) -> int:
    shape = session.get_inputs()[input_index].shape
    if axis >= len(shape):
        raise ValueError(f"ONNX input {label!r} is missing axis {axis}.")
    value = shape[axis]
    if not isinstance(value, int):
        raise ValueError(f"ONNX input {label!r} axis {axis} must be static, got {value!r}.")
    return int(value)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - logsumexp


class OnnxRNNTGreedyDecoder:
    def __init__(
        self,
        predictor_session,
        joint_session,
        *,
        blank_id: int,
        max_symbols_per_step: int = 100,
    ):
        self.predictor_session = predictor_session
        self.joint_session = joint_session
        self.blank_id = int(blank_id)
        self.max_symbols_per_step = int(max_symbols_per_step)
        if self.max_symbols_per_step <= 0:
            raise ValueError("max_symbols_per_step must be > 0.")
        self.predictor_input_names = [item.name for item in predictor_session.get_inputs()]
        self.joint_input_names = [item.name for item in joint_session.get_inputs()]
        self.num_layers = _static_onnx_dim(
            predictor_session,
            input_index=1,
            axis=0,
            label="state_h",
        )
        self.hidden_dim = _static_onnx_dim(
            predictor_session,
            input_index=1,
            axis=2,
            label="state_h",
        )

    def _predict_step(
        self,
        token: np.ndarray,
        state_h: np.ndarray,
        state_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_out, next_state_h, next_state_c = self.predictor_session.run(
            None,
            {
                self.predictor_input_names[0]: token.astype(np.int64, copy=False),
                self.predictor_input_names[1]: state_h.astype(np.float32, copy=False),
                self.predictor_input_names[2]: state_c.astype(np.float32, copy=False),
            },
        )
        return pred_out, next_state_h, next_state_c

    def _joint_step(
        self,
        enc_frame: np.ndarray,
        pred_out: np.ndarray,
    ) -> np.ndarray:
        outputs = self.joint_session.run(
            None,
            {
                self.joint_input_names[0]: enc_frame.astype(np.float32, copy=False),
                self.joint_input_names[1]: pred_out.astype(np.float32, copy=False),
            },
        )
        return outputs[0]

    def decode_batch(
        self,
        enc_out: np.ndarray,
        enc_lens: np.ndarray,
    ) -> list[DecodeResult]:
        results: list[DecodeResult] = []
        for batch_idx in range(enc_out.shape[0]):
            enc_len = int(enc_lens[batch_idx])
            if enc_len <= 0:
                results.append(DecodeResult(token_ids=[]))
                continue

            state_h = np.zeros((self.num_layers, 1, self.hidden_dim), dtype=np.float32)
            state_c = np.zeros((self.num_layers, 1, self.hidden_dim), dtype=np.float32)
            token = np.asarray([[self.blank_id]], dtype=np.int64)
            pred_out, state_h, state_c = self._predict_step(token, state_h, state_c)

            hypothesis: list[int] = []
            emitted_log_probs: list[float] = []
            for time_idx in range(enc_len):
                enc_frame = enc_out[batch_idx : batch_idx + 1, time_idx, :]
                emitted = 0
                while emitted < self.max_symbols_per_step:
                    logits = self._joint_step(enc_frame, pred_out[:, -1, :])
                    log_probs = _log_softmax(logits.astype(np.float32, copy=False))
                    next_token = int(np.argmax(log_probs, axis=-1)[0])
                    if next_token == self.blank_id:
                        break
                    token_log_prob = float(log_probs[0, next_token])
                    hypothesis.append(next_token)
                    emitted_log_probs.append(token_log_prob)
                    token = np.asarray([[next_token]], dtype=np.int64)
                    pred_out, state_h, state_c = self._predict_step(token, state_h, state_c)
                    emitted += 1

            confidence = (
                math.exp(sum(emitted_log_probs) / float(len(emitted_log_probs)))
                if emitted_log_probs
                else 0.0
            )
            results.append(
                DecodeResult(
                    token_ids=hypothesis,
                    score=float(sum(emitted_log_probs)),
                    confidence=float(confidence),
                )
            )
        return results


class VyasaOnnxRuntime:
    """Direct onnxruntime loader for the Vyasa standalone RNN-T bundle."""

    def __init__(
        self,
        repo_id: str,
        *,
        bundle_root: str | Path | None = None,
        device: str = "cpu",
        token_env_var: str = "HF_TOKEN",
    ):
        self.repo_id = repo_id
        self.device = device
        self.token_env_var = token_env_var
        if bundle_root is None:
            bundle_root = Path("models") / repo_id.split("/")[-1]
        self.bundle_root = Path(bundle_root).expanduser().resolve()
        self.runtime_config: dict[str, object] | None = None
        self.encoder_session = None
        self.predictor_session = None
        self.joint_session = None
        self.decoder: OnnxRNNTGreedyDecoder | None = None
        self.sp = None
        self.sample_rate_hz = 16000
        self.blank_id = 0
        self.token_id_offset = DEFAULT_TOKEN_ID_OFFSET
        self.tokenizer_special_ids = dict(DEFAULT_TOKENIZER_SPECIAL_IDS)
        self.input_names: list[str] = []
        self.output_names: list[str] = []

    def warmup(self):
        if self.encoder_session is not None:
            return

        bundle_root = self._ensure_bundle()
        runtime_config = self._load_runtime_config(bundle_root)
        self.runtime_config = runtime_config

        ort = self._import_onnxruntime()
        spm = self._import_sentencepiece()

        artifacts = dict(runtime_config["artifacts"])
        encoder_path = bundle_root / artifacts["encoder"]
        predictor_path = bundle_root / artifacts["predictor"]
        joint_path = bundle_root / artifacts["joint"]
        tokenizer_model = bundle_root / artifacts["tokenizer_model"]

        providers = self._select_providers(ort, self.device)
        self.encoder_session = ort.InferenceSession(str(encoder_path), providers=providers)
        self.predictor_session = ort.InferenceSession(str(predictor_path), providers=providers)
        self.joint_session = ort.InferenceSession(str(joint_path), providers=providers)
        self.input_names = list(runtime_config["onnx"]["encoder_inputs"])
        self.output_names = list(runtime_config["onnx"]["encoder_outputs"])
        self.sample_rate_hz = int(runtime_config["sample_rate_hz"])
        self.blank_id = int(runtime_config.get("rnnt_blank_id", 0))
        self.token_id_offset = int(runtime_config.get("tokenizer_id_offset", DEFAULT_TOKEN_ID_OFFSET))
        special_ids_cfg = runtime_config.get("tokenizer_special_ids", {})
        self.tokenizer_special_ids = {
            key: int(special_ids_cfg.get(key, default_value))
            for key, default_value in DEFAULT_TOKENIZER_SPECIAL_IDS.items()
        }
        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
        self.decoder = OnnxRNNTGreedyDecoder(
            self.predictor_session,
            self.joint_session,
            blank_id=self.blank_id,
            max_symbols_per_step=int(runtime_config.get("rnnt_max_symbols_per_step", 100)),
        )

    def transcribe(self, audio: np.ndarray) -> str:
        self.warmup()
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError(f"Expected mono waveform array, got shape {samples.shape}")
        if not samples.size:
            return ""

        waveform = samples[np.newaxis, :]
        lengths = np.asarray([samples.shape[0]], dtype=np.int64)
        enc_out, _, enc_lens = self.encoder_session.run(
            self.output_names,
            {
                self.input_names[0]: waveform.astype(np.float32, copy=False),
                self.input_names[1]: lengths.astype(np.int64, copy=False),
            },
        )
        result = self.decoder.decode_batch(enc_out, enc_lens)[0]
        text = decode_token_ids(
            result.token_ids,
            self.sp,
            token_id_offset=self.token_id_offset,
            special_ids=self.tokenizer_special_ids,
        )
        return normalize_text(text)

    def _ensure_bundle(self) -> Path:
        runtime_config_path = self.bundle_root / DEFAULT_RUNTIME_CONFIG
        if runtime_config_path.is_file():
            return self.bundle_root

        token = os.environ.get(self.token_env_var)
        if not token:
            raise RuntimeError(
                f"{self.token_env_var} is not set, and the Vyasa ONNX bundle is not available locally at {self.bundle_root}."
            )

        self.bundle_root.mkdir(parents=True, exist_ok=True)
        huggingface_hub = self._import_huggingface_hub()
        logger.info("Downloading Vyasa ONNX bundle from %s into %s", self.repo_id, self.bundle_root)
        huggingface_hub.snapshot_download(
            repo_id=self.repo_id,
            token=token,
            local_dir=str(self.bundle_root),
        )
        return self.bundle_root

    def _load_runtime_config(self, bundle_root: Path) -> dict[str, object]:
        path = bundle_root / DEFAULT_RUNTIME_CONFIG
        if not path.exists():
            raise RuntimeError(f"Vyasa bundle is missing {DEFAULT_RUNTIME_CONFIG} at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _select_providers(ort, device: str) -> list[str]:
        available = set(ort.get_available_providers())
        if str(device).lower() == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _import_onnxruntime():
        try:
            return importlib.import_module("onnxruntime")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Run `uv sync` before using the Vyasa ONNX backend."
            ) from exc

    @staticmethod
    def _import_sentencepiece():
        try:
            return importlib.import_module("sentencepiece")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentencepiece is not installed. Run `uv sync` before using the Vyasa ONNX backend."
            ) from exc

    @staticmethod
    def _import_huggingface_hub():
        try:
            return importlib.import_module("huggingface_hub")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Run `uv sync` before using the Vyasa ONNX backend."
            ) from exc
