from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.asr.vyasa_onnx import VyasaOnnxRuntime


class _FakeIo:
    def __init__(self, name: str, shape):
        self.name = name
        self.shape = shape


class _FakeEncoderSession:
    def get_inputs(self):
        return [_FakeIo("wav", [1, "samples"]), _FakeIo("wav_lens", [1])]

    def run(self, output_names, feeds):
        enc_out = np.zeros((1, 1, 4), dtype=np.float32)
        ctc_logits = np.zeros((1, 1, 6), dtype=np.float32)
        enc_lens = np.asarray([1], dtype=np.int64)
        return enc_out, ctc_logits, enc_lens


class _FakePredictorSession:
    def get_inputs(self):
        return [
            _FakeIo("tokens", [1, 1]),
            _FakeIo("state_h", [1, 1, 4]),
            _FakeIo("state_c", [1, 1, 4]),
        ]

    def run(self, output_names, feeds):
        pred_out = np.zeros((1, 1, 4), dtype=np.float32)
        next_state_h = np.zeros((1, 1, 4), dtype=np.float32)
        next_state_c = np.zeros((1, 1, 4), dtype=np.float32)
        return pred_out, next_state_h, next_state_c


class _FakeJointSession:
    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [
            _FakeIo("enc_frame", [1, 4]),
            _FakeIo("pred_out", [1, 4]),
        ]

    def run(self, output_names, feeds):
        self.calls += 1
        if self.calls == 1:
            return [np.asarray([[0.0, -5.0, -5.0, -5.0, -5.0, 6.0]], dtype=np.float32)]
        return [np.asarray([[7.0, -5.0, -5.0, -5.0, -5.0, -5.0]], dtype=np.float32)]


class _FakeOrtModule:
    def __init__(self):
        self._joint = _FakeJointSession()

    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider"]

    def InferenceSession(self, path: str, providers=None):
        if path.endswith("encoder.onnx"):
            return _FakeEncoderSession()
        if path.endswith("predictor.onnx"):
            return _FakePredictorSession()
        if path.endswith("joint.onnx"):
            return self._joint
        raise AssertionError(f"Unexpected session path: {path}")


class _FakeSentencePieceProcessor:
    def __init__(self, model_file: str):
        self.model_file = model_file

    def decode(self, ids):
        if ids == [1]:
            return "namaste"
        return ""


class _FakeSentencePieceModule:
    SentencePieceProcessor = _FakeSentencePieceProcessor


class _FakeHfHubModule:
    def __init__(self):
        self.calls = []

    def snapshot_download(self, *, repo_id: str, token: str, local_dir: str):
        self.calls.append((repo_id, token, local_dir))
        return local_dir


class VyasaOnnxRuntimeTests(unittest.TestCase):
    def _write_bundle(self, root: Path):
        (root / "export").mkdir(parents=True, exist_ok=True)
        (root / "tokenizer").mkdir(parents=True, exist_ok=True)
        runtime_config = {
            "artifacts": {
                "encoder": "export/encoder.onnx",
                "predictor": "export/predictor.onnx",
                "joint": "export/joint.onnx",
                "tokenizer_model": "tokenizer/tokenizer.model",
            },
            "onnx": {
                "encoder_inputs": ["wav", "wav_lens"],
                "encoder_outputs": ["enc_out", "ctc_logits", "enc_lens"],
            },
            "sample_rate_hz": 16000,
            "rnnt_blank_id": 0,
            "rnnt_max_symbols_per_step": 10,
            "tokenizer_id_offset": 4,
            "tokenizer_special_ids": {"blank": 0, "sos": 1, "eos": 2, "pad": 3},
        }
        (root / "runtime_config.json").write_text(json.dumps(runtime_config), encoding="utf-8")
        for rel in [
            "export/encoder.onnx",
            "export/predictor.onnx",
            "export/joint.onnx",
            "tokenizer/tokenizer.model",
        ]:
            (root / rel).write_bytes(b"stub")

    def test_transcribe_uses_direct_onnxruntime_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_root = Path(tmpdir)
            self._write_bundle(bundle_root)
            fake_ort = _FakeOrtModule()

            def fake_import_module(name: str):
                if name == "onnxruntime":
                    return fake_ort
                if name == "sentencepiece":
                    return _FakeSentencePieceModule()
                if name == "huggingface_hub":
                    return _FakeHfHubModule()
                raise ModuleNotFoundError(name)

            runtime = VyasaOnnxRuntime(
                repo_id="somyalab/Vyasa_mini_rnnt_onnx_v2",
                bundle_root=bundle_root,
            )
            with patch("app.asr.vyasa_onnx.importlib.import_module", side_effect=fake_import_module):
                transcript = runtime.transcribe(np.ones(1600, dtype=np.float32))

        self.assertEqual(transcript, "namaste")

    def test_missing_local_bundle_requires_hf_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_root = Path(tmpdir) / "missing_bundle"
            runtime = VyasaOnnxRuntime(
                repo_id="somyalab/Vyasa_mini_rnnt_onnx_v2",
                bundle_root=bundle_root,
            )
            with patch.dict(os.environ, {}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                    runtime.warmup()
