import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "models" / "qwen3vl.py"


@pytest.fixture(name="qwen3vl_module")
def fixture_qwen3vl_module():
    spec = importlib.util.spec_from_file_location("lmms_eval.models.qwen3vl", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except ImportError as exc:  # pragma: no cover - optional deps missing
        pytest.skip(f"Required dependency for qwen3vl unavailable: {exc}")
    return module


def test_coerce_video_payload_handles_path(qwen3vl_module):
    Qwen3VL = qwen3vl_module.Qwen3VL
    result = Qwen3VL._coerce_vllm_video_payload("foo.mp4")
    assert isinstance(result, list)
    assert result == ["foo.mp4"]


def test_coerce_video_payload_normalises_numpy_array(qwen3vl_module):
    np = pytest.importorskip("numpy")
    Qwen3VL = qwen3vl_module.Qwen3VL
    array = np.random.randint(0, 255, size=(4, 3, 8, 6), dtype=np.uint8)
    result = Qwen3VL._coerce_vllm_video_payload(array)
    assert len(result) == 1
    coerced = result[0]
    assert coerced.shape == (4, 8, 6, 3)
    assert coerced.dtype == np.uint8


def test_coerce_video_payload_accepts_mapping(qwen3vl_module):
    np = pytest.importorskip("numpy")
    Qwen3VL = qwen3vl_module.Qwen3VL
    clip = {"video": np.random.rand(2, 3, 4, 5)}
    result = Qwen3VL._coerce_vllm_video_payload(clip)
    assert len(result) == 1
    coerced = result[0]
    assert coerced.shape == (2, 4, 5, 3)


def test_coerce_video_payload_handles_sequence_of_frames(qwen3vl_module):
    np = pytest.importorskip("numpy")
    Qwen3VL = qwen3vl_module.Qwen3VL
    frames = [np.random.randint(0, 255, size=(4, 4, 3), dtype=np.uint8) for _ in range(3)]
    result = Qwen3VL._coerce_vllm_video_payload(frames)
    assert len(result) == 1
    coerced = result[0]
    assert coerced.shape == (3, 4, 4, 3)
    assert coerced.dtype == np.uint8


def test_coerce_video_payload_ignores_none_entries(qwen3vl_module):
    Qwen3VL = qwen3vl_module.Qwen3VL
    assert Qwen3VL._coerce_vllm_video_payload(None) == []

