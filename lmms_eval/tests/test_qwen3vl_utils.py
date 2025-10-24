import importlib.util
from collections import UserDict
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "models" / "qwen3vl.py"


def _load_qwen3vl_module():
    spec = importlib.util.spec_from_file_location("lmms_eval.models.qwen3vl", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except ImportError as exc:  # pragma: no cover - optional deps missing
        pytest.skip(f"Required dependency for qwen3vl unavailable: {exc}")
    return module


def test_ensure_video_metadata_fills_missing_dict():
    Qwen3VL = _load_qwen3vl_module().Qwen3VL
    payload = [{"video": "foo.mp4", "metadata": None}]
    fixed = Qwen3VL._ensure_video_metadata(payload)
    assert isinstance(fixed, list)
    assert fixed[0]["metadata"]["do_sample_frames"] is True


def test_ensure_video_metadata_preserves_existing_fields():
    Qwen3VL = _load_qwen3vl_module().Qwen3VL
    payload = {
        "clip": {"video": "bar.mp4", "metadata": {"fps": 30, "do_sample_frames": False}}
    }
    fixed = Qwen3VL._ensure_video_metadata(payload)
    assert fixed["clip"]["metadata"]["fps"] == 30
    assert fixed["clip"]["metadata"]["do_sample_frames"] is False


def test_ensure_video_metadata_handles_mapping_subclasses():
    Qwen3VL = _load_qwen3vl_module().Qwen3VL
    payload = UserDict({
        "videos": [
            UserDict({"video": "baz.mp4", "metadata": None}),
            UserDict({"video": "qux.mp4"}),
        ]
    })
    fixed = Qwen3VL._ensure_video_metadata(payload)
    assert isinstance(fixed["videos"], list)
    assert all("metadata" in clip for clip in fixed["videos"])
    assert all(clip["metadata"]["do_sample_frames"] is True for clip in fixed["videos"])


def test_ensure_video_metadata_converts_none_payload():
    Qwen3VL = _load_qwen3vl_module().Qwen3VL
    fixed = Qwen3VL._ensure_video_metadata(None)
    assert fixed["metadata"]["do_sample_frames"] is True


def test_ensure_video_metadata_injects_when_missing_metadata_key():
    Qwen3VL = _load_qwen3vl_module().Qwen3VL
    payload = {"videos": ({"video": "foo.mp4"},)}
    fixed = Qwen3VL._ensure_video_metadata(payload)
    assert fixed["metadata"]["do_sample_frames"] is True
    assert fixed["videos"][0]["metadata"]["do_sample_frames"] is True
    assert isinstance(fixed["videos"], tuple)
