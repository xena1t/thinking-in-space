import sys
import types

import pytest

if "accelerate" not in sys.modules:
    accelerate_stub = types.ModuleType("accelerate")

    class Accelerator:  # pragma: no cover - simple stub for import-time dependency
        def __init__(self, *_, **__):
            self.is_main_process = True

        def wait_for_everyone(self):
            return None

    accelerate_stub.Accelerator = Accelerator
    sys.modules["accelerate"] = accelerate_stub

if "loguru" not in sys.modules:
    loguru_stub = types.ModuleType("loguru")

    class _Logger:  # pragma: no cover - trivial logging stub
        def debug(self, *_, **__):
            return None

        info = warning = error = debug

    loguru_stub.logger = _Logger()
    sys.modules["loguru"] = loguru_stub

if "tenacity" not in sys.modules:
    tenacity_stub = types.ModuleType("tenacity")

    class _TenacityPolicy:  # pragma: no cover - minimal combinable policy
        def __or__(self, _):
            return self

    def _policy(*_, **__):
        return _TenacityPolicy()

    def retry(*_, **__):  # pragma: no cover - decorator stub
        def decorator(fn):
            return fn

        return decorator

    tenacity_stub.retry = retry
    tenacity_stub.retry_if_exception_type = _policy
    tenacity_stub.stop_after_attempt = _policy
    tenacity_stub.stop_after_delay = _policy
    tenacity_stub.wait_fixed = _policy
    sys.modules["tenacity"] = tenacity_stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    from importlib.machinery import ModuleSpec

    def _torch_noop(*_, **__):  # pragma: no cover - catch-all torch stub
        return None

    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.float32 = torch_stub.long = torch_stub.int64 = object()
    torch_stub.zeros = torch_stub.cat = torch_stub.stack = _torch_noop
    torch_stub.__getattr__ = lambda *_: _torch_noop
    torch_stub.__spec__ = ModuleSpec("torch", loader=None)
    sys.modules["torch"] = torch_stub

if "jinja2" not in sys.modules:
    jinja_stub = types.ModuleType("jinja2")

    class BaseLoader:  # pragma: no cover - config stub
        pass

    class Environment:  # pragma: no cover - template stub
        def __init__(self, *_, **__):
            self.filters = {}

        def from_string(self, template):
            class _Template:
                def render(self, **__):
                    return template

            return _Template()

    class StrictUndefined:  # pragma: no cover - placeholder type
        pass

    jinja_stub.BaseLoader = BaseLoader
    jinja_stub.Environment = Environment
    jinja_stub.StrictUndefined = StrictUndefined
    sys.modules["jinja2"] = jinja_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class StoppingCriteria:  # pragma: no cover - placeholder base class
        pass

    class PreTrainedTokenizer:  # pragma: no cover - placeholder tokenizer
        pass

    class StoppingCriteriaList(list):  # pragma: no cover - placeholder list
        pass

    transformers_stub.StoppingCriteria = StoppingCriteria
    transformers_stub.PreTrainedTokenizer = PreTrainedTokenizer
    transformers_stub.StoppingCriteriaList = StoppingCriteriaList
    sys.modules["transformers"] = transformers_stub

if "evaluate" not in sys.modules:
    evaluate_stub = types.ModuleType("evaluate")
    sys.modules["evaluate"] = evaluate_stub

if "sqlitedict" not in sys.modules:
    sqlitedict_stub = types.ModuleType("sqlitedict")

    class SqliteDict(dict):  # pragma: no cover - minimal stub
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    sqlitedict_stub.SqliteDict = SqliteDict
    sys.modules["sqlitedict"] = sqlitedict_stub

if "sacrebleu" not in sys.modules:
    sacrebleu_stub = types.ModuleType("sacrebleu")

    class _SacreBleuResult:  # pragma: no cover - placeholder result
        score = 0.0

        def signature(self):
            return ""

    def corpus_bleu(*_, **__):
        return _SacreBleuResult()

    sacrebleu_stub.corpus_bleu = corpus_bleu
    sys.modules["sacrebleu"] = sacrebleu_stub

try:  # pragma: no cover - prefer the real library when available
    import huggingface_hub  # type: ignore
    from huggingface_hub.hf_file_system import HfFileSystem  # type: ignore
except Exception:  # pragma: no cover - minimal fallback stub
    hf_stub = types.ModuleType("huggingface_hub")
    hf_file_system_stub = types.ModuleType("huggingface_hub.hf_file_system")

    class HfFileSystem:  # pragma: no cover - minimal stub, real behaviour tested below
        def glob(self, path, *_, **__):
            return [path]

    class CommitInfo:  # pragma: no cover - placeholder to satisfy datasets imports
        pass

    hf_stub.CommitInfo = CommitInfo
    hf_file_system_stub.HfFileSystem = HfFileSystem
    hf_stub.hf_file_system = hf_file_system_stub
    sys.modules["huggingface_hub"] = hf_stub
    sys.modules["huggingface_hub.hf_file_system"] = hf_file_system_stub

from lmms_eval.api import task as task_module
from lmms_eval.tasks import _glob_sanitize, _glob_validate


def test_sanitize_glob_preserves_original_and_rewrites():
    original = {"data_files": {"train": "scannet/**.mp4"}}
    sanitized = task_module._sanitize_globs(original)

    assert sanitized["data_files"]["train"] == "scannet/**/*.mp4"
    # Ensure the input dictionary has not been mutated.
    assert original["data_files"]["train"] == "scannet/**.mp4"


def test_sanitize_glob_inserts_missing_separator():
    original = {"data_files": {"train": "bad**.mp4"}}
    sanitized = task_module._sanitize_globs(original)

    assert sanitized["data_files"]["train"] == "bad/**/*.mp4"
    assert original["data_files"]["train"] == "bad**.mp4"


def test_glob_validate_flags_invalid_patterns():
    bad_value = {"data_files": {"train": "bad/**video.mp4"}}

    with pytest.raises(ValueError):
        _glob_validate(bad_value)

    sanitized = _glob_sanitize(bad_value)
    # Validation should succeed once sanitized.
    _glob_validate(sanitized)


def test_patch_hf_glob_sanitizes_paths(monkeypatch):
    from huggingface_hub.hf_file_system import HfFileSystem

    from lmms_eval.api import task as task_module

    calls = []

    def _original_glob(self, path, *args, **kwargs):
        calls.append(path)
        if path.endswith("**.mp4"):
            raise ValueError("Invalid pattern: '**' can only be an entire path component")
        return ["ok"]

    monkeypatch.setattr(HfFileSystem, "glob", _original_glob, raising=False)
    for attr in ("_lmms_eval_glob_patched", "_lmms_eval_original_glob"):
        if hasattr(HfFileSystem, attr):
            delattr(HfFileSystem, attr)
    task_module._HF_GLOB_PATCHED = False

    task_module._patch_hf_hub_glob()

    fs = HfFileSystem()
    assert fs.glob("videos/**.mp4") == ["ok"]
    assert calls == ["videos/**/*.mp4"]
