import multiprocessing as mp
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:  # pragma: no cover - optional dependency that is validated at runtime
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - handled during model init
    LLM = None
    SamplingParams = None


def _patch_vllm_qwen3_metadata_guard() -> None:
    """Monkey patch vLLM's Qwen3-VL metadata access to tolerate ``None``."""

    try:  # pragma: no cover - optional dependency
        import importlib
        import inspect
        import re

        module = importlib.import_module("vllm.model_executor.models.qwen3_vl")
    except Exception:  # pragma: no cover - if vLLM is unavailable or layout changes
        return

    target = getattr(module, "_call_hf_processor", None)
    if not callable(target):  # pragma: no cover - unexpected layout
        return

    try:
        source = inspect.getsource(target)
    except (OSError, IOError):  # pragma: no cover - source unavailable (pyc only)
        return

    if "(metadata or {})" in source and "meta_for_ctor" in source:
        return  # already patched upstream

    # Ensure metadata lookups are safe when ``metadata`` is ``None``.
    patched_source = source.replace("metadata.get(", "(metadata or {}).get(")

    # Strengthen the VideoMetadata construction with inferred defaults.
    pattern = re.compile(
        r"(?P<indent>\s*)metadata\s*=\s*VideoMetadata\(\*\*\{.*?\}\)",
        re.DOTALL,
    )

    replacement_block = """
metadata_dict = metadata.to_dict() if metadata else {}
_meta_dict = dict(metadata_dict)
_meta_src = dict(metadata_dict)
video_array = video_feat
try:
    _t = int(_meta_dict.get("total_num_frames", video_array.shape[0]))
except Exception:
    try:
        _t = int(len(video_array))
    except Exception:
        _t = _meta_src.get("total_num_frames", 0)
try:
    _h = int(getattr(video_array, "shape", [None, None, None])[1])
    _w = int(getattr(video_array, "shape", [None, None, None])[2])
except Exception:
    try:
        _t = int(len(video_array))
    except Exception:
        _t = _meta_src.get("total_num_frames", 0)
    _h = _meta_src.get("height")
    _w = _meta_src.get("width")

meta_for_ctor = {
    "total_num_frames": _meta_dict.get("total_num_frames", _t),
    "height": _meta_dict.get("height", _h),
    "width": _meta_dict.get("width", _w),
    "fps": _meta_dict.get("fps", _meta_dict.get("frame_rate", 30)),
}
for _k, _v in _meta_dict.items():
    if _k not in meta_for_ctor:
        meta_for_ctor[_k] = _v

metadata = VideoMetadata(**meta_for_ctor)
""".strip("\n")

    def _replace_metadata_block(match: "re.Match[str]") -> str:
        indent = match.group("indent")
        return "\n".join(f"{indent}{line}" for line in replacement_block.splitlines())

    patched_source, count = pattern.subn(_replace_metadata_block, patched_source)

    if patched_source == source and count == 0:
        return  # nothing to patch

    namespace: Dict[str, object] = {}
    exec(compile(patched_source, target.__code__.co_filename, "exec"), module.__dict__, namespace)
    module._call_hf_processor = namespace.get(target.__name__, target)

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("qwen3vl")
class Qwen3VL(lmms):
    """Wrapper that exposes Qwen3-VL models through the lmms-eval interface.

    The implementation mirrors the existing Qwen2-VL integration but allows
    loading checkpoints from the Qwen3-VL release. The model is served through
    vLLM for efficient batched generation and expects that the official
    `Qwen3-VL` repository (and its qwen-vl-utils helper package) is available on
    the python path. This matches the workflow used by the Thinking in Space
    project for other Qwen variants.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        modality: str = "video",
        device: str = "cuda",
        device_map: str = "cuda",
        batch_size: str = "1",
        max_frames_num: Optional[int] = None,
        max_model_len: int = 65536,
        tensor_parallel_size: Optional[int] = None,
        sampling_temperature: float = 0.0,
        max_new_tokens: int = 64,
        vllm_kwargs: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        super().__init__()

        if LLM is None or SamplingParams is None:
            raise ImportError(
                "vLLM is required for the Qwen3-VL backend. Install it with "
                "`pip install vllm>=0.5.4` before launching the evaluation."
            )

        repo_root = os.getenv("QWEN3_VL_REPO", "Qwen3-VL")
        utils_path = os.path.join(repo_root, "qwen-vl-utils", "src")
        if os.path.isdir(repo_root) and repo_root not in sys.path:
            sys.path.append(repo_root)
        if os.path.isdir(utils_path) and utils_path not in sys.path:
            sys.path.append(utils_path)
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError(
                "qwen_vl_utils was not found. Please clone the official "
                "Qwen3-VL repository next to this project or set the "
                "QWEN3_VL_REPO environment variable to its location."
            ) from exc
        self._process_vision_info = process_vision_info

        if kwargs:
            # Surface unsupported kwargs early to match other model wrappers.
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(f"Unexpected kwargs provided: {unexpected}")

        self.path = pretrained
        tensor_parallel = tensor_parallel_size or int(os.getenv("VLLM_TENSOR_PARALLISM", 1))
        vllm_kwargs = vllm_kwargs.copy() if vllm_kwargs else {}
        vllm_kwargs.setdefault("tensor_parallel_size", tensor_parallel)
        vllm_kwargs.setdefault("max_model_len", max_model_len)
        # trust_remote_code is required for Qwen releases and can still be
        # overridden through `vllm_kwargs` when necessary.
        vllm_kwargs.setdefault("trust_remote_code", True)

        self._ensure_spawn_start_method()
        _patch_vllm_qwen3_metadata_guard()
        self._model = LLM(self.path, **vllm_kwargs)
        self._processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)

        self.sampling_params = SamplingParams(temperature=sampling_temperature, max_tokens=max_new_tokens)

        batch_size = int(batch_size)
        if batch_size != 1:
            raise AssertionError(
                f"Batch size should be 1 for Qwen3VL, but got {batch_size}."
            )
        self.batch_size_per_gpu = batch_size

        self._config = None
        self._device = device
        self._rank = 0
        self._world_size = 1

        self.modality = modality
        self.max_frames_num = max_frames_num

    @staticmethod
    def _ensure_spawn_start_method() -> None:
        method = mp.get_start_method(allow_none=True)
        if method != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError as exc:  # pragma: no cover - defensive programming
                raise RuntimeError(
                    "Qwen3VL requires the multiprocessing start method to be 'spawn'. "
                    "Please restart the program and ensure no CUDA work happens before "
                    "lmms_eval imports, or pre-set VLLM_WORKER_MULTIPROC_METHOD=spawn."
                ) from exc

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        return [j for i in input for j in i]

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if self.modality == "image":
                raise NotImplementedError("Image inference for Qwen3VL is not supported yet.")
            if self.modality != "video":
                raise NotImplementedError(f"Unsupported modality: {self.modality}")

            if len(visuals) != 1:
                raise AssertionError(f"Only one video is supported, but got {len(visuals)} videos.")
            video_path = visuals[0]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"{video_path}",
                        },
                        {"type": "text", "text": f"{contexts}"},
                    ],
                }
            ]
            if self.max_frames_num:
                messages[0]["content"][0]["nframes"] = self.max_frames_num

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            _, video_inputs = self._process_vision_info(messages)
            video_inputs = self._coerce_vllm_video_payload(
                video_inputs,
                default_nframes=self.max_frames_num or 32,
            )
            generated = self._model.generate(
                {
                    "prompt": text,
                    "multi_modal_data": {"video": video_inputs},
                },
                sampling_params=self.sampling_params,
            )
            output_text = generated[0].outputs[0].text
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    @staticmethod
    def _coerce_vllm_video_payload(video_inputs: Any, default_nframes: int = 32) -> List[Any]:
        """Return ``multi_modal_data['video']`` as primitives understood by vLLM."""

        try:  # pragma: no cover - optional import
            import numpy as np  # type: ignore
        except ImportError:  # pragma: no cover - numpy optional
            np = None  # type: ignore

        try:  # pragma: no cover - optional import
            import torch  # type: ignore
        except ImportError:  # pragma: no cover - torch optional
            torch = None  # type: ignore

        def _normalise_array(value: Any) -> Any:
            if torch is not None and isinstance(value, torch.Tensor):  # type: ignore[attr-defined]
                tensor = value.detach().cpu()
                if tensor.ndim == 4 and tensor.shape[1] in (1, 3, 4):
                    tensor = tensor.permute(0, 2, 3, 1).contiguous()
                elif tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                    tensor = tensor.permute(1, 2, 0).contiguous()
                if tensor.dtype != torch.uint8:  # type: ignore[attr-defined]
                    tensor = tensor.clamp(0, 255).to(torch.uint8)
                value = tensor.numpy()

            if np is not None and isinstance(value, np.ndarray):  # type: ignore[attr-defined]
                array = value
                if array.ndim == 4 and array.shape[1] in (1, 3, 4):
                    array = array.transpose(0, 2, 3, 1)
                elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
                    array = array.transpose(1, 2, 0)
                elif array.ndim == 2:
                    array = array[:, :, None]
                if array.dtype != np.uint8:  # type: ignore[attr-defined]
                    array = np.clip(array, 0, 255).astype(np.uint8)
                return array

            return value

        def _coerce_video_value(value: Any) -> Any:
            if value is None:
                return None

            if isinstance(value, (str, bytes, bytearray)):
                return value

            if isinstance(value, Mapping):
                entry = dict(value)
                source = entry.get("video")
                if source is None:
                    source = entry.get("path")
                if source is None:
                    source = entry.get("frames")
                return _coerce_video_value(source)

            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and not (
                np is not None and isinstance(value, np.ndarray)  # type: ignore[attr-defined]
            ):
                sequence = list(value)
                if not sequence:
                    return None
                if len(sequence) == 2 and isinstance(sequence[0], (str, bytes, bytearray)):
                    return _coerce_video_value(sequence[0])
                if np is not None:  # type: ignore[attr-defined]
                    frames: List[Any] = []
                    for frame in sequence:
                        coerced = _coerce_video_value(frame)
                        if isinstance(coerced, np.ndarray):  # type: ignore[attr-defined]
                            if coerced.ndim == 4 and coerced.shape[0] == 1:
                                frames.append(coerced[0])
                            else:
                                frames.append(coerced)
                        else:
                            frames.append(coerced)
                    if frames and all(isinstance(frame, np.ndarray) and frame.ndim == 3 for frame in frames):  # type: ignore[attr-defined]
                        stacked = np.stack(frames, axis=0)  # type: ignore[attr-defined]
                        if stacked.dtype != np.uint8:  # type: ignore[attr-defined]
                            stacked = np.clip(stacked, 0, 255).astype(np.uint8)  # type: ignore[attr-defined]
                        return stacked
                return _coerce_video_value(sequence[0])

            normalised = _normalise_array(value)
            if np is not None and isinstance(normalised, np.ndarray):  # type: ignore[attr-defined]
                array = normalised
                if array.ndim == 3:
                    return array[None, ...]
                if array.ndim == 4:
                    return array
                if array.ndim == 2:
                    return array[None, ..., None]
            return normalised

        if video_inputs is None:
            return []

        if isinstance(video_inputs, Sequence) and not isinstance(video_inputs, (str, bytes, bytearray)) and not (
            np is not None and isinstance(video_inputs, np.ndarray)  # type: ignore[attr-defined]
        ):
            result: List[Any] = []
            for item in video_inputs:
                coerced = _coerce_video_value(item)
                if coerced is None:
                    continue
                result.append(coerced)
            return result

        coerced = _coerce_video_value(video_inputs)
        return [] if coerced is None else [coerced]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Log-likelihood computation is not implemented for Qwen3VL.")
