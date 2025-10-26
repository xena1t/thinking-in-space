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
        module_source = None
        module_path = getattr(module, "__file__", None)
        if module_path is not None:
            try:
                with open(module_path, "r", encoding="utf-8") as handle:
                    module_source = handle.read()
            except OSError:
                module_source = None
        if not module_source:
            return

        # Extract the target function manually from the module source.
        func_match = re.search(
            r"^def\\s+_call_hf_processor\\(.*?^\\s*return\\s+mm_processed_data",
            module_source,
            re.DOTALL | re.MULTILINE,
        )
        if not func_match:
            return
        source = func_match.group(0)

    if "(metadata or {})" in source and "meta_for_ctor" in source:
        return  # already patched upstream

    # Ensure metadata lookups are safe when ``metadata`` is ``None``.
    metadata_guard_pattern = re.compile(r"metadata\s*\.\s*get\(")
    patched_source = metadata_guard_pattern.sub("(metadata or {}).get(", source)

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
            # --- ENFORCE 1: keep messages/video content count in sync with multi_modal_data ---
            def _count_video_parts(msgs: List[Dict[str, Any]]) -> int:
                cnt = 0
                for m in msgs:
                    if m.get("role") != "user":
                        continue
                    content = m.get("content", [])
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "video":
                                cnt += 1
                return cnt

            num_mm = len(video_inputs)
            num_msg_videos = _count_video_parts(messages)

            if num_mm == 0 and num_msg_videos > 0:
                for m in messages:
                    if m.get("role") != "user":
                        continue
                    content = m.get("content", [])
                    if isinstance(content, list):
                        m["content"] = [
                            c for c in content if not (isinstance(c, dict) and c.get("type") == "video")
                        ]
                num_msg_videos = _count_video_parts(messages)

            if num_mm != num_msg_videos and num_mm > 0 and num_msg_videos > 0:
                keep = min(num_mm, num_msg_videos)
                video_inputs = video_inputs[:keep]
                trimmed = 0
                for m in messages:
                    if m.get("role") != "user":
                        continue
                    newc = []
                    for c in m.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "video":
                            if trimmed < keep:
                                newc.append(c)
                                trimmed += 1
                            else:
                                continue
                        else:
                            newc.append(c)
                    m["content"] = newc

            payload = {
                "prompt": text,
                "multi_modal_data": {"video": video_inputs} if video_inputs else {},
            }

            generated = self._model.generate(
                payload,
                sampling_params=self.sampling_params,
            )
            output_text = generated[0].outputs[0].text
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    @staticmethod
    def _coerce_vllm_video_payload(video_inputs: Any, default_nframes: int = 32) -> List[Dict[str, Any]]:
        """
        Normalize the incoming "video inputs" into a list of dicts:
            [{"video": np.ndarray[T,H,W,3]}, ...]
        Never return raw arrays/tensors; always wrap in {"video": ...}.
        """
        import numpy as np  # type: ignore

        try:
            import torch
        except Exception:  # pragma: no cover - torch is optional at runtime
            torch = None

        def _as_video_array(x: Any) -> Optional[Any]:
            if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            if getattr(x, "ndim", None) == 3:
                x = x[None, ...]
            return x if getattr(x, "ndim", None) == 4 else None

        def _stack_frames(frames: Any) -> Optional[Any]:
            if not isinstance(frames, list) or not frames:
                return None
            if all(getattr(f, "ndim", None) == 3 for f in frames):
                try:
                    arr = np.stack(frames, axis=0)
                    return arr
                except Exception:
                    return None
            return None

        out: List[Dict[str, Any]] = []
        if video_inputs is None:
            return out

        if isinstance(video_inputs, list):
            for entry in video_inputs:
                if isinstance(entry, dict):
                    if "frames" in entry and isinstance(entry["frames"], list):
                        arr = _stack_frames(entry["frames"])
                        if arr is not None:
                            out.append({"video": arr})
                            continue
                    if "video" in entry:
                        arr = _as_video_array(entry["video"])
                        if arr is not None:
                            out.append({"video": arr})
                            continue
                else:
                    arr = _as_video_array(entry)
                    if arr is not None:
                        out.append({"video": arr})
            return out

        if isinstance(video_inputs, dict):
            if "frames" in video_inputs and isinstance(video_inputs["frames"], list):
                arr = _stack_frames(video_inputs["frames"])
                if arr is not None:
                    return [{"video": arr}]
            if "video" in video_inputs:
                arr = _as_video_array(video_inputs["video"])
                if arr is not None:
                    return [{"video": arr}]
            return out

        arr = _as_video_array(video_inputs)
        if arr is not None:
            return [{"video": arr}]
        return out


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Log-likelihood computation is not implemented for Qwen3VL.")
