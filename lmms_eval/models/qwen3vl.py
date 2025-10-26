import multiprocessing as mp
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:  # pragma: no cover - optional based on transformers version
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover - fallback for older releases
    AutoModelForVision2Seq = None

try:  # pragma: no cover - optional model class
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:  # pragma: no cover - fallback if class unavailable
    Qwen2VLForConditionalGeneration = None

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
    loading checkpoints from the Qwen3-VL release. It can either execute through
    vLLM or fall back to Hugging Face Transformers; the latter is the default
    to avoid current vLLM limitations with video inputs. The wrapper still
    expects the official `Qwen3-VL` repository (and its qwen-vl-utils helper
    package) on the Python path, matching the workflow used by the Thinking in
    Space project for other Qwen variants.
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
        use_vllm: bool = False,
        hf_dtype: Optional[str] = "auto",
        vllm_kwargs: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        super().__init__()

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
            unexpected = ", ".join(kwargs.keys())
            raise ValueError(f"Unexpected kwargs provided: {unexpected}")

        self.path = pretrained
        self.use_vllm = bool(use_vllm)
        self.temperature = float(sampling_temperature)
        self.max_new_tokens = int(max_new_tokens)

        self._processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)

        batch_size_int = int(batch_size)
        if batch_size_int != 1:
            raise AssertionError(
                f"Batch size should be 1 for Qwen3VL, but got {batch_size_int}."
            )
        self.batch_size_per_gpu = batch_size_int

        self._config = None
        self._rank = 0
        self._world_size = 1
        self._hf_device_map: Optional[Any] = None

        if self.use_vllm:
            if LLM is None or SamplingParams is None:
                raise ImportError(
                    "vLLM is required for the Qwen3-VL backend. Install it with "
                    "`pip install vllm>=0.5.4` before launching the evaluation, or "
                    "set use_vllm=False to run with the Transformers backend."
                )

            tensor_parallel = tensor_parallel_size or int(os.getenv("VLLM_TENSOR_PARALLISM", 1))
            vllm_kwargs = vllm_kwargs.copy() if vllm_kwargs else {}
            vllm_kwargs.setdefault("tensor_parallel_size", tensor_parallel)
            vllm_kwargs.setdefault("max_model_len", max_model_len)
            vllm_kwargs.setdefault("trust_remote_code", True)

            self._ensure_spawn_start_method()
            _patch_vllm_qwen3_metadata_guard()
            self._model = LLM(self.path, **vllm_kwargs)
            self.sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
            self._device = device
        else:
            if hf_dtype is None or hf_dtype == "auto":
                torch_dtype = "auto"
            elif isinstance(hf_dtype, str):
                torch_dtype = getattr(torch, hf_dtype)
            else:
                torch_dtype = hf_dtype

            hf_device_map = None
            if device_map and device_map not in ("", "cuda"):
                hf_device_map = device_map
            self._device = torch.device(device)

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
                "device_map": hf_device_map,
            }

            model_loaded = False
            model_exceptions: List[Exception] = []

            if AutoModelForVision2Seq is not None:
                try:
                    self._model = AutoModelForVision2Seq.from_pretrained(self.path, **model_kwargs)
                    model_loaded = True
                except Exception as exc:  # pragma: no cover - fall back to specific class
                    model_exceptions.append(exc)

            if not model_loaded and Qwen2VLForConditionalGeneration is not None:
                try:
                    self._model = Qwen2VLForConditionalGeneration.from_pretrained(self.path, **model_kwargs)
                    model_loaded = True
                except Exception as exc:  # pragma: no cover - propagate below
                    model_exceptions.append(exc)

            if not model_loaded:
                error_messages = "; ".join(str(exc) for exc in model_exceptions) or "No compatible vision-language model class available."
                raise ValueError(
                    "Failed to load Qwen3-VL model with the Transformers backend. "
                    "Ensure your transformers version provides AutoModelForVision2Seq or Qwen2VLForConditionalGeneration. "
                    f"Underlying errors: {error_messages}"
                )

            if hf_device_map is None:
                self._model.to(self._device)
            self._model.eval()
            self.sampling_params = None
            self._hf_device_map = hf_device_map

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
            _, raw_video_inputs = self._process_vision_info(messages)
            if self.use_vllm:
                fallback_entry = {"video": str(video_path)}
                if self.max_frames_num:
                    fallback_entry["nframes"] = self.max_frames_num
                if not raw_video_inputs:
                    raw_video_inputs = [fallback_entry]
                video_inputs = self._coerce_vllm_video_payload(
                    raw_video_inputs,
                    default_nframes=self.max_frames_num or 32,
                )
                if not video_inputs:
                    video_inputs = [fallback_entry]
                payload = {
                    "prompt": text,
                    "multi_modal_data": {"video": video_inputs} if video_inputs else {},
                }

                generated = self._model.generate(
                    payload,
                    sampling_params=self.sampling_params,
                )
                output_text = generated[0].outputs[0].text
            else:
                video_tensors = self._prepare_hf_video_inputs(raw_video_inputs, video_path)
                if not video_tensors:
                    raise RuntimeError(f"Failed to load video frames for {video_path}.")
                inputs = self._processor(
                    text=[text],
                    videos=video_tensors,
                    return_tensors="pt",
                    padding=True,
                )
                tensor_inputs: Dict[str, torch.Tensor] = {}
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        target_device = self.device if self._hf_device_map is None else getattr(self._model, 'device', self.device)
                        tensor_inputs[key] = value.to(target_device)
                input_length = tensor_inputs["input_ids"].shape[-1]
                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": self.max_new_tokens,
                    "pad_token_id": self._tokenizer.eos_token_id,
                    "eos_token_id": self._tokenizer.eos_token_id,
                }
                if self.temperature > 0:
                    gen_kwargs.update({"do_sample": True, "temperature": self.temperature})
                else:
                    gen_kwargs.update({"do_sample": False})

                with torch.no_grad():
                    generated_ids = self._model.generate(**tensor_inputs, **gen_kwargs)
                new_tokens = generated_ids[:, input_length:]
                output_text = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            res.append(output_text)
            pbar.update(1)
        pbar.close()
        return res

    def _prepare_hf_video_inputs(
        self, raw_video_inputs: List[Any], video_path: str
    ) -> List[Any]:
        if raw_video_inputs:
            prepared: List[Any] = []
            for entry in raw_video_inputs:
                if isinstance(entry, dict):
                    if entry.get("video") is not None:
                        prepared.append(entry["video"])
                    elif entry.get("frames"):
                        stacked = self._stack_frames(entry["frames"])
                        if stacked is not None:
                            prepared.append(stacked)
                else:
                    prepared.append(entry)
            prepared = [item for item in prepared if item is not None]
            if prepared:
                return prepared
        fallback = self._load_video_frames(video_path, self.max_frames_num)
        return [fallback] if fallback is not None else []

    @staticmethod
    def _stack_frames(frames: Sequence[Any]) -> Optional[Any]:
        try:
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover - numpy optional
            return None

        valid_frames = [frame for frame in frames if getattr(frame, "ndim", None) == 3]
        if not valid_frames:
            return None
        try:
            stacked = np.stack(valid_frames, axis=0)
        except Exception:
            return None
        return stacked

    @staticmethod
    def _load_video_frames(video_path: str, max_frames: Optional[int]) -> Optional[Any]:
        try:
            import decord  # type: ignore
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover - decord optional
            decord = None  # type: ignore

        if decord is not None:
            try:
                vr = decord.VideoReader(video_path)
                total = len(vr)
                if total == 0:
                    return None
                if max_frames and max_frames > 0:
                    indices = np.linspace(0, total - 1, num=min(max_frames, total), dtype=int)
                else:
                    indices = list(range(total))
                frames = vr.get_batch(indices).asnumpy()
                return frames
            except Exception:
                pass

        try:
            import imageio.v3 as imageio  # type: ignore
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover - imageio optional
            imageio = None  # type: ignore

        if imageio is not None:
            try:
                frames_iter = imageio.imiter(video_path)
                frames = []
                for idx, frame in enumerate(frames_iter):
                    frames.append(frame)
                    if max_frames and max_frames > 0 and idx + 1 >= max_frames:
                        break
                if frames:
                    return np.stack(frames, axis=0)
            except Exception:
                return None

        return None

    @staticmethod
    def _coerce_vllm_video_payload(video_inputs: Any, default_nframes: int = 32) -> List[Dict[str, Any]]:
        """
        Normalize arbitrary "video inputs" into a vLLM-friendly payload that only
        references on-disk videos. Frames, tensors, and arrays are discarded so the
        downstream parser never receives inline pixel buffers (which vLLM rejects).
        """
        import os

        def _to_path_entry(value: Any) -> Optional[Dict[str, Any]]:
            if isinstance(value, dict):
                candidate = value.get("video") or value.get("video_path") or value.get("path")
            else:
                candidate = value

            if isinstance(candidate, (str, os.PathLike)):
                path_str = os.fspath(candidate)
                if path_str:
                    entry: Dict[str, Any] = {"video": path_str}
                    if isinstance(value, dict) and isinstance(value.get("nframes"), int):
                        entry["nframes"] = value["nframes"]
                    elif isinstance(default_nframes, int) and default_nframes > 0:
                        entry["nframes"] = default_nframes
                    return entry
            return None

        if video_inputs is None:
            return []

        if isinstance(video_inputs, list):
            coerced: List[Dict[str, Any]] = []
            for item in video_inputs:
                entry = _to_path_entry(item)
                if entry:
                    coerced.append(entry)
            return coerced

        entry = _to_path_entry(video_inputs)
        return [entry] if entry else []


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Log-likelihood computation is not implemented for Qwen3VL.")
