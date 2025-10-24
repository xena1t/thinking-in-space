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
    def _coerce_vllm_video_payload(video_inputs: Any, default_nframes: int = 32) -> List[Dict[str, Any]]:
        """Return ``multi_modal_data['video']`` in the exact structure vLLM expects."""

        default_nframes = default_nframes or 32

        def _coerce_metadata(value: Any) -> Dict[str, Any]:
            if isinstance(value, Mapping):
                metadata = dict(value)
            else:
                try:
                    metadata = dict(value)  # type: ignore[arg-type]
                except Exception:
                    metadata = {}
            metadata.setdefault("do_sample_frames", True)
            return metadata

        def _coerce_video_value(value: Any) -> Any:
            """Convert tensors/arrays into numpy uint8 arrays accepted by vLLM."""
            try:  # torch tensors -> numpy
                import torch  # type: ignore
            except ImportError:  # pragma: no cover - torch optional
                torch = None  # type: ignore

            if torch is not None and isinstance(value, torch.Tensor):  # type: ignore[attr-defined]
                tensor = value.detach().cpu()
                if tensor.ndim == 4 and tensor.shape[1] in (1, 3, 4):
                    tensor = tensor.permute(0, 2, 3, 1).contiguous()
                elif tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                    tensor = tensor.permute(1, 2, 0).contiguous()
                if tensor.dtype != torch.uint8:  # type: ignore[attr-defined]
                    tensor = tensor.clamp(0, 255).to(torch.uint8)
                value = tensor.numpy()

            try:  # normalise numpy arrays as well
                import numpy as np  # type: ignore
            except ImportError:  # pragma: no cover - numpy optional
                np = None  # type: ignore

            if np is not None and isinstance(value, np.ndarray):  # type: ignore[attr-defined]
                array = value
                if array.ndim == 4 and array.shape[1] in (1, 3, 4):
                    array = array.transpose(0, 2, 3, 1)
                elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
                    array = array.transpose(1, 2, 0)
                if array.dtype != np.uint8:  # type: ignore[attr-defined]
                    array = np.clip(array, 0, 255).astype(np.uint8)
                value = array

            return value

        def _wrap_item(item: Any) -> Dict[str, Any]:
            if isinstance(item, (str, bytes, bytearray)):
                video_value = item
            elif isinstance(item, Mapping):
                entry = dict(item)
                if "video" not in entry and "path" in entry:
                    entry["video"] = entry.pop("path")
                entry["video"] = _coerce_video_value(entry.get("video"))
                entry["metadata"] = _coerce_metadata(entry.get("metadata", {}))
                entry.setdefault("nframes", default_nframes)
                return entry
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                seq_values = list(item)
                video_value = _coerce_video_value(seq_values[0] if seq_values else None)
                metadata_value = _coerce_metadata(seq_values[1] if len(seq_values) > 1 else {})
                return {
                    "video": video_value,
                    "metadata": metadata_value,
                    "nframes": default_nframes,
                }
            else:
                video_value = item

            return {
                "video": _coerce_video_value(video_value),
                "metadata": {"do_sample_frames": True},
                "nframes": default_nframes,
            }

        if video_inputs is None:
            return [
                {
                    "video": None,
                    "metadata": {"do_sample_frames": True},
                    "nframes": default_nframes,
                }
            ]

        if isinstance(video_inputs, (list, tuple)):
            return [_wrap_item(elem) for elem in video_inputs]

        if isinstance(video_inputs, Sequence) and not isinstance(video_inputs, (str, bytes, bytearray)):
            return [_wrap_item(elem) for elem in video_inputs]

        if isinstance(video_inputs, Mapping):
            return [_wrap_item(video_inputs)]

        return [_wrap_item(video_inputs)]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Log-likelihood computation is not implemented for Qwen3VL.")
