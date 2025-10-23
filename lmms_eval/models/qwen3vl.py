import multiprocessing as mp
import os
import sys
from typing import Dict, List, Optional, Tuple

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
            video_inputs = self._ensure_video_metadata(video_inputs)
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
    def _ensure_video_metadata(video_inputs):
        """Fill in missing metadata dictionaries expected by vLLM's Qwen3 backend.

        Qwen3's multimodal processor assumes every video payload includes a
        ``metadata`` mapping. Older revisions of ``qwen-vl-utils`` may return
        ``None`` for this field, which leads to ``AttributeError`` inside
        vLLM's sampling utilities.  To remain compatible with those versions we
        recursively replace ``None`` metadata entries with empty dictionaries
        and leave any existing metadata untouched.
        """

        if isinstance(video_inputs, dict):
            # Dictionaries that describe a single clip usually contain either a
            # ``video`` key or an explicit ``metadata`` field. In that case we
            # normalize the metadata in-place instead of recursing into each
            # scalar entry.
            if any(key in video_inputs for key in ("video", "image", "metadata")):
                normalized = dict(video_inputs)
                metadata = normalized.get("metadata")
                if metadata is None:
                    normalized["metadata"] = {}
                elif not isinstance(metadata, dict):
                    try:
                        normalized["metadata"] = dict(metadata)
                    except TypeError:
                        normalized["metadata"] = {}
                # Guarantee the key the backend probes exists.
                normalized["metadata"].setdefault("do_sample_frames", True)
                return normalized

            return {k: Qwen3VL._ensure_video_metadata(v) for k, v in video_inputs.items()}

        if isinstance(video_inputs, list):
            return [Qwen3VL._ensure_video_metadata(item) for item in video_inputs]

        if isinstance(video_inputs, tuple):
            return tuple(Qwen3VL._ensure_video_metadata(item) for item in video_inputs)

        return video_inputs

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Log-likelihood computation is not implemented for Qwen3VL.")
