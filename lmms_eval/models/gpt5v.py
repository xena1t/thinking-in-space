import base64
import hashlib
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Tuple

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:  # pragma: no cover - optional dependency
    VideoReader = None
    cpu = None

from PIL import Image

API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = int(os.getenv("RETRY_SLEEP_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("RETRY_MAX_ATTEMPTS", "5"))

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.cognitive.microsoft.com")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt5-vision")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    API_URL = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }
else:  # pragma: no cover - validated at runtime
    raise ValueError(f"Unsupported API_TYPE: {API_TYPE}")

if not os.path.isdir("gpt_cache"):
    os.mkdir("gpt_cache")


def calculate_sha256(input_string: str) -> str:
    return hashlib.sha256(input_string.encode()).hexdigest()


@register_model("gpt5v")
class GPT5V(lmms):
    def __init__(
        self,
        model_version: str = os.getenv("OPENAI_GPT5V_MODEL", "gpt-5"),
        modality: str = "video",
        max_frames_num: int = 10,
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str | None = None,
        **_: dict,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.image_token = "<image>"
        self.timeout = timeout
        self.continual_mode = continual_mode

        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError(
                    "Continual mode requires a persistent path for the response. "
                    "Please provide a valid path."
                )
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, f"{self.model_version}_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        cache_file = f"gpt_cache/{self.model_version}_md5_to_responses.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as cache_fp:
                self.md5_to_responses = json.load(cache_fp)
        else:
            self.md5_to_responses = {}
        self._cache_file = cache_file

        self.device = self.accelerator.device

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        return base64.b64encode(output_buffer.getvalue()).decode("utf-8")

    def encode_image(self, image: Image.Image | str) -> str:
        if isinstance(image, Image.Image):
            return self._image_to_base64(image)
        with Image.open(image) as img:
            return self._image_to_base64(img)

    def encode_video(self, video_path: str, for_get_frames_num: int) -> List[str]:
        if VideoReader is None:
            raise ImportError("decord is required for video inputs but is not installed")
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        if total_frame_num == 0:
            return []
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
        if (total_frame_num - 1) not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)
        frames = vr.get_batch(uniform_sampled_frames.tolist()).asnumpy()
        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            base64_frames.append(self._image_to_base64(img))
        return base64_frames

    @staticmethod
    def flatten(input_list):
        return [j for i in input_list for j in i]

    def generate_until(self, requests) -> List[str]:
        responses: List[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode and getattr(self, "cache_mode", None) == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    cached_response = self.response_cache[doc_uuid]
                    if cached_response:
                        responses.append(cached_response)
                        pbar.update(1)
                        continue

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            imgs: List[str] = []
            for visual in visuals:
                if self.modality == "image":
                    imgs.append(self.encode_image(visual))
                elif self.modality == "video":
                    imgs.extend(self.encode_video(visual, self.max_frames_num))

            payload = {"messages": []}
            payload["model"] = self.model_version

            response_json = {"role": "user", "content": []}
            if self.image_token not in contexts:
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][0]["content"].append({"type": "text", "text": contexts})
                for img in imgs:
                    payload["messages"][0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    })
            else:
                parts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    payload["messages"].append(deepcopy(response_json))
                    text_part = parts[idx] if idx < len(parts) else ""
                    if text_part:
                        payload["messages"][idx]["content"].append({"type": "text", "text": text_part})
                    payload["messages"][idx]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    })
                if len(parts) > len(imgs):
                    payload["messages"].append(deepcopy(response_json))
                    trailing_text = parts[-1]
                    if trailing_text:
                        payload["messages"][-1]["content"].append({"type": "text", "text": trailing_text})

            gen_kwargs = dict(gen_kwargs)
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]

            metainfo = {
                "context": contexts,
                "visuals": visuals,
                "max_frames_num": self.max_frames_num,
                "gen_kwargs": gen_kwargs,
                "model_version": self.model_version,
            }
            metainfo_hash = calculate_sha256(json.dumps(metainfo, sort_keys=True))

            if self.md5_to_responses.get(metainfo_hash):
                responses.append(self.md5_to_responses[metainfo_hash])
                pbar.update(1)
                continue

            response_text = ""
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = url_requests.post(
                        API_URL, headers=headers, json=payload, timeout=self.timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    response_text = data["choices"][0]["message"]["content"].strip()
                    break
                except Exception as error:  # pragma: no cover - network I/O
                    try:
                        error_msg = response.json() if "response" in locals() else ""
                    except Exception:  # pragma: no cover - best effort logging
                        error_msg = ""
                    eval_logger.info(
                        f"[Attempt {attempt}/{MAX_RETRIES}] Error: {error}.\nResponse: {error_msg}"
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:
                        eval_logger.error(
                            f"All {MAX_RETRIES} attempts failed. Last error: {error}."
                        )
                        response_text = ""

            responses.append(response_text)
            if response_text:
                self.md5_to_responses[metainfo_hash] = response_text
                with open(self._cache_file, "w") as cache_fp:
                    json.dump(self.md5_to_responses, cache_fp)

            if self.continual_mode:
                doc_uuid = f"{task}___{split}___{doc_id}"
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as cache_fp:
                    json.dump(self.response_cache, cache_fp)

            pbar.update(1)

        pbar.close()
        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("GPT5V does not support loglikelihood")
