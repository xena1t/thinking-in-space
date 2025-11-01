"""Thin wrapper for evaluating Qwen2.5-VL models via the Qwen3VL implementation."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

from lmms_eval.api.registry import register_model

from .qwen3vl import Qwen3VL


@register_model("qwen25vl")
class Qwen25VL(Qwen3VL):
    """Expose Qwen/Qwen2.5-VL checkpoints using the Qwen3VL backend."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        repo_root: Optional[Union[str, os.PathLike]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("repo_env_var", "QWEN25_VL_REPO")
        kwargs.setdefault("repo_default", "Qwen2.5-VL")
        super().__init__(pretrained=pretrained, repo_root=repo_root, **kwargs)
