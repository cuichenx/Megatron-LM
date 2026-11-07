# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental Super-VL conversion slice hosted by Megatron-LM."""

from megatron.training.config.instantiate_utils import target_allowlist

# These configs are local package code, just like megatron.training/models.
target_allowlist.add_prefix("megatron.bridge.")

from megatron.bridge.models import AutoBridge

__all__ = ["AutoBridge"]
