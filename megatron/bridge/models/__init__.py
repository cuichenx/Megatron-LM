# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Register only Super VL and its native Nemotron-H language bridge."""

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.nemotron_super_vl.bridge import Nemotron35SuperVLBridge
from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge
