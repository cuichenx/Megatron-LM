# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.sequential_mlp = MoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )
        self.router = self.sequential_mlp.router

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == 12 * 4, num_weights

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    def test_router_forward(self, moe_router_pre_softmax):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            # [num tokens, hidden size]
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda()
            scores, indices = self.router(hidden_states)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_aux_loss(self):
        self.sequential_mlp = self.sequential_mlp.cuda()

        # Without aux loss
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda()
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() == 0

        # With aux loss
        self.transformer_config.moe_aux_loss_coeff = 1
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

        # With Z loss
        self.transformer_config.moe_aux_loss_coeff = 0
        self.transformer_config.moe_z_loss_coeff = 1
        self.sequential_mlp.router.weight.grad.fill_(0)
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0


class TestGroupLimitedRouter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")

        num_moe_experts = 16
        self.transformer_config = TransformerConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
            num_moe_experts=num_moe_experts,
            moe_router_topk=4,
            moe_router_group_topk=2,
            moe_router_num_groups=8,
            moe_router_pre_softmax=True,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0,
            moe_token_dispatcher_type="alltoall",
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        # init MoE layer
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.moe_layer = MoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        ).cuda()
        self.router = self.moe_layer.router

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert (
            num_weights
            == self.transformer_config.hidden_size * self.transformer_config.num_moe_experts
        ), num_weights

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize(
        "moe_router_group_topk,moe_router_num_groups,moe_router_pre_softmax",
        [(3, 8, True), (3, 8, False), (2, 4, True), (2, 4, False)],
    )
    def test_router_forward(
        self, moe_router_group_topk, moe_router_num_groups, moe_router_pre_softmax
    ):
        with torch.no_grad():
            self.router.config.moe_router_group_topk = moe_router_group_topk
            self.router.config.moe_router_num_groups = moe_router_num_groups
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            if moe_router_pre_softmax:
                self.router.config.moe_router_topk_scaling_factor = 16.0

            seq_len = 2
            batch_size = 2
            num_tokens = seq_len * batch_size
            # hidden_states shape: [seq_len, batch_size, hidden_size]
            hidden_states = torch.randn(
                (seq_len, batch_size, self.router.config.hidden_size)
            ).cuda()
            scores, routing_map = self.router(hidden_states)
            assert scores.shape == (num_tokens, self.router.config.num_moe_experts), scores.shape
            assert routing_map.shape == (
                num_tokens,
                self.router.config.num_moe_experts,
            ), routing_map.shape

            group_routing_map = (
                routing_map.reshape(num_tokens, moe_router_num_groups, -1).max(dim=-1).values
            )
            assert torch.all(group_routing_map.sum(dim=-1) <= moe_router_group_topk)
