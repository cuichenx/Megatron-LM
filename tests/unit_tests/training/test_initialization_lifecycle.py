"""Tests for config construction before training runtime initialization."""

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer import TransformerConfig
from megatron.training import arguments, global_vars
from megatron.training.argument_utils import gpt_config_from_args, hybrid_config_from_args
from megatron.training.config.training_config import TokenizerConfig


@pytest.fixture
def isolated_globals(monkeypatch):
    """Avoid changing services owned by the distributed test harness."""
    for name in ("_GLOBAL_ARGS", "_GLOBAL_CFG", "_GLOBAL_TOKENIZER"):
        monkeypatch.setattr(global_vars, name, None)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUNTIME_INITIALIZED", False)


def _runtime_args():
    return Namespace(
        rank=0,
        global_batch_size=8,
        micro_batch_size=2,
        data_parallel_size=2,
        gtp_weight_remat_size=1,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        seq_length=32,
        padded_vocab_size=None,
        enable_experimental=False,
        exit_signal_handler=False,
        exit_signal_handler_for_training=False,
        disable_jit_fuser=False,
    )


@pytest.mark.parametrize("initialize_globals", [False, True])
def test_parse_selects_explicit_or_legacy_bootstrap(monkeypatch, initialize_globals):
    args = Namespace(use_checkpoint_args=False, yaml_cfg=None, enable_experimental=False)
    monkeypatch.setattr(arguments, "parse_args", Mock(return_value=args))
    validate = Mock()
    legacy = Mock()
    monkeypatch.setattr(arguments, "validate_args", validate)
    monkeypatch.setattr(arguments, "set_global_variables", legacy)

    assert arguments.parse_and_validate_args(initialize_globals=initialize_globals) is args
    validate.assert_called_once_with(args, {})
    if initialize_globals:
        legacy.assert_called_once_with(args)
    else:
        legacy.assert_called_once_with(args, initialize_runtime=False)


def test_default_parse_preserves_legacy_bootstrap(monkeypatch):
    args = Namespace(use_checkpoint_args=False, yaml_cfg=None, enable_experimental=False)
    monkeypatch.setattr(arguments, "parse_args", Mock(return_value=args))
    monkeypatch.setattr(arguments, "validate_args", Mock())
    legacy = Mock()
    monkeypatch.setattr(arguments, "set_global_variables", legacy)

    assert arguments.parse_and_validate_args() is args
    legacy.assert_called_once_with(args)


def test_deferred_bootstrap_only_registers_args(monkeypatch, isolated_globals):
    args = _runtime_args()
    initialize = Mock()
    monkeypatch.setattr(global_vars, "_initialize_runtime_services", initialize)
    global_vars.set_global_variables(args, initialize_runtime=False)
    assert global_vars.get_args() is args
    initialize.assert_not_called()
    with pytest.raises(AssertionError, match="already initialized"):
        global_vars.set_global_variables(args, initialize_runtime=False)


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("checkpoint_vocab", [None, 256])
def test_vocabulary_resolves_after_config_construction(
    monkeypatch, isolated_globals, adapter, checkpoint_vocab
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.padded_vocab_size = checkpoint_vocab
    args.vocab_size = None
    model_cfg = adapter(
        args,
        config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        defer_vocab_size=True,
    )
    assert model_cfg.vocab_size == checkpoint_vocab
    cfg = SimpleNamespace(model=model_cfg, tokenizer=TokenizerConfig())
    global_vars.set_args(args)

    def initialize_services(received_args):
        # The full container has already been registered when tokenizer work starts.
        assert global_vars.get_cfg() is cfg
        assert received_args is args
        if received_args.padded_vocab_size is None:
            received_args.padded_vocab_size = 128
        monkeypatch.setattr(global_vars, "_GLOBAL_RUNTIME_INITIALIZED", True)

    initialize = Mock(side_effect=initialize_services)
    monkeypatch.setattr(global_vars, "_initialize_runtime_services", initialize)
    global_vars.initialize_training_globals(cfg)

    assert model_cfg.vocab_size == (128 if checkpoint_vocab is None else checkpoint_vocab)
    assert model_cfg.should_pad_vocab is False
    assert cfg.tokenizer.padded_vocab_size == model_cfg.vocab_size
    initialize.assert_called_once_with(args)

    global_vars.initialize_training_globals(cfg)
    initialize.assert_called_once_with(args)


def test_runtime_service_order_and_microbatch_inputs(monkeypatch, isolated_globals):
    args = _runtime_args()
    cfg = SimpleNamespace(model=None, tokenizer=TokenizerConfig())
    global_vars.set_args(args)
    calls = []
    microbatches = Mock(side_effect=lambda **kwargs: calls.append("microbatches"))
    monkeypatch.setattr(global_vars, "init_num_microbatches_calculator", microbatches)

    for name in (
        "_build_tokenizer",
        "_set_tensorboard_writer",
        "_set_wandb_writer",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_timers",
        "_set_energy_monitor",
        "_set_telemetry",
    ):

        def record(received_args, service=name):
            assert received_args is args
            assert global_vars.get_cfg() is cfg
            calls.append(service)

        monkeypatch.setattr(global_vars, name, record)

    global_vars.initialize_training_globals(cfg)
    assert calls == [
        "microbatches",
        "_build_tokenizer",
        "_set_tensorboard_writer",
        "_set_wandb_writer",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_timers",
        "_set_energy_monitor",
        "_set_telemetry",
    ]
    microbatches.assert_called_once_with(
        rank=0,
        global_batch_size=8,
        micro_batch_size=2,
        data_parallel_size=2,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        seq_length=32,
    )


def test_legacy_services_are_not_constructed_again(monkeypatch, isolated_globals):
    global_vars.set_args(_runtime_args())
    monkeypatch.setattr(global_vars, "_GLOBAL_RUNTIME_INITIALIZED", True)
    initialize = Mock()
    monkeypatch.setattr(global_vars, "_initialize_runtime_services", initialize)
    cfg = SimpleNamespace(model=None, tokenizer=TokenizerConfig())
    global_vars.initialize_training_globals(cfg)
    assert global_vars.get_cfg() is cfg
    initialize.assert_not_called()


def test_custom_model_config_does_not_receive_vocabulary(monkeypatch, isolated_globals):
    args = _runtime_args()
    args.padded_vocab_size = 128
    global_vars.set_args(args)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUNTIME_INITIALIZED", True)
    custom_model = SimpleNamespace()
    global_vars.initialize_training_globals(
        SimpleNamespace(model=custom_model, tokenizer=TokenizerConfig())
    )
    assert vars(custom_model) == {}


@pytest.mark.parametrize("adapter", [gpt_config_from_args, hybrid_config_from_args])
@pytest.mark.parametrize("pad_vocab,checkpoint_vocab", [(True, None), (True, 512), (False, None)])
def test_config_first_matches_legacy_tokenizer_and_model_inputs(
    monkeypatch, isolated_globals, adapter, pad_vocab, checkpoint_vocab
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 133
    args.pad_vocab_size = pad_vocab
    args.padded_vocab_size = checkpoint_vocab
    args.tensor_model_parallel_size = 2
    transformer = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)

    legacy_args = deepcopy(args)
    legacy_tokenizer = build_tokenizer(legacy_args)
    legacy_model = adapter(legacy_args, config=deepcopy(transformer))

    model = adapter(args, config=deepcopy(transformer), defer_vocab_size=True)
    cfg = SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    global_vars.set_args(args)

    def initialize(received_args):
        global_vars._build_tokenizer(received_args)
        monkeypatch.setattr(global_vars, "_GLOBAL_RUNTIME_INITIALIZED", True)

    monkeypatch.setattr(global_vars, "_initialize_runtime_services", initialize)
    global_vars.initialize_training_globals(cfg)
    assert model.as_dict() == legacy_model.as_dict()
    assert cfg.tokenizer.padded_vocab_size == legacy_args.padded_vocab_size
    assert global_vars.get_tokenizer().vocab_size == legacy_tokenizer.vocab_size


def test_explicit_model_vocabulary_is_not_overwritten(monkeypatch, isolated_globals):
    from megatron.training.models import GPTModelConfig

    args = _runtime_args()
    args.padded_vocab_size = 128
    global_vars.set_args(args)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUNTIME_INITIALIZED", True)
    model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
        vocab_size=512,
        should_pad_vocab=False,
    )
    global_vars.initialize_training_globals(
        SimpleNamespace(model=model, tokenizer=TokenizerConfig())
    )
    assert model.vocab_size == 512
    assert model.should_pad_vocab is False


@pytest.mark.parametrize("cleanup", ["unset_global_variables", "destroy_global_vars"])
def test_config_cleanup_allows_new_run(monkeypatch, isolated_globals, cleanup):
    # Preserve the distributed test session's other services across this test.
    for name, value in vars(global_vars).copy().items():
        if name.startswith("_GLOBAL_"):
            monkeypatch.setattr(global_vars, name, value)
    clear_calculator = Mock()
    monkeypatch.setattr(global_vars, "unset_num_microbatches_calculator", clear_calculator)
    first = SimpleNamespace(model=None)
    second = SimpleNamespace(model=None)
    global_vars.set_cfg(first)
    with pytest.raises(AssertionError, match="already initialized"):
        global_vars.set_cfg(second)

    getattr(global_vars, cleanup)()
    if cleanup == "unset_global_variables":
        clear_calculator.assert_called_once_with()
    with pytest.raises(AssertionError, match="not initialized"):
        global_vars.get_cfg()
    assert not global_vars._GLOBAL_RUNTIME_INITIALIZED
    global_vars.set_cfg(second)
    assert global_vars.get_cfg() is second
