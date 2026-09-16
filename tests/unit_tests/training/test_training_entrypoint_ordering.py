"""Guard config-first wiring while legacy callers retain their startup order."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINTS = (
    "pretrain_gpt.py",
    "pretrain_hybrid.py",
    "pretrain_vlm.py",
    "examples/bert/pretrain_bert.py",
    "examples/t5/pretrain_t5.py",
    "examples/mimo/train.py",
    "examples/post_training/modelopt/finetune.py",
)


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_entrypoint_defers_services_until_pretrain(entrypoint: str):
    tree = ast.parse((ROOT / entrypoint).read_text())
    parse_calls = _calls(tree, "parse_and_validate_args")
    assert len(parse_calls) == 1
    kwargs = {keyword.arg: keyword.value for keyword in parse_calls[0].keywords}
    assert ast.literal_eval(kwargs["initialize_globals"]) is False
    pretrain_calls = _calls(tree, "pretrain")
    assert pretrain_calls
    assert all(call.lineno > parse_calls[0].lineno for call in pretrain_calls)


@pytest.mark.parametrize(
    "entrypoint,adapter",
    [
        ("pretrain_gpt.py", "gpt_config_from_args"),
        ("pretrain_hybrid.py", "hybrid_config_from_args"),
    ],
)
def test_model_adapter_defers_tokenizer_dependent_vocabulary(entrypoint: str, adapter: str):
    calls = _calls(ast.parse((ROOT / entrypoint).read_text()), adapter)
    assert calls
    for call in calls:
        kwargs = {keyword.arg: keyword.value for keyword in call.keywords}
        assert ast.literal_eval(kwargs["defer_vocab_size"]) is True


def test_pretrain_registers_config_before_distributed_startup():
    tree = ast.parse((ROOT / "megatron/training/training.py").read_text())
    pretrain = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pretrain"
    )
    globals_calls = _calls(pretrain, "initialize_training_globals")
    distributed_calls = _calls(pretrain, "initialize_megatron")
    assert len(globals_calls) == 1
    assert distributed_calls
    assert globals_calls[0].lineno < min(call.lineno for call in distributed_calls)
    assert isinstance(globals_calls[0].args[0], ast.Name)
    assert globals_calls[0].args[0].id == "cfg_container"
