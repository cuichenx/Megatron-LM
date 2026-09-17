# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Strict, address-independent snapshot encoding and comparison."""

import dataclasses
import enum
import functools
import inspect
import math
from pathlib import Path
from typing import Any


def qualified_name(value: Any) -> str:
    """Return the import identity of a type or function."""
    return f"{value.__module__}.{value.__qualname__}"


def encode(value: Any, roots: dict[str, str] | None = None, references: dict[int, dict] | None = None) -> Any:
    """Encode supported semantic values; reject unsupported objects, never use repr."""
    roots = roots or {}
    references = references or {}
    if id(value) in references:
        return references[id(value)]
    if isinstance(value, enum.Enum):
        return {"enum": qualified_name(type(value)), "name": value.name}
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("Non-finite snapshot value")
        if math.isinf(value):
            return {"float": "+inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (str, Path)):
        result = str(value)
        for root, label in sorted(roots.items(), key=lambda item: -len(item[0])):
            if result == root or result.startswith(root + "/"):
                return label + result[len(root) :]
        return result
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": qualified_name(type(value)),
            "fields": {
                field.name: (
                    encode(getattr(value, field.name), roots, references)
                    if hasattr(value, field.name)
                    else {"missing_attribute": True}
                )
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: encode(item, roots, references) for key, item in sorted(value.items())}
        pairs = [[encode(key, roots, references), encode(item, roots, references)] for key, item in value.items()]
        return {"mapping": pairs}
    if isinstance(value, (tuple, list)):
        items = [encode(item, roots, references) for item in value]
        return {"tuple": items} if isinstance(value, tuple) else items
    if isinstance(value, functools.partial):
        return {
            "partial": encode(value.func, roots, references),
            "args": encode(value.args, roots, references),
            "kwargs": encode(value.keywords, roots, references),
        }
    if inspect.ismethod(value):
        owner = value.__self__ if isinstance(value.__self__, type) else type(value.__self__)
        return {"bound_method": qualified_name(value.__func__), "owner_type": qualified_name(owner)}
    if inspect.isfunction(value):
        return {
            "callable": qualified_name(value),
            "defaults": encode(value.__defaults__, roots, references),
            "closure": [encode(cell.cell_contents, roots, references) for cell in (value.__closure__ or ())],
        }
    if inspect.isbuiltin(value) or isinstance(value, type):
        return {"callable": qualified_name(value)}
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return {type(value).__name__: str(value)}
    if qualified_name(type(value)) == "megatron.core.timers.Timers":
        return {"type": qualified_name(type(value)), "log_level": value._log_level, "log_option": value._log_option}
    if type(value).__module__.startswith("torch.distributed") and type(value).__name__ == "ProcessGroup":
        from torch import distributed

        return {
            "process_group_ranks": distributed.get_process_group_ranks(value),
            "backend": str(distributed.get_backend(value)),
        }
    if type(value).__module__.startswith("megatron.core.tokenizers"):
        metadata = {"type": qualified_name(type(value)), "vocab_size": value.vocab_size}
        for key in ("eod", "pad", "bos", "eos", "unk"):
            try:
                metadata[key] = getattr(value, key)
            except (AttributeError, NotImplementedError):
                metadata[key] = {"unavailable": True}
        return encode(metadata, roots, references)
    raise TypeError(f"Unsupported snapshot type: {qualified_name(type(value))}")


def differences(left: Any, right: Any, path: str = "$") -> list[dict]:
    """Return every field difference, preserving types and absent versus null."""
    if type(left) is not type(right):
        return [{"path": path, "baseline": left, "candidate": right}]
    result = []
    if isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                result.append({"path": f"{path}.{key}", "missing": "baseline" if key not in left else "candidate"})
            else:
                result.extend(differences(left[key], right[key], f"{path}.{key}"))
    elif isinstance(left, list):
        if len(left) != len(right):
            result.append({"path": path + ".length", "baseline": len(left), "candidate": len(right)})
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(differences(a, b, f"{path}[{index}]"))
    elif left != right:
        result.append({"path": path, "baseline": left, "candidate": right})
    return result
