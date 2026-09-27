"""The native stub's docs for the extension classes the public API exposes.

Editors read ``_hessboost.pyi``, while ``help()`` reads the extension's Rust
docs. A native class that a public module hands out as is (not wrapped in
Python) must carry the runtime docstrings in its stub, so the two cannot
drift.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import hessboost
from hessboost import _hessboost

STUB = Path(hessboost.__file__).with_name("_hessboost.pyi")


def _exposed_native_classes() -> dict[str, type]:
    """The extension's classes reachable from a public module."""
    exposed: dict[str, type] = {}
    modules = [hessboost] + [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(hessboost.__path__, "hessboost.")
        if not any(part.startswith("_") for part in info.name.split("."))
        # Needs scikit-learn (not installed on musl); its estimators wrap Booster.
        and info.name != "hessboost.sklearn"
    ]
    for module in modules:
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if isinstance(value, type) and getattr(_hessboost, value.__name__, None) is value:
                exposed[value.__name__] = value
    return exposed


def _stub_classes() -> dict[str, ast.ClassDef]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


EXPOSED = _exposed_native_classes()


def test_distributions_is_the_exposed_native_class() -> None:
    # Pins the set, so a newly exposed native class is added here knowingly
    # (and the parametrized checks below cannot pass vacuously).
    assert set(EXPOSED) == {"Distributions"}


@pytest.mark.parametrize("name", sorted(EXPOSED))
def test_stub_docs_match_runtime(name: str) -> None:
    cls = EXPOSED[name]
    stub = _stub_classes()[name]
    assert ast.get_docstring(stub) == inspect.cleandoc(cls.__doc__ or "")

    members = {
        node.name: node
        for node in stub.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    runtime = {n for n in vars(cls) if not n.startswith("_")}
    assert runtime == {n for n in members if not n.startswith("_")}
    for member, node in members.items():
        doc = ast.get_docstring(node)
        assert doc, f"{name}.{member} has no stub docstring"
        if member.startswith("__"):
            continue  # PyO3 replaces slot docs with CPython's ("Return len(self).")
        assert doc == inspect.cleandoc(getattr(cls, member).__doc__ or ""), f"{name}.{member}"
