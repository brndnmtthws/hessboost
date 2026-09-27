"""Model file I/O and the pickle state shared by the model classes."""

from __future__ import annotations

import os
from typing import Any, TypeAlias

from hessboost import _data

PathLike: TypeAlias = str | os.PathLike[str]


def read_bytes(path: PathLike) -> bytes:
    """The contents of the file ``path``."""
    with open(path, "rb") as file:
        return file.read()


def write_bytes(path: PathLike, data: bytes) -> None:
    """Writes ``data`` to the file ``path``."""
    with open(path, "wb") as file:
        file.write(data)


def read_text(path: PathLike) -> str:
    """The UTF-8 text of the file ``path``."""
    with open(path, encoding="utf-8") as file:
        return file.read()


def write_text(path: PathLike, text: str) -> None:
    """Writes ``text`` to the file ``path`` as UTF-8."""
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


class _SchemaState:
    """The feature schema a model's wrapper records (names, types and
    pandas categories, which model files do not store), and its pickle
    state: the model's native bytes (:meth:`_model_state`, restored by
    :meth:`_restore_model`) with the schema."""

    _feature_names: list[str] | None
    _feature_types: list[str] | None
    _categories: _data.Categories

    def _set_schema(
        self,
        feature_names: list[str] | None,
        feature_types: list[str] | None,
        categories: _data.Categories,
    ) -> None:
        self._feature_names = feature_names
        self._feature_types = feature_types
        self._categories = categories

    def _model_state(self) -> bytes | None:
        """The model's native bytes (``None`` for no model)."""
        raise NotImplementedError

    def _restore_model(self, model: Any) -> None:
        """Replaces the model with the one :meth:`_model_state` returned."""
        raise NotImplementedError

    def __getstate__(self) -> dict[str, object]:
        return {
            "model": self._model_state(),
            "feature_names": self._feature_names,
            "feature_types": self._feature_types,
            "categories": self._categories,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._restore_model(state["model"])
        self._set_schema(state["feature_names"], state["feature_types"], state["categories"])
