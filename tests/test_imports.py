"""Test that all modules can be imported without side-effects."""

import importlib
import pathlib
import pkgutil

import src.arc_dslearn as arc_dslearn


def test_all_modules_importable() -> None:
    """Smoke-test that every module under src/ can be imported without side-effects."""
    root = pathlib.Path(arc_dslearn.__file__).parent
    for m in pkgutil.walk_packages([str(root)]):
        importlib.import_module(f"{arc_dslearn.__name__}.{m.name}")
