"""Tests for the unification probe's own correctness.

The probe exists to answer one question empirically, and its first run answered
it **wrongly** — reporting "the unified stack is not viable" for two reasons that
were both bugs in the probe:

1. It listed `transformers.models.bart.modeling_bart.BartDecoder` as a module and
   handed it to `importlib.import_module`, which takes a *module*. `BartDecoder`
   is a class. That raised `ModuleNotFoundError: 'modeling_bart' is not a
   package` in every environment at every version — a permanent false negative.
2. The Dockerfile ran it with `--offline` in an image where nothing is cached, so
   it could never fetch SheetSage2's code at all.

Both produced the same printed verdict, and it was not evidence about anything.
These tests pin the properties that make the probe's answer meaningful, because
a probe that lies is worse than no probe: it converts "unknown" into "no".
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE = REPO_ROOT / "worker" / "transcribe_sheetsage" / "probe_unified_stack.py"
DOCKERFILE = REPO_ROOT / "Dockerfile"

sys.path.insert(0, str(REPO_ROOT / "worker" / "transcribe_sheetsage"))


@pytest.fixture(scope="module")
def probe_module():
    spec = importlib.util.spec_from_file_location("probe_unified_stack", PROBE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# The bug that made the verdict meaningless
# =============================================================================


def test_every_probe_target_is_a_module_not_a_class(probe_module) -> None:
    """Each probe's first element must name a module importable on its own.

    This is the defect that produced the false verdict. `import_module` takes a
    module path; a dotted class name fails, permanently, everywhere. Asserting
    the *shape* is not enough — `X.Y` looks identical whether `Y` is a module or
    a class — so each is actually imported here, in an environment where
    transformers is absent, and the failure has to be "not installed" rather than
    "not a package".
    """
    for module_path, _attribute, label in probe_module.PROBES:
        try:
            importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            message = str(exc)
            assert "is not a package" not in message, (
                f"{label}: {module_path!r} is not a module — import_module cannot take a class. {message}"
            )
            # A missing third-party package is the expected failure here.
            assert any(name in message for name in ("torch", "transformers", "numpy", "huggingface", "safetensors")), (
                f"{label}: unexpected import failure: {message}"
            )


def test_a_named_attribute_is_checked_not_just_the_module(probe_module) -> None:
    """`import X` succeeding does not mean `X.Class` still exists.

    SheetSage2 imports `BartDecoder` *from* that module. The module surviving a
    transformers upgrade is not the same claim, and only the second one matters.
    """
    with_attribute = [(m, a, label) for m, a, label in probe_module.PROBES if a is not None]
    assert with_attribute, "no probe checks an attribute — the class existence is unverified"
    for module_path, attribute, label in with_attribute:
        assert attribute, f"{label}: empty attribute name"
        assert hasattr(probe_module, "probe_imports")


def test_the_transformer_internal_is_probed_as_module_plus_attribute(probe_module) -> None:
    """Pins the specific fix, so the old form cannot come back."""
    targets = {label: (module, attribute) for module, attribute, label in probe_module.PROBES}
    internals = [(m, a) for label, (m, a) in targets.items() if "internal" in label]
    assert internals, "the transformers internal is no longer probed at all"
    for module_path, attribute in internals:
        assert module_path.count(".") >= 2, module_path
        assert "." not in (attribute or ""), "the attribute must be a bare name, not a dotted path"
        assert module_path.endswith("modeling_bart")
        assert attribute == "BartDecoder"


def test_probe_imports_reports_a_missing_attribute(probe_module, monkeypatch) -> None:
    """A module that imports but lacks the attribute must be a failure."""
    import types

    fake = types.ModuleType("pretend_module_without_the_class")
    monkeypatch.setattr(probe_module, "PROBES", (("sys", "definitely_not_an_attribute", "synthetic"),))
    monkeypatch.setitem(sys.modules, "pretend_module_without_the_class", fake)

    results = probe_module.probe_imports()
    assert len(results) == 1
    label, ok, detail = results[0]
    assert not ok
    assert "no attribute" in detail


def test_probe_imports_reports_success_for_a_real_attribute(probe_module, monkeypatch) -> None:
    monkeypatch.setattr(probe_module, "PROBES", (("sys", "version", "synthetic"),))
    results = probe_module.probe_imports()
    assert results == [("synthetic", True, "")]


# =============================================================================
# The bug that stopped it running at all
# =============================================================================


def test_the_dockerfile_does_not_run_the_probe_offline() -> None:
    """`--offline` in an image with nothing cached means it can never fetch.

    The probe reported "could not fetch SheetSage2 code: LocalEntryNotFoundError"
    and then a verdict, and the verdict was treated as an answer. It was not.
    """
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        if "probe_unified_stack.py" in line and line.strip().startswith("RUN"):
            assert "--offline" not in line, (
                "the probe runs offline, so it cannot fetch SheetSage2's code and its verdict is meaningless"
            )


def test_the_probe_is_still_non_fatal() -> None:
    """It reports; it does not gate.

    The split stack is what ships, so a failure is information. Making it fatal
    would produce a red build that says nothing about whether the product works.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    probe_lines = [ln for ln in text.splitlines() if "probe_unified_stack.py" in ln and ln.strip().startswith("RUN")]
    assert probe_lines, "the probe is no longer run at build time"
    assert any("||" in ln for ln in probe_lines), "the probe gates the build; it should only report"


def test_the_probe_does_not_fetch_weights() -> None:
    """Only the Python source — a few hundred KB, never the multi-GB checkpoints.

    The image's weight guard would catch a checkpoint landing, but the probe
    should not be trying in the first place.
    """
    source = PROBE.read_text(encoding="utf-8")
    assert "allow_patterns" in source, "the probe downloads without a pattern filter"
    match = re.search(r"allow_patterns=\[(.*?)\]", source, re.S)
    assert match, "could not read the allow_patterns list"
    patterns = match.group(1)
    assert "*.py" in patterns
    assert "*.safetensors" not in patterns
    assert "*.bin" not in patterns


# =============================================================================
# The verdict must be readable, and must say what it does not know
# =============================================================================


def test_the_probe_states_its_limits(probe_module) -> None:
    """A pass means "the imports resolve", not "the model is correct".

    Numerical correctness needs weights and a GPU. The probe's docstring and its
    success output both have to say so, or a green result will be read as more
    than it is.
    """
    source = PROBE.read_text(encoding="utf-8")
    assert "cannot verify numerical output" in source or "Numerical correctness" in source
    assert "weights" in source and "GPU" in source


def test_the_verdict_is_machine_readable(probe_module, capsys, monkeypatch) -> None:
    """A JSON trailer so a build step or a test can parse the outcome."""
    monkeypatch.setattr(probe_module, "probe_imports", lambda: [("synthetic", True, "")])
    monkeypatch.setattr(probe_module, "probe_sheetsage2", lambda repo, offline: (True, "synthetic ok"))
    monkeypatch.setattr(probe_module, "versions", lambda: {"torch": "2.10.0+cu128"})

    code = probe_module.main([])
    out = capsys.readouterr().out
    assert code == 0
    trailer = [ln for ln in out.splitlines() if ln.strip().startswith("{")][-1]
    import json

    payload = json.loads(trailer)
    assert payload["unified_stack_ok"] is True
    assert payload["versions"]["torch"] == "2.10.0+cu128"


def test_the_verdict_is_false_when_an_import_fails(probe_module, capsys, monkeypatch) -> None:
    """And it must be able to say no — for a real reason."""
    monkeypatch.setattr(probe_module, "probe_imports", lambda: [("transformers internal", False, "boom")])
    monkeypatch.setattr(probe_module, "probe_sheetsage2", lambda repo, offline: (True, "ok"))
    monkeypatch.setattr(probe_module, "versions", lambda: {})

    assert probe_module.main([]) == 1
    assert "do not unify" in capsys.readouterr().out


def test_the_probe_is_valid_python_and_imports_are_static() -> None:
    """It runs inside the *main* environment, not a venv, so it must not import
    anything the main environment lacks."""
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    stdlib = {"__future__", "argparse", "json", "sys", "traceback", "pathlib", "importlib", "dataclasses", "metadata"}
    third_party = {"huggingface_hub", "transformers"}
    unexpected = imported - stdlib - third_party
    assert not unexpected, f"unexpected imports: {sorted(unexpected)}"
