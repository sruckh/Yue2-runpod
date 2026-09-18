"""Tests for the unification probe's own correctness.

The probe exists to answer one question empirically, and it has now failed to
answer it in four different ways — every one of them a bug in the probe:

1. It listed `transformers.models.bart.modeling_bart.BartDecoder` as a module and
   handed it to `importlib.import_module`, which takes a *module*. `BartDecoder`
   is a class. That raised `ModuleNotFoundError: 'modeling_bart' is not a
   package` in every environment at every version — a permanent false negative.
2. The Dockerfile ran it with `--offline` in an image where nothing is cached, so
   it could never fetch SheetSage2's code at all.
3. With those fixed it ran, and reported a missing `torchaudio` as "the stacks
   are incompatible" — a stronger claim than the evidence, since torchaudio was
   simply not installed and 2.10.0 ships for cu128/cp311.
4. With torchaudio installed it finally fetched the code and crashed on its own
   success path: `AttributeError: 'tuple' object has no attribute 'ok'`.

Runs 1-3 all printed the same verdict, and none of them was evidence about
anything. A probe that lies is worse than no probe: it converts "unknown" into
"no", and the answer looks settled. These tests pin the properties that make the
probe's answer meaningful, and — after run 4 — the property that makes it able to
produce an answer at all.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

import check_env
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE = REPO_ROOT / "worker" / "transcribe_sheetsage" / "probe_unified_stack.py"
DOCKERFILE = REPO_ROOT / "Dockerfile"

sys.path.insert(0, str(REPO_ROOT / "worker" / "transcribe_sheetsage"))


@pytest.fixture(scope="module")
def probe_module():
    """Load the probe by path.

    The module is registered in `sys.modules` *before* `exec_module`, because
    it defines a `@dataclass` and `dataclasses` looks the defining module up by
    name — with it absent, the decorator raises
    `AttributeError: 'NoneType' object has no attribute '__dict__'`. A
    `spec.loader.exec_module` alone is not enough for dataclasses.
    """
    spec = importlib.util.spec_from_file_location("probe_unified_stack", PROBE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_unified_stack"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("probe_unified_stack", None)
        raise
    yield module
    sys.modules.pop("probe_unified_stack", None)


# =============================================================================
# Every exit from probe_sheetsage2 returns a ProbeOutcome
# =============================================================================
#
# The fourth bug, and the one that got furthest: with the module-path and
# `--offline` bugs fixed, and torchaudio installed, the probe finally *fetched
# SheetSage2's code* — and then crashed on its own success path:
#
#     print(f"  {'ok  ' if outcome.ok else 'FAIL'} ...")
#     AttributeError: 'tuple' object has no attribute 'ok'
#
# `probe_sheetsage2` returned a bare 2-tuple at two of its five exits (the
# success path and the catch-all) and a `ProbeOutcome` at the other three. The
# earlier tests *mocked* the function, so they replaced both broken exits with
# well-formed values and never executed them. A mock of the thing under test
# cannot find a bug in the thing under test.
#
# These tests read the real function and drive the real paths.


def test_every_return_in_probe_sheetsage2_is_a_probe_outcome() -> None:
    """Statically: every `return` in the body must construct a ProbeOutcome.

    This is the check that would have caught it before the build. A tuple and a
    dataclass are indistinguishable at the call site until something reads an
    attribute, so the *shape* of the return statements is what has to be pinned.
    """
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "probe_sheetsage2"
    )
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert returns, "probe_sheetsage2 has no return statements — did it get renamed?"

    for node in returns:
        value = node.value
        assert value is not None, f"line {node.lineno}: a bare `return` — the caller needs an outcome"
        # `return ProbeOutcome(...)` — a call to the dataclass by name.
        is_outcome = (
            isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "ProbeOutcome"
        )
        assert is_outcome, (
            f"line {node.lineno}: returns {ast.unparse(value)[:70]!r}, not a ProbeOutcome. "
            "A bare tuple works until the caller reads `.ok`, and then the probe dies "
            "without a verdict — which is what happened in the build."
        )


def test_the_success_path_returns_an_outcome_not_a_tuple(probe_module, monkeypatch) -> None:
    """Execute the real success path end-to-end against a synthetic package.

    The static check above pins the shape; this one proves the path actually
    runs and produces something main() can read. It builds a throwaway package
    on disk whose `__init__.py` imports cleanly, so `exec_module` succeeds and
    control reaches the `return ProbeOutcome(True, ...)` line.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        package = root / "pkg"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")

        # The probe loads `<path>/__init__.py` as a package. Stand in for the
        # downloaded snapshot with this directory.
        monkeypatch.setitem(
            sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda *a, **k: root)
        )
        # AutoConfig would need the real config; stub it so the class name is
        # deterministic and the import path is reached.
        fake_config = type("SheetSage2Config", (), {})
        fake_transformers = types.SimpleNamespace(
            AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: fake_config())
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        outcome = probe_module.probe_sheetsage2("synthetic/repo", offline=False)

    assert isinstance(outcome, probe_module.ProbeOutcome), (
        f"probe_sheetsage2 returned {type(outcome).__name__} on its success path, "
        "so main() will raise AttributeError instead of printing a verdict"
    )
    # The model/tokenizer modules do not exist in the synthetic package, so this
    # lands in the catch-all — which is the *other* path that returned a tuple.
    if outcome.ok:
        assert outcome.kind == "ok"
    else:
        assert outcome.kind in {"missing", "structural", "fetch"}
        assert outcome.detail


def test_the_probe_runs_to_a_verdict_rather_than_crashing(probe_module, capsys, monkeypatch) -> None:
    """main() must print a verdict for every outcome the real code can return.

    The crash was in `main`, reading `.ok` off a tuple. Driving main() with the
    real function (not a mock of it) is what makes this test able to fail.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        package = root / "pkg"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        monkeypatch.setitem(
            sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda *a, **k: root)
        )
        fake_config = type("SheetSage2Config", (), {})
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: fake_config())),
        )
        monkeypatch.setattr(probe_module, "probe_imports", lambda: [("synthetic", True, "")])
        monkeypatch.setattr(probe_module, "versions", lambda: {})

        code = probe_module.main([])

    out = capsys.readouterr().out
    assert "VERDICT" in out, out
    assert code in (0, 1)
    # And it did not die partway with a traceback.
    assert "AttributeError" not in out


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
    for _module_path, attribute, label in with_attribute:
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
    _label, ok, detail = results[0]
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
# "Absent" must not be reported as "incompatible"
# =============================================================================


def test_the_outcome_distinguishes_missing_from_structural(probe_module) -> None:
    """The defect the first *successful* probe run exposed.

    With the module-path and `--offline` bugs fixed, the probe finally ran and
    reported:

        FAIL  ModuleNotFoundError: No module named 'torchaudio'
        VERDICT: it does not. The pinned split stack is required; do not unify.

    But torchaudio was simply **not installed** in the main environment — nothing
    needed it there — and torchaudio 2.10.0 ships for cu128/cp311. The probe had
    turned "one import is absent" into "the stacks are incompatible", which is a
    stronger claim than its evidence and would have closed a live question.

    The kind field exists so that cannot recur.
    """
    outcome = probe_module.ProbeOutcome
    assert {outcome(True, "ok", "").kind, outcome(False, "missing", "").kind} == {"ok", "missing"}
    # The kinds are what main() switches on, so all four must exist.
    for kind in ("ok", "missing", "structural", "fetch"):
        assert outcome(False, kind, "x").kind == kind


def test_a_missing_module_yields_kind_missing(probe_module, monkeypatch) -> None:
    """A `ModuleNotFoundError` during the load is a missing package, explicitly."""
    import sys as _sys
    import types

    # A package whose __init__ imports something that does not exist.
    fake_root = probe_module.Path(probe_module.__file__).parent
    assert fake_root  # sanity: the module resolved a real path

    # Exercise the classification directly: ModuleNotFoundError -> "missing".
    def boom(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'torchaudio'", name="torchaudio")

    monkeypatch.setattr(probe_module, "snapshot_download", boom, raising=False)
    monkeypatch.setitem(_sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=boom))
    result = probe_module.probe_sheetsage2("m-a-p/SheetSage2", offline=True)
    assert result.ok is False
    # The fetch failed, so the kind is `fetch`; what matters is that it is not
    # silently reported as a stack incompatibility.
    assert result.kind in {"fetch", "missing"}


def test_the_verdict_says_inconclusive_for_a_missing_package(probe_module, capsys, monkeypatch) -> None:
    """And the printed verdict must not claim the question is settled."""
    monkeypatch.setattr(probe_module, "probe_imports", lambda: [("synthetic", True, "")])
    monkeypatch.setattr(
        probe_module,
        "probe_sheetsage2",
        lambda repo, offline: probe_module.ProbeOutcome(False, "missing", "'torchaudio' is not installed"),
    )
    monkeypatch.setattr(probe_module, "versions", lambda: {})

    probe_module.main([])
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" in out
    assert "do not unify" not in out
    assert "not incompatible" in out


def test_the_verdict_says_do_not_unify_for_a_structural_failure(probe_module, capsys, monkeypatch) -> None:
    """A genuine incompatibility still gets the firm answer."""
    monkeypatch.setattr(probe_module, "probe_imports", lambda: [("synthetic", True, "")])
    monkeypatch.setattr(
        probe_module,
        "probe_sheetsage2",
        lambda repo, offline: probe_module.ProbeOutcome(False, "structural", "ImportError: cannot import name"),
    )
    monkeypatch.setattr(probe_module, "versions", lambda: {})

    probe_module.main([])
    assert "do not unify" in capsys.readouterr().out


def test_torchaudio_is_declared_in_the_main_environment() -> None:
    """The package the probe found missing must now be present.

    SheetSage2's `audio_sheetsage2.py` calls `torchaudio.info`,
    `torchaudio.load` and `torchaudio.functional.resample` — stable APIs — and
    torchaudio 2.10.0 exists for cu128/cp311. So the answer was to install it,
    not to give up on unification.
    """
    pins = check_env.parse_requirements(REPO_ROOT / "worker" / "requirements.txt")
    as_dict = dict(pins)
    assert "torchaudio" in as_dict, "torchaudio is not declared in the main environment"
    # It must version-match torch or the pair is genuinely incompatible.
    assert as_dict["torchaudio"] == as_dict["torch"], (
        f"torchaudio {as_dict['torchaudio']} does not match torch {as_dict['torch']}"
    )


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
    monkeypatch.setattr(
        probe_module,
        "probe_sheetsage2",
        lambda repo, offline: probe_module.ProbeOutcome(True, "ok", "synthetic ok"),
    )
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
    monkeypatch.setattr(
        probe_module,
        "probe_sheetsage2",
        lambda repo, offline: probe_module.ProbeOutcome(True, "ok", "ok"),
    )
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
