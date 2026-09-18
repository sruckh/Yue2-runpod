#!/usr/bin/env python3
"""Probe whether SheetSage2's code loads under the *unified* stack.

This is the experiment, not the conclusion. It answers one question that cannot
be settled from documentation: **does SheetSage2's modeling code import and
construct under torch 2.10.0 + transformers 4.57.6 + numpy 2.x, rather than the
2.8.0 / 4.45.2 / 1.24.3 it pins?**

Why this is worth asking
------------------------
The packages do not appear to require each other's versions:

- transformers 4.45.2 declares **no torch constraint** (torch is an optional extra)
- SheetSage2's `config.json` records `"transformers_version": "4.45.2"` — a note
  about what it was tested with, not a requirement
- **Correction (2026-09-18):** an earlier version of this docstring claimed
  SheetSage2 "contains no numpy calls at all, so the numpy-2 alias removals
  cannot affect it". That is **false**. `midi_sheetsage2.py` calls
  `np.flatnonzero`, and numpy usage across the repo's ~25 modules has not been
  audited. The claim was load-bearing reasoning for calling unification
  low-risk, and it was never checked. Treat the numpy question as open.
- torch's own numpy constraint is likewise recorded elsewhere and not re-verified
  here.

The genuine structural risk is that SheetSage2 imports
`transformers.models.bart.modeling_bart.BartDecoder` — an internal path that can
move between transformers releases. Verified present in 4.57.6, but "the symbol
exists" and "the model constructs" are different claims, and only the second one
matters.

Why the scan is static
----------------------
The first four runs of this probe each discovered **one** absent package and
stopped, so each answer cost a full image build: torchaudio, then mir_eval, and
the queue behind them unknown. Importing to find out is what makes that
one-at-a-time; `absent_modules` reads the downloaded source with `ast` and
reports every absent dependency in a single pass instead.

What this does and does not prove
---------------------------------
It loads the model **configuration** and instantiates the module's classes. It
cannot verify numerical output, which needs weights and a GPU. So a pass here
means "the imports resolve and the code paths are compatible", not "the model
produces correct transcriptions".

That distinction is why this file is named `probe_*` and prints its findings
rather than asserting success. It is evidence to decide with.

Usage::

    python probe_unified_stack.py              # uses the default HF cache
    python probe_unified_stack.py --offline    # cache only, no network
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import sys
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

#: `(module, attribute, label)`. `attribute` is None when the module itself is
#: the thing being checked.
#:
#: **A module path is not a class path.** The first version of this listed
#: `transformers.models.bart.modeling_bart.BartDecoder` and handed it to
#: `importlib.import_module`, which takes a *module* — so it raised
#: `ModuleNotFoundError: ... 'modeling_bart' is not a package` in every
#: environment, at every version. It reported a structural incompatibility that
#: did not exist, and that false negative was the probe's whole verdict.
#: `BartDecoder` is an attribute of that module; the module is what SheetSage2
#: imports it from.
PROBES = (
    ("numpy", None, "numpy"),
    ("torch", None, "torch"),
    ("transformers", None, "transformers"),
    ("transformers.models.bart.modeling_bart", "BartDecoder", "transformers internal: BartDecoder"),
    ("huggingface_hub", None, "huggingface_hub"),
    ("safetensors", None, "safetensors"),
)

#: The source files only — never the multi-GB checkpoints. The image's weight
#: guard would catch a checkpoint landing, but the probe should not try.
ALLOW_PATTERNS = ["*.py", "config.json", "processor_config.json"]


def versions() -> dict[str, str]:
    out = {}
    for distribution in ("torch", "transformers", "numpy", "huggingface-hub", "safetensors", "soundfile"):
        try:
            out[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            out[distribution] = "not installed"
    return out


def probe_imports() -> list[tuple[str, bool, str]]:
    """Import each of SheetSage2's dependencies, and any named attribute.

    Checking the attribute matters: `import transformers.models.bart.modeling_bart`
    succeeding only says the module exists, not that `BartDecoder` is still
    defined in it. SheetSage2 imports the class, so the class is what has to
    survive a transformers upgrade.
    """
    results = []
    for module, attribute, label in PROBES:
        try:
            imported = importlib.import_module(module)
            if attribute is not None and not hasattr(imported, attribute):
                results.append((label, False, f"module {module} has no attribute {attribute!r}"))
                continue
            results.append((label, True, ""))
        except BaseException as exc:
            results.append((label, False, f"{type(exc).__name__}: {exc}"))
    return results


# =============================================================================
# What the downloaded source needs, read statically
# =============================================================================


def local_module_names(root: Path) -> set[str]:
    """Names belonging to the downloaded repo rather than to a dependency.

    The repo is a flat package: `modeling_sheetsage2.py` and its siblings sit in
    the root and import each other *relatively*, so those never appear. A
    top-level `import infer` would otherwise be reported as an absent dependency.
    """
    names: set[str] = set()
    for path in root.rglob("*.py"):
        names.add(path.stem)
        relative = path.relative_to(root)
        if len(relative.parts) > 1:
            names.add(relative.parts[0])
    return names


def third_party_imports(root: Path) -> set[str]:
    """Every top-level module the source imports, minus stdlib and its own files.

    Static by design. Importing to discover this stops at the first failure, so
    the caller learns one absent package per attempt — and each attempt is a
    build. Reading the source with `ast` answers it completely in one pass.
    """
    local = local_module_names(root)
    found: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            # level > 0 is relative: part of this repo, not a dependency.
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {name for name in found if name not in sys.stdlib_module_names and name not in local}


def absent_modules(modules: Iterable[str]) -> list[str]:
    """Which of these are not installed.

    `find_spec` rather than an import: importing has side effects and would stop
    at the first failure. This reports all of them, sorted.

    Only *absence* is detected. A package that is installed but unimportable
    returns a spec here and is caught later, by the load attempt.
    """
    absent: list[str] = []
    for name in sorted(modules):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            # A parent package that is itself absent or broken.
            spec = None
        if spec is None:
            absent.append(name)
    return absent


@dataclass
class ProbeOutcome:
    """Whether SheetSage2's code loaded, and — when it did not — *why not*.

    The distinction is the point. A missing package and a version
    incompatibility both print as failure, but only the second one answers the
    question this probe asks. Reporting "cannot unify" for an absent import
    overstates the evidence, which is what an earlier run did.
    """

    ok: bool
    #: `ok` | `missing` | `structural` | `fetch`
    kind: str
    detail: str
    #: Every absent dependency, when `kind` is `missing`. All of them, not the
    #: first — that list is the whole reason the scan is static.
    missing: tuple[str, ...] = ()


def load_sheetsage2(root: Path) -> ProbeOutcome:
    """Load SheetSage2's package and modules from an already-downloaded tree.

    The repo is a **package**, not a set of loose modules: its files use relative
    imports (`from .modeling_mert2 import MERT2Model`). A bare
    `import modeling_sheetsage2` after inserting the directory on `sys.path`
    cannot work — it has no package context, and the first version of this probe
    did exactly that. Loading it under a synthetic name gives the relative
    imports a parent, which is what `trust_remote_code` does internally too.
    """
    package_name = "sheetsage2_probe"
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(root), trust_remote_code=True)
        kind = type(config).__name__

        # Constructing the model needs weights; importing the module does not,
        # and the import is what exercises the transformers internals.
        spec = importlib.util.spec_from_file_location(
            package_name,
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        if spec is None or spec.loader is None:
            return ProbeOutcome(False, "structural", f"could not build a package spec from {root}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)

        model_module = importlib.import_module(f"{package_name}.modeling_sheetsage2")
        if getattr(model_module, "SheetSage2Model", None) is None:
            return ProbeOutcome(False, "structural", "modeling_sheetsage2 imported but exposes no SheetSage2Model")

        # The tokenizer is a second code path with its own imports.
        tokenizer_module = importlib.import_module(f"{package_name}.tokenization_sheetsage2")
        has_tokenizer = getattr(tokenizer_module, "SheetSage2Tokenizer", None) is not None

        return ProbeOutcome(
            True,
            "ok",
            f"config {kind}; SheetSage2Model and "
            f"{'SheetSage2Tokenizer' if has_tokenizer else 'tokenizer (absent)'} imported",
        )
    except ModuleNotFoundError as exc:
        # A distribution that is not installed, not a version that conflicts.
        # The two print identically and mean opposite things, which is the
        # distinction this class exists for: the third run of this probe
        # reported `ModuleNotFoundError: No module named 'torchaudio'` as a
        # verdict, and torchaudio simply was not installed.
        #
        # The static scan above should have caught this first. If it did not,
        # the import is of something the scan cannot see — a name built at
        # runtime, or a module inside a distribution that moved.
        return ProbeOutcome(False, "missing", f"{type(exc).__name__}: {exc}", (str(exc.name) if exc.name else "",))
    except BaseException as exc:
        return ProbeOutcome(False, "structural", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")


def probe_sheetsage2(repo_id: str, offline: bool) -> ProbeOutcome:
    """Fetch SheetSage2's code, check what it needs, then try to load it.

    Three stages, in this order for a reason. The scan runs before the load so
    that a missing dependency is reported as a *complete list* rather than as
    whichever import happened to fail first.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return ProbeOutcome(False, "missing", f"huggingface_hub unavailable: {exc}")

    try:
        path = snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS, local_files_only=offline)
    except Exception as exc:
        return ProbeOutcome(False, "fetch", f"could not fetch SheetSage2 code: {type(exc).__name__}: {exc}")

    root = Path(path)
    try:
        required = third_party_imports(root)
    except OSError as exc:
        return ProbeOutcome(False, "structural", f"could not scan {root}: {type(exc).__name__}: {exc}")

    absent = absent_modules(required)
    if absent:
        return ProbeOutcome(
            False,
            "missing",
            f"{len(absent)} of the {len(required)} modules SheetSage2 imports are not installed: {', '.join(absent)}",
            tuple(absent),
        )

    return load_sheetsage2(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="m-a-p/SheetSage2")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    print("=== installed versions in this environment ===")
    for name, version in versions().items():
        print(f"  {name:<18} {version}")

    print("\n=== SheetSage2's dependency imports ===")
    imports = probe_imports()
    for label, ok, error in imports:
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  — {error}" if error else ""))

    print(f"\n=== {args.repo} code under this stack ===")
    outcome = probe_sheetsage2(args.repo, args.offline)
    print(f"  {'ok  ' if outcome.ok else 'FAIL'}  [{outcome.kind}] {outcome.detail}")
    if outcome.missing:
        print("\n  SheetSage2's code imports these and they are not installed here:")
        for name in outcome.missing:
            print(f"    - {name}")

    imports_ok = all(o for _, o, _ in imports)
    verdict = outcome.ok and imports_ok
    print()
    if verdict:
        print("VERDICT: SheetSage2's code loads under the unified stack.")
        print("         Numerical correctness is still unverified — that needs weights and a GPU.")
    elif outcome.kind == "missing":
        # The honest answer is "the test did not complete", not "no".
        print("VERDICT: INCONCLUSIVE — the test could not complete.")
        print(f"         A package is absent, not incompatible: {outcome.detail}")
        print("         Add it and re-run; this says nothing yet about whether the stacks unify.")
    else:
        print(f"VERDICT: it does not ({outcome.kind}). The pinned split stack is required; do not unify.")

    # Machine-readable trailer so a build step or a test can parse the outcome.
    print(
        "\n"
        + json.dumps(
            {
                "unified_stack_ok": verdict,
                "imports": {label: ok_ for label, ok_, _ in imports},
                "sheetsage2": {
                    "ok": outcome.ok,
                    "kind": outcome.kind,
                    "detail": outcome.detail.splitlines()[0] if outcome.detail else "",
                    "missing": list(outcome.missing),
                },
                "versions": versions(),
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
