#!/usr/bin/env python3
"""Probe whether SheetSage2's code loads under the *unified* stack.

This is the experiment, not the conclusion. It answers one question that cannot
be settled from documentation: **does SheetSage2's modeling code import and
construct under torch 2.10.0 + transformers 4.57.6 + numpy 2.x, rather than the
2.8.0 / 4.45.2 / 1.24.3 it pins?**

Why this is worth asking
------------------------
Reading the packages, none of them actually require the others' versions:

- torch 2.10.0 declares **no numpy constraint at all**
- transformers 4.45.2 declares **no torch constraint** (torch is an optional extra)
- SheetSage2's `config.json` records `"transformers_version": "4.45.2"` — a note
  about what it was tested with, not a requirement
- its `requirements.txt` numpy floor is `>=1.17`, satisfied by both 1.24.3 and 2.2.6
- its code contains **no numpy calls at all**, so the numpy-2 alias removals
  cannot affect it

The one genuine risk is structural: SheetSage2 imports
`transformers.models.bart.modeling_bart.BartDecoder` — an internal path that can
move between transformers releases. Verified present in 4.57.6, but "the symbol
exists" and "the model constructs" are different claims, and only the second one
matters.

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
import json
import sys
import traceback
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
    import importlib

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


@dataclass
class ProbeOutcome:
    """Whether SheetSage2's code loaded, and — when it did not — *why not*.

    The distinction is the point. A missing package and a version
    incompatibility both print as failure, but only the second one answers the
    question this probe asks. Reporting "cannot unify" for an absent import
    overstates the evidence, which is what the first real run did.
    """

    ok: bool
    #: `ok` | `missing` | `structural` | `fetch`
    kind: str
    detail: str


def probe_sheetsage2(repo_id: str, offline: bool) -> ProbeOutcome:
    """Download SheetSage2's *code* and import it, without loading weights.

    `AutoConfig` pulls `configuration_sheetsage2.py` and its imports; the model
    class is imported directly to exercise `modeling_sheetsage2.py`. Neither
    needs the safetensors, so this stays a few-hundred-KB operation rather than
    a multi-GB one.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return ProbeOutcome(False, "missing", f"huggingface_hub unavailable: {exc}")

    try:
        # Only the Python source and the config — no weights, no assets.
        path = snapshot_download(
            repo_id,
            allow_patterns=["*.py", "config.json", "processor_config.json"],
            local_files_only=offline,
        )
    except Exception as exc:
        return ProbeOutcome(False, "fetch", f"could not fetch SheetSage2 code: {type(exc).__name__}: {exc}")

    # The repo is a **package**, not a set of loose modules: its files use
    # relative imports (`from .modeling_mert2 import MERT2Model`). A bare
    # `import modeling_sheetsage2` after inserting the directory on `sys.path`
    # cannot work — it has no package context. The first version of this probe
    # did exactly that and would have reported "not viable" for a reason that
    # has nothing to do with the dependency stack.
    #
    # Loading it as a package under a synthetic name gives the relative imports
    # a parent, which is what `trust_remote_code` does internally too.
    import importlib
    import importlib.util

    package_name = "sheetsage2_probe"
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
        kind = type(config).__name__

        # Now the package, so its relative imports resolve. Constructing the
        # model needs weights; importing the module does not, and the import is
        # what exercises the transformers internals.
        spec = importlib.util.spec_from_file_location(
            package_name,
            Path(path) / "__init__.py",
            submodule_search_locations=[str(path)],
        )
        if spec is None or spec.loader is None:
            return ProbeOutcome(False, "structural", f"could not build a package spec from {path}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)

        model_module = importlib.import_module(f"{package_name}.modeling_sheetsage2")
        model_class = getattr(model_module, "SheetSage2Model", None)
        if model_class is None:
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
        # distinction this class exists for: the third real run of this probe
        # reported `ModuleNotFoundError: No module named 'torchaudio'` as a
        # verdict, and torchaudio simply was not installed in that environment.
        #
        # Residual risk: a submodule that moved *inside* an installed
        # distribution also raises ModuleNotFoundError, and would be called
        # "missing" here. `probe_imports` checks those paths separately and the
        # verdict is the conjunction of the two, so a moved internal surfaces as
        # INCONCLUSIVE rather than as a false "do not unify". The detail line
        # carries the missing name so a reader can tell which it was.
        return ProbeOutcome(False, "missing", f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        return ProbeOutcome(False, "structural", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")


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
                },
                "versions": versions(),
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
