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
from importlib import metadata
from pathlib import Path

#: The classes SheetSage2's own `config.json` auto_map names as its entry points.
#: Constructing a config exercises `configuration_sheetsage2.py`; importing the
#: model class exercises its `transformers` imports without needing weights.
PROBES = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("transformers.models.bart.modeling_bart.BartDecoder", "BartDecoder internals"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors", "safetensors"),
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
    """Import each of SheetSage2's dependencies in turn."""
    import importlib

    results = []
    for module, label in PROBES:
        try:
            importlib.import_module(module)
            results.append((label, True, ""))
        except BaseException as exc:
            results.append((label, False, f"{type(exc).__name__}: {exc}"))
    return results


def probe_sheetsage2(repo_id: str, offline: bool) -> tuple[bool, str]:
    """Download SheetSage2's *code* and import it, without loading weights.

    `AutoConfig` pulls `configuration_sheetsage2.py` and its imports; the model
    class is imported directly to exercise `modeling_sheetsage2.py`. Neither
    needs the safetensors, so this stays a few-hundred-KB operation rather than
    a multi-GB one.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return False, f"huggingface_hub unavailable: {exc}"

    try:
        # Only the Python source and the config — no weights, no assets.
        path = snapshot_download(
            repo_id,
            allow_patterns=["*.py", "config.json", "processor_config.json"],
            local_files_only=offline,
        )
    except Exception as exc:
        return False, f"could not fetch SheetSage2 code: {type(exc).__name__}: {exc}"

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
            return False, f"could not build a package spec from {path}"
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)

        model_module = importlib.import_module(f"{package_name}.modeling_sheetsage2")
        model_class = getattr(model_module, "SheetSage2Model", None)
        if model_class is None:
            return False, "modeling_sheetsage2 imported but exposes no SheetSage2Model"

        # The tokenizer is a second code path with its own imports.
        tokenizer_module = importlib.import_module(f"{package_name}.tokenization_sheetsage2")
        has_tokenizer = getattr(tokenizer_module, "SheetSage2Tokenizer", None) is not None

        return True, (
            f"config {kind}; SheetSage2Model and "
            f"{'SheetSage2Tokenizer' if has_tokenizer else 'tokenizer (absent)'} imported"
        )
    except BaseException as exc:
        return False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"


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
    ok, detail = probe_sheetsage2(args.repo, args.offline)
    print(f"  {'ok  ' if ok else 'FAIL'}  {detail}")

    verdict = ok and all(o for _, o, _ in imports)
    print()
    if verdict:
        print("VERDICT: SheetSage2's code loads under the unified stack.")
        print("         Numerical correctness is still unverified — that needs weights and a GPU.")
    else:
        print("VERDICT: it does not. The pinned split stack is required; do not unify.")

    # Machine-readable trailer so a build step or a test can parse the outcome.
    print(
        "\n"
        + json.dumps(
            {
                "unified_stack_ok": verdict,
                "imports": {label: ok_ for label, ok_, _ in imports},
                "sheetsage2": {"ok": ok, "detail": detail.splitlines()[0] if detail else ""},
                "versions": versions(),
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
