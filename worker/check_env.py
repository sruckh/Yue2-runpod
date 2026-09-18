#!/usr/bin/env python3
"""Check that this environment has the versions it was built to have.

Run inside the image — at build time in each environment, and at worker boot.
Exits non-zero if anything is wrong, which is the whole point: the checks that
were here before *printed* versions and could not fail, so a venv that resolved
the wrong torch reported the wrong number and the build passed. The entire
reason that venv exists is the version pin.

Where the expected versions come from
-------------------------------------
The `requirements.txt` files in this repository — not a list written here. The
project's rule is one fact in one file, and a checker that restates the pins
would be a third copy to drift out of date. It reads them.

Also checked, because a wrong version is not the only way an environment is
broken:

- **Importability.** `torch==2.10.0` being installed says nothing about whether
  it loads. Import is attempted for every package that is importable without
  CUDA, which catches a broken wheel or a missing system library — the failure
  `libsndfile1` causes when it is absent.
- **The CUDA build.** `torch==2.10.0` matches both the CPU wheel from PyPI and
  the `+cu128` wheel from PyTorch's index. Only the second one can run on the
  GPU, and the requirement cannot tell them apart, so the local version
  identifier is checked explicitly.

Usage::

    python check_env.py                      # check the running interpreter
    python check_env.py --requirements PATH  # against a specific pins file
    python check_env.py --json               # machine-readable, for tests

Exit codes: 0 all good, 1 a mismatch or a failed import, 2 the checker itself
could not run (a missing requirements file, say — that is a build error, not a
version error, and it should not be silent).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

#: Where the pins live, relative to this file. One fact, one file.
YUEE2_REQUIREMENTS = Path(__file__).parent / "requirements.txt"
SHEETSAGE_REQUIREMENTS = Path(__file__).parent / "transcribe_sheetsage" / "requirements.txt"

#: Packages that are *expected* to be absent from some environments, so their
#: absence is reported but does not fail a check. `qwen-asr` pins no torch at
#: all, so which torch it resolved cannot be asserted — only reported.
REPORT_ONLY = frozenset({"qwen-asr"})

#: Packages whose import is worth attempting. Deliberately excludes anything
#: needing a display, a GPU, or a large model to load — the build has none of
#: those, and a check that cannot run where it is needed is not a check.
#: `runpod`, `boto3` and `accelerate` are importable but pull heavy optional
#: paths, so they are version-checked only.
IMPORT_CHECKS = {
    "torch": "torch",
    "transformers": "transformers",
    "numpy": "numpy",
    "soundfile": "soundfile",
    "safetensors": "safetensors",
    "tiktoken": "tiktoken",
    "yue2-infer": "yue2",
    "qwen-asr": "qwen_asr",
    "mido": "mido",
    "scipy": "scipy",
}

#: Distribution name -> import name, where they differ.
IMPORT_NAMES = {
    "yue2-infer": "yue2",
    "qwen-asr": "qwen_asr",
    "pretty-midi": "pretty_midi",
    "huggingface-hub": "huggingface_hub",
}

_PIN_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")


@dataclass
class Finding:
    """One package's state. `ok` is False only for a real problem."""

    package: str
    expected: str
    installed: str | None
    ok: bool
    #: True when a mismatch is reported but does not fail the check — an
    #: unpinned package, or one this environment is not expected to have.
    advisory: bool = False
    detail: str = ""

    def line(self) -> str:
        mark = "ok  " if self.ok else ("warn" if self.advisory else "FAIL")
        got = self.installed or "not installed"
        text = f"  [{mark}] {self.package:<20} expected {self.expected:<14} installed {got}"
        return text + (f"  — {self.detail}" if self.detail else "")


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    interpreter: str = ""
    requirements: str = ""

    @property
    def failures(self) -> list[Finding]:
        """Findings that make the environment wrong.

        Advisory findings are excluded: an unpinned package resolving to
        something unexpected is worth printing and is not worth failing a build
        over, because no pin says what it should have been.
        """
        return [f for f in self.findings if not f.ok and not f.advisory]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.advisory and not f.ok]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        out = [
            f"environment: {self.interpreter}",
            f"pins from:   {self.requirements}",
            "",
        ]
        out += [f.line() for f in self.findings]
        out.append("")
        if self.failures:
            out.append(f"{len(self.failures)} MISMATCH(ES) — this environment is not what it should be")
        else:
            out.append(f"all {len(self.findings)} declared packages match")
        if self.warnings:
            out.append(f"({len(self.warnings)} advisory, not counted as failure)")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "interpreter": self.interpreter,
            "requirements": self.requirements,
            "ok": self.ok,
            "failures": [f.package for f in self.failures],
            "warnings": [f.package for f in self.warnings],
            "packages": [
                {
                    "package": f.package,
                    "expected": f.expected,
                    "installed": f.installed,
                    "ok": f.ok,
                    "advisory": f.advisory,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


def normalise(name: str) -> str:
    """PEP 503 distribution-name normalisation: `yue2_infer` == `yue2-infer`."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: Path) -> list[tuple[str, str]]:
    """Read `name==version` pins. Ignores options, markers, comments, blanks.

    Only exact pins are returned. A loose requirement (`torch>=2`) cannot be
    asserted against an installed version, and inventing a target for it would
    be making up a fact.
    """
    pins: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-", "[")):
            continue
        match = _PIN_RE.match(stripped)
        if match:
            pins.append((match.group(1), match.group(2)))
    return pins


def installed_version(package: str) -> str | None:
    """The installed version, or None if the distribution is absent."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # a corrupt dist-info should not crash the check
        return None


def versions_match(expected: str, installed: str) -> bool:
    """Whether an installed version satisfies an exact pin.

    Handles the local version identifier: `torch==2.10.0` is satisfied by
    `2.10.0+cu128`, because PEP 440 ignores the local segment when the
    requirement does not specify one. Comparing strings directly would report
    every CUDA build as a mismatch.
    """
    if installed == expected:
        return True
    base = installed.split("+", 1)[0]
    return base == expected


def check_import(import_name: str) -> str:
    """Import a package, returning "" on success or the error on failure.

    Importing is a stronger check than reading a version: it is what catches a
    wheel that installed cleanly and cannot load. That is exactly how a missing
    `libsndfile1` presents — `soundfile` reports version 0.13.1 and then raises
    on import.
    """
    import importlib

    try:
        importlib.import_module(import_name)
        return ""
    except BaseException as exc:
        return f"{type(exc).__name__}: {exc}"


def verify_cuda_build(package: str) -> str:
    """Check that a torch has a CUDA build, not the CPU wheel from PyPI.

    `torch==2.10.0` is satisfied by both. Only one of them can use the GPU, and
    the failure mode is a worker that boots, accepts jobs, and then fails at
    `device="cuda"`.
    """
    if package != "torch":
        return ""
    version = installed_version("torch") or ""
    if not version:
        return ""
    if "+" not in version:
        return "no local version identifier — this looks like the CPU-only wheel from PyPI"
    if "cu" not in version:
        return f"local version {version.split('+', 1)[1]!r} does not name a CUDA build"
    return ""


def check_environment(requirements: Path, *, do_imports: bool = True) -> Report:
    """Compare the running interpreter against a pins file."""
    report = Report(
        interpreter=f"{sys.executable} (Python {sys.version.split()[0]})",
        requirements=str(requirements),
    )
    for package, expected in parse_requirements(requirements):
        found = installed_version(package)
        advisory = normalise(package) in {normalise(p) for p in REPORT_ONLY}

        if found is None:
            report.findings.append(
                Finding(
                    package,
                    expected,
                    None,
                    ok=False,
                    advisory=advisory,
                    detail="not installed",
                )
            )
            continue

        ok = versions_match(expected, found)
        detail = "" if ok else f"expected exactly {expected}"

        if ok and do_imports:
            import_name = IMPORT_NAMES.get(normalise(package), package.replace("-", "_"))
            if import_name in IMPORT_CHECKS.values() or package in IMPORT_CHECKS:
                error = check_import(IMPORT_CHECKS.get(package, import_name))
                if error:
                    ok = False
                    detail = f"import failed — {error}"

        if ok and package == "torch":
            cuda_problem = verify_cuda_build(package)
            if cuda_problem:
                ok = False
                detail = cuda_problem

        report.findings.append(Finding(package, expected, found, ok=ok, advisory=advisory and not ok, detail=detail))

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requirements", type=Path, default=YUEE2_REQUIREMENTS)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-imports", action="store_true", help="version-check only, skip imports")
    args = parser.parse_args(argv)

    if not args.requirements.is_file():
        print(f"check_env: requirements file not found: {args.requirements}", file=sys.stderr)
        return 2

    report = check_environment(args.requirements, do_imports=not args.no_imports)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
