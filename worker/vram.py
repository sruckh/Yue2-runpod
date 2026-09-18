#!/usr/bin/env python3
"""Sample device-wide GPU memory across one job, so a stage's peak is evidence.

Why device-wide, not `torch.cuda.max_memory_allocated()`
-------------------------------------------------------
The claim this exists to test is about the **subprocess boundary**. A `cover`
job runs SheetSage2 and Qwen3-ASR as separate child processes that exit —
releasing their VRAM — before YuE2 generation starts in the parent. The design
note in `shared/vram-budget.md` says this is what holds a cover job's peak at
YuE2's own ceiling rather than the sum of three model families.

No single process can observe that, because the thing under test is what the
*device* looked like while different processes came and went. Torch's counter is
per-process; `nvidia-smi` reports device-wide usage. The device is the unit the
24 GB budget is expressed in, so the device is what gets sampled.

What it cannot do — stated, not hidden
--------------------------------------
It samples; it does not instrument the allocator. A peak that rises and falls
entirely between two samples is never seen, so **the reported peak is a lower
bound on the true peak.** At the default 0.5 s interval that is a non-issue for
model loading, which takes seconds, and `samples` is reported alongside so a
reader can judge the resolution. It is recorded here rather than glossed because
a check that cannot fail is worse than no check, and a check that can only
under-report is the next-worst thing.

Absence is not failure
----------------------
On a machine with no GPU — a dev box, CI — `nvidia-smi` is missing or errors.
That yields an *unavailable* report and the job proceeds normally. A job must
never fail because its instrument was absent.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field

#: How often to sample. Half a second is dense relative to a model load (seconds)
#: and cheap relative to a 45 s job. Override with `VRAM_SAMPLE_INTERVAL_SECONDS`.
DEFAULT_INTERVAL_SECONDS = 0.5

_QUERY = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]


@dataclass
class VramReport:
    """What the GPU looked like during one job.

    `peak_mib` is device-wide and therefore includes CUDA context and any child
    process. `baseline_mib` is the first sample — an idle container still holds
    a context, so the interesting figure is the rise above baseline, not the
    absolute number.
    """

    available: bool
    note: str = ""
    device_total_mib: int | None = None
    baseline_mib: int | None = None
    peak_mib: int | None = None
    samples: int = 0
    #: phase name -> peak MiB seen while that phase was current.
    per_phase: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "note": self.note,
            "device_total_mib": self.device_total_mib,
            "baseline_mib": self.baseline_mib,
            "peak_mib": self.peak_mib,
            "peak_gib": round(self.peak_mib / 1024, 2) if self.peak_mib is not None else None,
            "samples": self.samples,
            "per_phase": dict(self.per_phase),
        }

    def render(self) -> str:
        if not self.available:
            return f"VRAM: not sampled ({self.note})"
        parts = [f"peak {self.peak_mib} MiB (baseline {self.baseline_mib}, n={self.samples})"]
        parts += [f"{name}={peak} MiB" for name, peak in self.per_phase.items()]
        return "VRAM: " + ", ".join(parts)


def _read_device_memory() -> tuple[int, int] | None:
    """`(used_mib, total_mib)` for the busiest device, or None if unreadable.

    Takes the maximum across devices when several are listed: the budget is
    per-24-GB-card and `gpuCount` is 1 here, but a future multi-GPU pod should
    report the worst card rather than an arbitrary one.
    """
    try:
        completed = subprocess.run(_QUERY, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    used: list[int] = []
    total = 0
    for line in completed.stdout.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            used.append(int(float(fields[0])))
            total = max(total, int(float(fields[1])))
        except ValueError:
            continue
    if not used:
        return None
    return max(used), total


def available() -> tuple[bool, str]:
    """Whether sampling is possible here, with a reason when it is not."""
    reading = _read_device_memory()
    if reading is None:
        return False, "nvidia-smi unavailable or reported no devices"
    return True, ""


class VramSampler:
    """Sample device memory in a background thread for the life of one job.

    Use as a context manager, or `start()`/`stop()` explicitly. `stop()` returns
    the report and is idempotent, so a `finally` cannot double-report.
    """

    def __init__(self, interval: float = DEFAULT_INTERVAL_SECONDS) -> None:
        self._interval: float = max(0.05, float(interval))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._phase = "job"
        self._samples = 0
        self._baseline: int | None = None
        self._peak: int | None = None
        self._total: int | None = None
        self._per_phase: dict[str, int] = {}
        self._note = ""
        self._available = True

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        ok, note = available()
        self._available = ok
        self._note = note
        if not ok:
            return
        self._thread = threading.Thread(target=self._run, name="vram-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> VramReport:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            # Bounded: a stuck sampler must not hold a finished job open.
            thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            return VramReport(
                available=self._available,
                note=self._note,
                device_total_mib=self._total,
                baseline_mib=self._baseline,
                peak_mib=self._peak,
                samples=self._samples,
                per_phase=dict(self._per_phase),
            )

    def mark(self, phase: str) -> None:
        """Name the phase that subsequent samples belong to.

        Called at phase boundaries by the handler. The value is a plain string
        so a stage can be renamed without touching this module.
        """
        with self._lock:
            self._phase = phase

    # --- context manager ---------------------------------------------------

    def __enter__(self) -> VramSampler:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # --- internals ---------------------------------------------------------

    def _record(self, used_mib: int, total_mib: int) -> None:
        with self._lock:
            if self._baseline is None:
                self._baseline = used_mib
            self._total = total_mib
            self._peak = used_mib if self._peak is None else max(self._peak, used_mib)
            current = self._per_phase.get(self._phase)
            self._per_phase[self._phase] = used_mib if current is None else max(current, used_mib)
            self._samples += 1

    def _run(self) -> None:
        """Sample until stopped. Never raises: a sampler cannot fail the job."""
        while not self._stop_event.is_set():
            try:
                reading = _read_device_memory()
                if reading is not None:
                    self._record(*reading)
            except BaseException:  # a background thread must not take the process down
                pass
            self._stop_event.wait(self._interval)
