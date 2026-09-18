"""Tests for the VRAM sampler.

This instrument exists for one claim, from the Stage 03 Human check:

    confirm via VRAM sampling that no stage exceeded the create-job peak
    (14.08 GiB)

The claim is about the **subprocess boundary** — that a cover job's SheetSage2
and Qwen3-ASR stages exit and release their VRAM before YuE2 generation starts,
so the peak is YuE2's own ceiling rather than the sum of three model families.
No single process can observe that about itself, which is why the sampler is
device-wide rather than `torch.cuda.max_memory_allocated()`.

The tests below are therefore mostly about the two ways an instrument lies:
reporting a number when it measured nothing, and failing the job it was
supposed to be watching.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import vram

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "worker"))


# =============================================================================
# Device-wide, because the claim is about the device
# =============================================================================


def test_it_reads_device_memory_not_process_memory() -> None:
    """The query must be `memory.used`, which is device-wide.

    `torch.cuda.max_memory_allocated()` is per-process and would report only the
    parent's own allocations — blind to exactly the child processes whose
    release is the thing being verified. The `_QUERY` constant is pinned so a
    switch to a per-process metric cannot slip in unnoticed.
    """
    query = " ".join(vram._QUERY)
    assert "memory.used" in query, "the sampler must read device-wide usage"
    assert "memory.total" in query, "the total is needed to judge headroom against 24 GB"
    assert "nvidia-smi" in vram._QUERY[0]
    # Not a per-process query: nvidia-smi's per-process mode uses --query-compute-apps.
    assert "--query-compute-apps" not in query


def test_multiple_devices_report_the_busiest(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-GPU pod must report the worst card, not an arbitrary one.

    `gpuCount` is 1 here, so this is defensive — but reporting the *first* card
    on a 2-GPU worker would silently understate the peak on the card that
    actually ran the job.
    """
    fake = subprocess.CompletedProcess(
        args=vram._QUERY, returncode=0, stdout="1000, 24564, NVIDIA L4\n9000, 24564, NVIDIA L4\n", stderr=""
    )
    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: fake)
    assert vram._read_device_memory() == (9000, 24564, "NVIDIA L4")


def test_a_failed_query_is_none_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """`None` and `0` mean opposite things and must not be conflated.

    Returning 0 for an unreadable device would report a *perfect* peak — a check
    that always passes, which is the failure mode this project keeps producing.
    """
    fake = subprocess.CompletedProcess(args=vram._QUERY, returncode=9, stdout="", stderr="boom")
    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: fake)
    assert vram._read_device_memory() is None


def test_a_missing_nvidia_smi_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dev box with no GPU must yield `None`, not an exception."""

    def boom(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(vram.subprocess, "run", boom)
    assert vram._read_device_memory() is None


def test_unparseable_output_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = subprocess.CompletedProcess(args=vram._QUERY, returncode=0, stdout="[N/A], [N/A], [N/A]\n", stderr="")
    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: fake)
    assert vram._read_device_memory() is None


# =============================================================================
# Absence is reported, never fatal
# =============================================================================


def test_absent_gpu_yields_an_unavailable_report_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """This worker is built and unit-tested on a box with no GPU.

    If an absent instrument raised, every test and every CI run would fail —
    and worse, a real job on a bad node would die for a reason unrelated to the
    work it was doing.
    """
    monkeypatch.setattr(vram, "available", lambda: (False, "nvidia-smi unavailable or reported no devices"))
    sampler = vram.VramSampler()
    sampler.start()
    report = sampler.stop()

    assert report.available is False
    assert report.peak_mib is None, "an unavailable sampler must not report a number"
    assert report.samples == 0
    assert "nvidia-smi" in report.note
    assert "not sampled" in report.render()


def test_the_report_never_raises_on_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """`stop()` is called from a `finally`; raising there would mask the real error."""
    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    sampler = vram.VramSampler()
    sampler.start()
    sampler.stop()
    sampler.stop()  # idempotent: a finally can run twice in a retry


# =============================================================================
# Peak, baseline and per-phase attribution
# =============================================================================


def test_peak_is_the_maximum_not_the_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """The number the Human check compares against 14.08 GiB is a *peak*.

    A sampler that reported the final reading would show a small number after a
    tall spike, and every stage would look like it passed.
    """
    readings = iter([(1000, 24564, "NVIDIA L4"), (12000, 24564, "NVIDIA L4"), (3000, 24564, "NVIDIA L4")])
    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    monkeypatch.setattr(vram, "_read_device_memory", lambda: next(readings, (3000, 24564, "NVIDIA L4")))

    sampler = vram.VramSampler(interval=0.01)
    sampler.start()
    time.sleep(0.2)
    report = sampler.stop()

    assert report.peak_mib == 12000, "the peak, not the last reading"
    assert report.baseline_mib == 1000, "the first reading is the baseline"


def test_phases_are_attributed_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-stage peaks are the evidence for the subprocess-boundary claim.

    A single job-wide peak cannot distinguish "the ASR stage held 6 GiB" from
    "the generation stage held 6 GiB", and only the second would be alarming.
    """
    seen: dict[str, int] = {}

    def fake_read():
        return seen.get("value", 0), 24564, "NVIDIA L4"

    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    monkeypatch.setattr(vram, "_read_device_memory", fake_read)

    sampler = vram.VramSampler(interval=0.01)
    sampler.start()

    seen["value"] = 5000
    sampler.mark("transcribe")
    time.sleep(0.15)

    seen["value"] = 9000
    sampler.mark("generate")
    time.sleep(0.15)

    report = sampler.stop()
    assert report.per_phase.get("transcribe") == 5000
    assert report.per_phase.get("generate") == 9000
    assert report.peak_mib == 9000


def test_it_captures_a_peak_from_a_child_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The property the whole instrument exists for.

    The sampler reads the *device*, so a spike produced by a process that is not
    the sampler's own must still be seen. This simulates that by changing what
    the device reports while a separate thread — standing in for a child — is
    "running", then letting it exit.

    If this ever fails, the sampler has been rewritten to read per-process
    memory and can no longer verify the subprocess boundary at all.
    """
    state = {"used": 1000}
    lock = threading.Lock()
    stop = threading.Event()

    def fake_read():
        with lock:
            return state["used"], 24564, "NVIDIA L4"

    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    monkeypatch.setattr(vram, "_read_device_memory", fake_read)

    sampler = vram.VramSampler(interval=0.01)
    sampler.start()

    def child():
        """A separate process's allocation, invisible to the parent's own counter."""
        with lock:
            state["used"] = 11000
        stop.wait(0.2)
        with lock:
            state["used"] = 1200

    thread = threading.Thread(target=child)
    thread.start()
    thread.join()
    report = sampler.stop()

    assert report.peak_mib == 11000, "a child process's peak was not observed"
    assert report.baseline_mib == 1000


# =============================================================================
# It must not become the problem it is measuring
# =============================================================================


def test_the_interval_has_a_floor() -> None:
    """A zero interval would fork `nvidia-smi` continuously and starve generation."""
    assert vram.VramSampler(interval=0)._interval == pytest.approx(0.05)
    assert vram.VramSampler(interval=-5)._interval == pytest.approx(0.05)


def test_a_sampling_failure_does_not_kill_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient nvidia-smi error must not end sampling for the rest of the job."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient driver error")
        return 7000, 24564, "NVIDIA L4"

    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    monkeypatch.setattr(vram, "_read_device_memory", flaky)

    sampler = vram.VramSampler(interval=0.01)
    sampler.start()
    time.sleep(0.3)
    report = sampler.stop()

    assert report.peak_mib == 7000, "sampling stopped after the first error"
    assert calls["n"] >= 3


def test_the_report_is_json_serialisable(monkeypatch: pytest.MonkeyPatch) -> None:
    """It is attached to the job response, which RunPod serialises."""
    import json

    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    monkeypatch.setattr(vram, "_read_device_memory", lambda: (8192, 24564, "NVIDIA L4"))

    sampler = vram.VramSampler(interval=0.01)
    sampler.start()
    time.sleep(0.1)
    report = sampler.stop()

    payload = report.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["peak_gib"] == pytest.approx(8.0)


def test_render_names_the_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    """The log line is what a human reads while watching a build."""
    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    monkeypatch.setattr(vram, "_read_device_memory", lambda: (4096, 24564, "NVIDIA L4"))

    sampler = vram.VramSampler(interval=0.01)
    sampler.start()
    sampler.mark("generate")
    time.sleep(0.1)
    report = sampler.stop()

    rendered = report.render()
    assert "generate" in rendered
    assert "4096" in rendered


# =============================================================================
# The report must say which device produced it
# =============================================================================
#
# The GPU pool behind this endpoint is not homogeneous: the same endpoint has
# reported device totals of 23034 MiB and 24564 MiB, from an L4-class card and a
# larger one, both sold as the same "24 GB" tier. A peak with no card attached
# cannot be compared against another peak — which is exactly what the Stage 03
# check asks someone to do.
#
# MIG makes this sharper. A MIG instance is a partitioned slice; it is not
# certain that `nvidia-smi --query-gpu=memory.*` inside such a container reports
# the *slice* rather than the *parent*. If it reports the parent, the peak
# describes a machine the job did not run on.


def test_the_card_name_is_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = subprocess.CompletedProcess(args=vram._QUERY, returncode=0, stdout="900, 23034, NVIDIA L4\n", stderr="")
    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: fake)
    assert vram._read_device_memory() == (900, 23034, "NVIDIA L4")


def test_a_name_containing_a_comma_is_not_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some drivers' names carry punctuation; the name is the whole remainder."""
    fake = subprocess.CompletedProcess(
        args=vram._QUERY, returncode=0, stdout="900, 97871, NVIDIA RTX PRO 6000, MIG 1g.24gb\n", stderr=""
    )
    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: fake)
    reading = vram._read_device_memory()
    assert reading is not None
    assert reading[2] == "NVIDIA RTX PRO 6000, MIG 1g.24gb", reading[2]


def test_the_report_carries_the_name_and_mig_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    monkeypatch.setattr(vram, "mig_mode", lambda: "Enabled")
    monkeypatch.setattr(vram, "_read_device_memory", lambda: (8192, 97871, "NVIDIA RTX PRO 6000"))

    sampler = vram.VramSampler(interval=0.01)
    sampler.start()
    time.sleep(0.1)
    payload = sampler.stop().to_dict()

    assert payload["device_name"] == "NVIDIA RTX PRO 6000"
    assert payload["mig_mode"] == "Enabled"
    # The log line's own coverage lives in the next test; asserting it here with
    # an `or True` tail would be a check that cannot fail.


def test_render_names_the_device_and_flags_mig(monkeypatch: pytest.MonkeyPatch) -> None:
    """The log line is what a human reads; it must carry the caveat too."""
    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    monkeypatch.setattr(vram, "mig_mode", lambda: "Enabled")
    monkeypatch.setattr(vram, "_read_device_memory", lambda: (4096, 97871, "NVIDIA RTX PRO 6000"))

    sampler = vram.VramSampler(interval=0.01)
    sampler.start()
    time.sleep(0.1)
    rendered = sampler.stop().render()

    assert "NVIDIA RTX PRO 6000" in rendered
    assert "MIG=Enabled" in rendered, "a MIG reading must be flagged where a human will see it"


def test_a_non_mig_reading_is_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vram, "available", lambda: (True, ""))
    monkeypatch.setattr(vram, "mig_mode", lambda: "Disabled")
    monkeypatch.setattr(vram, "_read_device_memory", lambda: (4096, 23034, "NVIDIA L4"))

    sampler = vram.VramSampler(interval=0.01)
    sampler.start()
    time.sleep(0.1)
    rendered = sampler.stop().render()
    assert "MIG=" not in rendered


def test_mig_mode_tolerates_a_driver_without_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsupported query field fails the whole call; that is not an error."""
    fake = subprocess.CompletedProcess(args=vram._MIG_QUERY, returncode=1, stdout="", stderr="Field not supported")
    monkeypatch.setattr(vram.subprocess, "run", lambda *a, **k: fake)
    assert vram.mig_mode() == ""


def test_mig_mode_tolerates_a_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(vram.subprocess, "run", boom)
    assert vram.mig_mode() == ""
