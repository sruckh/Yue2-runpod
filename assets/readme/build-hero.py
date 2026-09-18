#!/usr/bin/env python3
"""Generate assets/readme/hero.svg from a real YuE2 run.

The piano-roll is not decoration: it is the melody from `score.abc`, parsed from
an actual job's output. `score-notes.json` holds the extracted note events
(time in 1/32-note units, MIDI pitch, duration) so the art can be regenerated or
corrected without re-parsing ABC.

Re-run:  python3 build-hero.py > hero.svg
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).parent
NOTES = json.loads((HERE / "score-notes.json").read_text())

# --- frozen palette ----------------------------------------------------------
GROUND, SURFACE, RULE = "#0B1015", "#141C24", "#22303B"
INK, MUTED = "#E9EEF2", "#7C8B98"
SIGNAL, STEEL = "#F2A93B", "#5E8399"

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"

# --- canvas ------------------------------------------------------------------
W, H = 1200, 412

# piano-roll geometry
RX0, RX1 = 664.0, 1112.0
RY0, RY1 = 112.0, 300.0
LO_MIDI, HI_MIDI = 69, 83
SPAN = float(NOTES["span"])
ROW = (RY1 - RY0) / (HI_MIDI - LO_MIDI + 1)
NOTE_H = 10.0


def x_for(t: float) -> float:
    return RX0 + (t / SPAN) * (RX1 - RX0)


def w_for(d: float) -> float:
    return max(2.0, (d / SPAN) * (RX1 - RX0) - 1.6)


def y_for(midi: int) -> float:
    return RY0 + (HI_MIDI - midi) * ROW + (ROW - NOTE_H) / 2


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


out: list[str] = []
add = out.append

add(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">'
)
add("<title id=\"t\">YuE2 RunPod Worker</title>")
add(
    "<desc id=\"d\">Text and lyrics become a full song on serverless GPU. "
    "The panel plots a melody from YuE2\u2019s ABC score \u2014 the plan the model "
    "writes before it synthesizes any audio.</desc>"
)
add("<defs>")
add('  <clipPath id="roll"><rect x="%.0f" y="%.0f" width="%.0f" height="%.0f"/></clipPath>'
    % (RX0, RY0 - 4, RX1 - RX0, RY1 - RY0 + 8))
add("</defs>")

# ground
add(f'<rect width="{W}" height="{H}" rx="18" fill="{GROUND}"/>')

# --- title block -------------------------------------------------------------
add('<g id="title-block">')
add(f'<text x="64" y="78" font-family="{MONO}" font-size="13.5" letter-spacing="2.4" '
    f'fill="{MUTED}">YUE2 &#183; RUNPOD SERVERLESS WORKER</text>')

add(f'<text x="64" y="148" font-family="{SANS}" font-size="50" font-weight="600" '
    f'fill="{INK}" letter-spacing="-1.1">Style and lyrics</text>')
add(f'<text x="64" y="198" font-family="{SANS}" font-size="50" font-weight="600" '
    f'fill="{INK}" letter-spacing="-1.1">become a song.</text>')

add(f'<text x="64" y="242" font-family="{SANS}" font-size="16" fill="{MUTED}">'
    "YuE2-3B plans the music as an ABC score first, then realizes it</text>")
add(f'<text x="64" y="265" font-family="{SANS}" font-size="16" fill="{MUTED}">'
    "as 48 kHz stereo audio. One request, one song, one GPU.</text>")

add(f'<line x1="64" y1="300" x2="560" y2="300" stroke="{RULE}" stroke-width="1"/>')

add(f'<text x="64" y="330" font-family="{MONO}" font-size="12.5" fill="{MUTED}">'
    "one request &#183; one song &#183; one GPU</text>")
add(f'<text x="64" y="352" font-family="{MONO}" font-size="12.5" fill="{MUTED}">'
    "weights cached on the volume, never in the image</text>")
add("</g>")

# --- proof panel: the real score --------------------------------------------
add('<g id="score-panel">')
add(f'<rect x="616" y="52" width="520" height="292" rx="14" fill="{SURFACE}" '
    f'stroke="{RULE}" stroke-width="1"/>')

notes = NOTES["ins"] + NOTES["vocal"]
add(f'<text x="640" y="82" font-family="{MONO}" font-size="12.5" fill="{MUTED}">'
    "score.abc &#183; the plan, written before any audio</text>")

# section markers — real structure the model planned
for name, t in sorted(NOTES["marks"].items(), key=lambda kv: kv[1]["Ins"]):
    tx = x_for(t["Ins"])
    if t["Ins"] > 0:
        add(f'<line x1="{tx:.1f}" y1="{RY0 - 6:.0f}" x2="{tx:.1f}" y2="{RY1:.0f}" '
            f'stroke="{RULE}" stroke-width="1"/>')
    add(f'<text x="{tx + 6:.1f}" y="104" font-family="{MONO}" font-size="11.5" '
        f'fill="{MUTED}">{esc(name)}</text>')

# octave reference line — pitch structure, not decoration
c_midi = 72
add(f'<line x1="{RX0}" y1="{y_for(c_midi) + NOTE_H / 2:.1f}" x2="{RX1}" '
    f'y2="{y_for(c_midi) + NOTE_H / 2:.1f}" stroke="{RULE}" stroke-width="1"/>')
add(f'<text x="{RX0 - 8}" y="{y_for(c_midi) + NOTE_H / 2 + 4:.1f}" text-anchor="end" '
    f'font-family="{MONO}" font-size="11" fill="{MUTED}">C5</text>')

# the notes
add('<g clip-path="url(#roll)">')
for voice, colour in (("ins", STEEL), ("vocal", SIGNAL)):
    for t, midi, dur in NOTES[voice]:
        add(f'<rect x="{x_for(t):.1f}" y="{y_for(midi):.1f}" width="{w_for(dur):.1f}" '
            f'height="{NOTE_H}" rx="2" fill="{colour}" opacity="0.92"/>')
add("</g>")

# legend
add(f'<rect x="640" y="320" width="9" height="9" rx="2" fill="{SIGNAL}"/>')
add(f'<text x="656" y="328" font-family="{MONO}" font-size="12" fill="{MUTED}">vocal</text>')
add(f'<rect x="712" y="320" width="9" height="9" rx="2" fill="{STEEL}"/>')
add(f'<text x="728" y="328" font-family="{MONO}" font-size="12" fill="{MUTED}">instrumental</text>')
add(f'<text x="1112" y="328" text-anchor="end" font-family="{MONO}" font-size="12" '
    f'fill="{MUTED}">melody, planned</text>')
add("</g>")

add("</svg>")
print("\n".join(out))
