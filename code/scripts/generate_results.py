#!/usr/bin/env python3
"""Generate every manuscript result value from immutable metrics artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def pp(value: float, *, sign: bool = True) -> str:
    return f"{100.0 * value:+.1f}" if sign else f"{100.0 * value:.1f}"


def interval(values: list[float], *, invert: bool = False) -> str:
    low, high = values
    if invert:
        low, high = -high, -low
    return f"[${100.0 * low:+.1f}$, ${100.0 * high:+.1f}$]"


def command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5", type=Path, required=True)
    parser.add_argument("--phase8", type=Path, required=True)
    parser.add_argument("--phase6", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    phase5 = load_json(args.phase5)
    phase8 = load_json(args.phase8)
    phase6 = load_json(args.phase6)
    tiny = phase5["sealed_test"]
    qasc = phase8["qasc"]
    synthetic = phase6["aggregate"]

    if phase5["verdict"] != "pass":
        raise RuntimeError("Tiny-A2D HM-Synth artifact is not positive")
    if phase5["canvas_assignment_count"] != 0:
        raise RuntimeError("Tiny-A2D HM-Synth artifact contains a canvas assignment")
    if not phase8["integrity"]["exact_no_message_parity"]:
        raise RuntimeError("QASC communication-off parity is not exact")
    if not phase8["integrity"]["independent_agent_token_rows"]:
        raise RuntimeError("QASC agent token rows are not independent")
    if not phase8["integrity"]["designated_receiver_no_vote"]:
        raise RuntimeError("QASC primary output is not the designated receiver")
    if phase6["verdict"] != "PASS_PHASE6_DEPENDENCY_SPAN_SYNTHETIC_MULTISEED":
        raise RuntimeError("controlled aggregate is not the sealed positive artifact")
    if not phase6["all_seed_gates_passed"]:
        raise RuntimeError("controlled aggregate contains an unpassed member")

    qacc = qasc["accuracy"]
    late_gain = qacc["final_only"] - qacc["matched"]
    late_ci = qasc["matched_minus_final_only_paired_bootstrap_ci95"]

    macros = [
        "% Generated from evidence/*.json; do not edit result values manually.",
        command("TinyNoMessage", pct(tiny["accuracy"]["no_message"])),
        command("TinyMatched", pct(tiny["accuracy"]["matched"])),
        command("TinyDeranged", pct(tiny["accuracy"]["deranged"])),
        command("TinyPresenceGain", pp(tiny["matched_minus_no_message"])),
        command(
            "TinyPresenceCI",
            interval(tiny["matched_minus_no_message_pair_bootstrap_ci95"]),
        ),
        command("TinyContentGain", pp(tiny["matched_minus_deranged"])),
        command(
            "TinyContentCI",
            interval(tiny["matched_minus_deranged_pair_bootstrap_ci95"]),
        ),
        command("TinyWrongFactTarget", pct(tiny["wrong_fact_target_rate"])),
        command("TinyWrongFactFlip", pct(tiny["wrong_fact_answer_flip_rate"])),
        command("SyntheticNoMessage", pct(synthetic["no_message_accuracy"]["mean"])),
        command("SyntheticMatched", pct(synthetic["matched_accuracy"]["mean"])),
        command("SyntheticDeranged", pct(synthetic["deranged_accuracy"]["mean"])),
        command("SyntheticLate", pct(synthetic["final_only_accuracy"]["mean"])),
        command("SyntheticContentGain", pp(synthetic["matched_minus_deranged"]["mean"])),
        command("SyntheticLateGain", pp(-synthetic["matched_minus_final_only"]["mean"])),
        command("SyntheticWrongFactTarget", pct(synthetic["wrong_fact_target_rate"]["mean"])),
        command("QASCNoMessage", pct(qacc["no_message"])),
        command("QASCRecurrent", pct(qacc["matched"])),
        command("QASCDeranged", pct(qacc["deranged"])),
        command("QASCLate", pct(qacc["final_only"])),
        command("QASCPresenceGain", pp(qasc["matched_minus_no_message"])),
        command("QASCPresenceCI", interval(qasc["matched_minus_no_message_paired_bootstrap_ci95"])),
        command("QASCContentGain", pp(qasc["matched_minus_deranged"])),
        command("QASCContentCI", interval(qasc["matched_minus_deranged_paired_bootstrap_ci95"])),
        command("QASCLateGain", pp(late_gain)),
        command("QASCLateCI", interval(late_ci, invert=True)),
    ]

    table = rf"""% Generated from evidence/*.json; do not edit result values manually.
\begin{{table}}[t]
\centering
\caption{{Three views of the hidden channel.  Accuracies are percentages and contrasts are
percentage points.  Tiny-A2D establishes content transfer on HM-Synth; the Dream-7B rendezvous
tests a sequential dependency, and distributed QASC tests split natural evidence.  The schedule
reversal in the last column is the central timing result: late fusion fails on the Dream-7B
rendezvous but wins on QASC.}}
\label{{tab:main-results}}
\small
\setlength{{\tabcolsep}}{{2.7pt}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Setting & No msg. & Matched & Deranged & Late & Match.$-$der. & Late$-$match. \\
\midrule
HM-Synth (Tiny-A2D) & {pct(tiny['accuracy']['no_message'])} &
{pct(tiny['accuracy']['matched'])} & {pct(tiny['accuracy']['deranged'])} &
\multicolumn{{1}}{{c}}{{--}} & ${pp(tiny['matched_minus_deranged'])}$ &
\multicolumn{{1}}{{c}}{{--}} \\
Dream-7B rendezvous & {pct(synthetic['no_message_accuracy']['mean'])} &
{pct(synthetic['matched_accuracy']['mean'])} & {pct(synthetic['deranged_accuracy']['mean'])} &
{pct(synthetic['final_only_accuracy']['mean'])} &
${pp(synthetic['matched_minus_deranged']['mean'])}$ &
${pp(-synthetic['matched_minus_final_only']['mean'])}$ \\
QASC (Dream-7B) & {pct(qacc['no_message'])} & {pct(qacc['matched'])} &
{pct(qacc['deranged'])} & {pct(qacc['final_only'])} &
${pp(qasc['matched_minus_deranged'])}$ & ${pp(late_gain)}$ \\
\bottomrule
\end{{tabular}}
\vspace{{1pt}}

\footnotesize Tiny-A2D content CI: {interval(tiny['matched_minus_deranged_pair_bootstrap_ci95'])};
QASC content CI: {interval(qasc['matched_minus_deranged_paired_bootstrap_ci95'])};
QASC late$-$matched CI: {interval(late_ci, invert=True)}.
\end{{table}}
"""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    (args.output_dir / "results_table.tex").write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
