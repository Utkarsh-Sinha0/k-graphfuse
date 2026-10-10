"""Generate publication-ready K GraphFuse presentation figures from audited sources."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.hashsalt"] = "k-graphfuse-presentation-v1"


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
PRECISION_SOURCE = Path("experiments/precision_gpu/PRODUCT_VERIFIED.json")
NATIVE_SOURCE = Path("evidence/2026-10-10/comparison_summary.json")
NAVY = "#17324D"
TEAL = "#007C78"
GREEN = "#39724D"
BLUE = "#4D6F91"
MUTED = "#718494"
PALE = "#EAF0F4"
GRID = "#D7E0E6"
FAIL = "#B64A32"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reduction_percent(old_ms: float, new_ms: float) -> float:
    return 100.0 * (1.0 - new_ms / old_ms)


def serial_rate_increase_percent(old_ms: float, new_ms: float) -> float:
    return 100.0 * (old_ms / new_ms - 1.0)


def selfcheck() -> None:
    assert math.isclose(reduction_percent(1.0, 0.5), 50.0)
    assert math.isclose(serial_rate_increase_percent(1.0, 0.5), 100.0)


def load_sources() -> tuple[list[dict], dict, dict]:
    precision_path = ROOT / PRECISION_SOURCE
    summary_path = ROOT / NATIVE_SOURCE
    precision = json.loads(precision_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert precision["status"] == "verified"
    cases = precision["t4_cases"]
    assert len(cases) == precision["t4_dual_gates"] == 6
    for case in cases:
        old_ms, new_ms = case["old_ms"], case["new_ms"]
        assert old_ms > new_ms > 0 and case["dual_gate"]
        assert math.isclose(case["speedup"], old_ms / new_ms, rel_tol=1e-6)

    # The native summary pins the detailed records used to produce its counts.
    integrity = summary["source_sha256"]
    for relative, expected in integrity.items():
        source = ROOT / Path(relative)
        assert source.is_file() and sha256(source) == expected, relative

    native = summary["native_cugraph"]
    assert native["comparable_cases"] + native["K_accuracy_failures"] == native["attempted_cases"]
    assert native["speed_gate_wins"] <= native["comparable_cases"]
    assert native["speed_and_both_reference_errors_wins"] <= native["speed_gate_wins"]
    assert summary["strongest_tested_overall"]["cases"] == native["comparable_cases"]
    assert summary["strongest_tested_gpu"]["cases"] == native["comparable_cases"]
    assert summary["new_precision_portfolio"]["comparator"] != "native cuGraph"

    sample_protocol = {"randomized_confirmation_blocks": 5, "samples_per_block": 31}
    sample_protocol["confirmation_samples_per_case"] = (
        sample_protocol["randomized_confirmation_blocks"] * sample_protocol["samples_per_block"]
    )
    assert sample_protocol["confirmation_samples_per_case"] == 155

    cited = [
        PRECISION_SOURCE,
        NATIVE_SOURCE,
        Path("experiments/precision_gpu/PRODUCT_RESULTS.md"),
        Path("experiments/precision_gpu/DERIVATION.md"),
        Path("docs/benchmarks/precision-gpu.md"),
    ]
    source_hashes = {str(path).replace("\\", "/"): sha256(ROOT / path) for path in cited}
    source_hashes.update(integrity)

    metrics = {
        "schema_version": 1,
        "precision_portfolio": {
            "source": str(PRECISION_SOURCE).replace("\\", "/"),
            "comparator_scope": "Tuned equivalent FP64 CuPy/PyTorch controls; not native cuGraph",
            "timing_scope": "Resident compute; similarity input/output scaling and owned GPU output copy included; validation, preparation, JIT and host transfer excluded",
            "protocol": sample_protocol,
            "point_and_interval_estimands": "Point uses pooled median ratio; interval bootstraps paired ratios of median block medians",
            "cases": [],
        },
        "native_coverage": {
            "source": str(NATIVE_SOURCE).replace("\\", "/"),
            **native,
            "strongest_tested_overall": summary["strongest_tested_overall"],
            "strongest_tested_gpu": summary["strongest_tested_gpu"],
            "precision_portfolio_separate": summary["new_precision_portfolio"],
            "original_single_workload_metrics": summary["original_single_workload_metrics"],
            "limit": "Eight favorable original-workload metrics are correlated observations, not independent replications.",
        },
        "pipeline": {
            "tested_family": "One constructed asymmetric family; outgoing couplings 1 and 1/64",
            "tested_condition_penalty": 8,
            "channel_support": [1, 16],
            "gpu_direction": "diagonal affine direction",
            "performance_api_separation": "Timings use frozen experimental sources; checked/repaired API tests are separate correctness evidence.",
        },
        "source_sha256": dict(sorted(source_hashes.items())),
    }

    for case in cases:
        old_ms, new_ms = case["old_ms"], case["new_ms"]
        ratio = old_ms / new_ms
        low_ratio, high_ratio = case["ci_low"], case["ci_high"]
        assert 0 < low_ratio <= high_ratio
        metrics["precision_portfolio"]["cases"].append(
            {
                "n": case["n"],
                "q": case["q"],
                "calculation": "value + derivative" if case["sensitivity"] else "value only",
                "method": case["method"],
                "control": case["old_method"],
                "old_ms": old_ms,
                "new_ms": new_ms,
                "speedup_ratio": ratio,
                "compute_time_reduction_percent": reduction_percent(old_ms, new_ms),
                "derived_serial_compute_rate_increase_percent": serial_rate_increase_percent(old_ms, new_ms),
                "compute_time_reduction_ci_percent": [100 * (1 - 1 / low_ratio), 100 * (1 - 1 / high_ratio)],
                "derived_serial_rate_increase_ci_percent": [100 * (low_ratio - 1), 100 * (high_ratio - 1)],
                "ci_speedup_ratio": [low_ratio, high_ratio],
                "dual_gate": case["dual_gate"],
            }
        )
    return cases, summary, metrics


def save_figure(fig: plt.Figure, stem: str, title: str, description: str) -> None:
    metadata_svg = {"Title": title, "Description": description, "Creator": "K GraphFuse research graphics", "Date": None}
    metadata_png = {"Title": title, "Description": description, "Software": "Matplotlib"}
    fig.savefig(ASSETS / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white", metadata=metadata_png)
    with (ASSETS / f"{stem}.svg").open("w", encoding="utf-8", newline="\n") as output:
        fig.savefig(output, format="svg", bbox_inches="tight", facecolor="white", metadata=metadata_svg)
    plt.close(fig)


def precision_chart(cases: list[dict]) -> None:
    rows = []
    for case in cases:
        label = "Value + derivative" if case["sensitivity"] else ("Wide filter" if case["q"] == 16 else "Value only")
        label += f"\nn = {case['n']:,}, q = {case['q']}"
        old_ms, new_ms = case["old_ms"], case["new_ms"]
        ratio = old_ms / new_ms
        rows.append(
            {
                "label": label,
                "control": case["old_method"],
                "old_ms": old_ms,
                "new_ms": new_ms,
                "reduction": reduction_percent(old_ms, new_ms),
                "rate": serial_rate_increase_percent(old_ms, new_ms),
                "ci_low": 100 * (1 - 1 / case["ci_low"]),
                "ci_high": 100 * (1 - 1 / case["ci_high"]),
                "color": BLUE if case["sensitivity"] else TEAL,
            }
        )

    fig, ax = plt.subplots(figsize=(15.5, 8.4))
    fig.subplots_adjust(left=0.30, right=0.98, top=0.82, bottom=0.32)
    ys = list(range(len(rows) - 1, -1, -1))
    ax.set_xlim(0, 125)
    ax.set_ylim(-0.65, len(rows) - 0.05)
    ax.set_yticks(ys, [row["label"] for row in rows], color=NAVY, fontsize=10)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0", "20", "40", "60", "80"], color=MUTED)
    ax.set_xlabel("Compute-time reduction from resident median (%)", color=NAVY, labelpad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    for y, row in zip(ys, rows):
        ax.barh(y, row["reduction"], height=0.43, color=row["color"], zorder=2)
        ax.hlines(y + 0.31, row["ci_low"], row["ci_high"], color=NAVY, linewidth=1.4, zorder=4)
        ax.vlines([row["ci_low"], row["ci_high"]], y + 0.23, y + 0.39, color=NAVY, linewidth=1.2, zorder=4)
        ax.scatter(row["reduction"], y + 0.31, color="white", edgecolor=NAVY, linewidth=1.1, s=25, zorder=5)
        ax.text(row["reduction"] + 1.0, y, f"{row['reduction']:.1f}%", va="center", color=NAVY, fontsize=9.5, weight="bold")
        ax.text(84.5, y, f"{row['control'].title()} {row['old_ms']:.3f} → {row['new_ms']:.3f}",
                va="center", ha="left", color=NAVY, fontsize=8.8)
        ax.text(113, y, f"{row['rate']:.1f}%", va="center", ha="center", color=NAVY, fontsize=9.5, weight="bold")

    ax.text(84.5, len(rows) - 0.18, "CONTROL → K (ms)", ha="left", va="bottom", color=MUTED, fontsize=8.5, weight="bold")
    ax.text(113, len(rows) - 0.18, "DERIVED SERIAL-RATE\nINCREASE", ha="center", va="bottom", color=MUTED, fontsize=8.5, weight="bold")
    fig.suptitle("Six constructed T4 workloads: faster resident compute with sampled accuracy gates",
                 x=0.30, y=0.965, ha="left", color=NAVY, fontsize=17, weight="bold")
    fig.text(0.30, 0.91, "Points use pooled-median ratios; whiskers are the monotone transform of the paired 5-block bootstrap interval for the ratio of median block medians.",
             ha="left", color=MUTED, fontsize=9.4)
    fig.text(0.30, 0.245,
             "Rate increase is derived from reciprocal latency (old/new − 1); it is not observed concurrent throughput. 5 × 31 = 155 confirmation samples per case.",
             ha="left", color=NAVY, fontsize=9.2)
    fig.text(0.30, 0.185,
             "Timing includes similarity input/output scaling and an owned GPU output copy; excludes validation, preparation, JIT and host transfer. One constructed asymmetric family; q = 4 or 16.",
             ha="left", color=NAVY, fontsize=8.7)
    fig.text(0.30, 0.125,
             "All six pass both gates against tuned FP64 CuPy/PyTorch controls (not native cuGraph). Independent 256/384-bit checks cover 16 fixture rows; errors are near the FP64 floor.",
             ha="left", color=NAVY, fontsize=8.7)
    fig.text(0.30, 0.065, "Sources: experiments/precision_gpu/PRODUCT_VERIFIED.json; docs/benchmarks/precision-gpu.md",
             ha="left", color=MUTED, fontsize=8)
    save_figure(fig, "precision_percent_comparison", "Precision GPU: time reduction and derived serial rate",
                "Six T4 workloads with compute-time reduction bars, transformed block-bootstrap intervals, control and K medians, and derived serial-rate increases.")


def native_coverage_chart(summary: dict) -> None:
    native = summary["native_cugraph"]
    attempted, comparable = native["attempted_cases"], native["comparable_cases"]
    speed, dual, failed = native["speed_gate_wins"], native["speed_and_both_reference_errors_wins"], native["K_accuracy_failures"]
    overall = summary["strongest_tested_overall"]["wins"]
    gpu = summary["strongest_tested_gpu"]["wins"]
    rows = [
        ("Native K: speed gate", speed, f"{speed} / {comparable} comparable", TEAL),
        ("Native K: speed + both reference errors no worse", dual, f"{dual} / {comparable} comparable · subset of {speed}", GREEN),
        ("K accuracy failures", failed, f"{failed} / {attempted} attempts · not comparable", FAIL),
        ("Strongest tested overall", overall, f"{overall} / {comparable} comparable", MUTED),
        ("Strongest tested GPU", gpu, f"{gpu} / {comparable} comparable", MUTED),
    ]
    fig, ax = plt.subplots(figsize=(14.5, 8.3))
    fig.subplots_adjust(left=0.35, right=0.97, top=0.78, bottom=0.33)
    ys = [4, 3, 2, 0.45, -0.55]
    ax.set_xlim(0, 90)
    ax.set_ylim(-1.1, 5.05)
    ax.set_yticks(ys, [row[0] for row in rows], color=NAVY, fontsize=10)
    ax.tick_params(axis="y", length=0, pad=11)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Number of cases", color=NAVY, labelpad=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(comparable, color=NAVY, linewidth=1.1, linestyle=(0, (3, 3)), alpha=0.7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    for y, (_, count, label, color) in zip(ys, rows):
        ax.barh(y, count, height=0.48, color=color, zorder=2)
        ax.text(count + 1.3, y, label, va="center", color=NAVY, fontsize=9.3, weight="bold" if color != MUTED else "normal")
    ax.text(0, 4.65, "Native cuGraph coverage", color=MUTED, fontsize=9, weight="bold")
    ax.text(0, -0.02, "Comparator context", color=MUTED, fontsize=9, weight="bold")
    fig.suptitle("Native comparison coverage keeps accuracy failures in view",
                 x=0.35, y=0.96, ha="left", color=NAVY, fontsize=17, weight="bold")
    fig.text(0.35, 0.89, f"{comparable} comparable cases from {attempted} K attempts; dashed line marks the comparable-case denominator.",
             ha="left", color=MUTED, fontsize=9.5)
    fig.text(0.35, 0.235,
             "Speed gate: at least 10% lower median, paired-block 95% interval above 1, and p95 no worse. The 4 K accuracy failures are shown separately.",
             ha="left", color=NAVY, fontsize=9.0)
    fig.text(0.35, 0.175,
             "Separate precision portfolio: 6/6 dual-gate wins versus tuned FP64 CuPy/PyTorch; it is not part of these native cuGraph counts.",
             ha="left", color=BLUE, fontsize=9.0, weight="bold")
    fig.text(0.35, 0.115,
             "The 26 dual-error wins are a subset of the 59 speed wins. Eight favorable earlier workload metrics are correlated observations, not independent replications.",
             ha="left", color=NAVY, fontsize=8.7)
    fig.text(0.35, 0.055, "Source: evidence/2026-10-10/comparison_summary.json and its SHA-256-pinned detailed records",
             ha="left", color=MUTED, fontsize=8)
    save_figure(fig, "native_win_coverage", "Native cuGraph comparison coverage",
                "Count of native speed-gate wins, dual reference-error wins, K accuracy failures, and tested comparator context, with attempted and comparable denominators.")


def pipeline_chart() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.82, bottom=0.30)
    fig.suptitle("K GraphFuse research path: ordered fusion to accuracy-checked GPU outputs",
                 x=0.035, y=0.96, ha="left", color=NAVY, fontsize=17, weight="bold")

    boxes = [
        ("Validate", "Canonical CSR A\nFinite input X\nCompatible stages"),
        ("Fuse in order", "Σ Aᵏ X Cₖᵀ\nPreserve coefficient\norder"),
        ("Balance + budget", "B = D⁻¹ A D\nκ₂(D) = 8 tested\nq = 1–16"),
        ("Fused evaluation", "Horner value + derivative\nFMA / mixed /\ncompensated\nGPU: diagonal direction"),
        ("Restore coordinates", "Y = D H₀\nJ = D J₀\nOwned outputs"),
        ("Accuracy + timing gates", "Full-output check\n256/384-bit refs\non 16 fixture rows\nHeld-out speed + p95"),
    ]
    left, width, gap, y0, height = 0.015, 0.14, 0.025, 0.45, 0.42
    for index, (heading, body) in enumerate(boxes):
        x = left + index * (width + gap)
        fill = PALE if index in (0, 1, 2) else "#E8F2EF"
        rect = matplotlib.patches.FancyBboxPatch(
            (x, y0), width, height, boxstyle="round,pad=0.009,rounding_size=0.012",
            facecolor=fill, edgecolor=GRID, linewidth=1.0, transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + 0.012, y0 + height - 0.065, f"{index + 1:02d}", transform=ax.transAxes,
                color=TEAL if index < 3 else GREEN, fontsize=9, weight="bold")
        ax.text(x + 0.012, y0 + height - 0.125, heading, transform=ax.transAxes,
                color=NAVY, fontsize=10.1, weight="bold", va="top")
        ax.text(x + 0.012, y0 + height - 0.185, body, transform=ax.transAxes,
                color=NAVY, fontsize=8.5, va="top", linespacing=1.45)
        if index < len(boxes) - 1:
            x2 = x + width + gap * 0.12
            ax.annotate("", xy=(x2 + gap * 0.76, y0 + height / 2), xytext=(x2, y0 + height / 2),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.2})

    ax.text(0.025, 0.38,
            "Scope: one constructed asymmetric family with outgoing couplings 1 and 1/64; the tested metric pays an eightfold condition penalty.",
            transform=ax.transAxes, color=NAVY, fontsize=9.1, weight="bold")
    ax.text(0.025, 0.31,
            "Timing note: frozen experimental kernels supply the performance results; checked/repaired API tests are separate correctness evidence, not a remeasured timing run.",
            transform=ax.transAxes, color=NAVY, fontsize=9.0)
    ax.text(0.025, 0.245,
            "The fused GPU profile is calibrated for 1–16 channels and diagonal affine directions; general directions use the existing fallback.",
            transform=ax.transAxes, color=MUTED, fontsize=8.8)
    ax.text(0.025, 0.12,
            "Sources: experiments/precision_gpu/DERIVATION.md; docs/benchmarks/precision-gpu.md",
            transform=ax.transAxes, color=MUTED, fontsize=8.0)
    save_figure(fig, "graphfuse_pipeline", "K GraphFuse research pipeline",
                "Six-step pipeline from canonical sparse input validation through ordered coefficient fusion, dyadic similarity, fused value and derivative evaluation, coordinate restoration, and independent accuracy and performance gates.")


def main() -> None:
    selfcheck()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cases, summary, metrics = load_sources()
    precision_chart(cases)
    native_coverage_chart(summary)
    pipeline_chart()
    metrics["artifacts"] = [
        "docs/assets/precision_percent_comparison.png",
        "docs/assets/precision_percent_comparison.svg",
        "docs/assets/native_win_coverage.png",
        "docs/assets/native_win_coverage.svg",
        "docs/assets/graphfuse_pipeline.png",
        "docs/assets/graphfuse_pipeline.svg",
    ]
    metrics["artifact_sha256"] = {name: sha256(ROOT / name) for name in metrics["artifacts"]}
    (ASSETS / "presentation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Generated {len(metrics['artifacts'])} chart files and presentation_metrics.json")


if __name__ == "__main__":
    main()
