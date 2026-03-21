#!/usr/bin/env python3
"""Generate SVG comparison charts from scalar_vs_amx benchmark CSV output.

Usage:
    python3 scripts/chart_scalar_vs_amx.py benchmark-results/scalar-vs-amx-*.csv
"""

import csv
import math
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"

COLORS = {"scalar": "#6366f1", "amx": "#f59e0b", "amx_par": "#ef4444", "blas": "#10b981"}
LABELS = {"scalar": "Scalar Rust", "amx": "AMX (1 core)", "amx_par": "AMX (all cores)", "blas": "Accelerate"}


def load_csv(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(csv.DictReader(f))
    for r in rows:
        for k in ("m", "k", "n"):
            r[k] = int(r[k])
        for k in ("flops", "scalar_us", "amx_us", "amx_par_us", "blas_us",
                   "scalar_gflops", "amx_gflops", "amx_par_gflops", "blas_gflops"):
            r[k] = float(r[k])
    return rows


def svg_header(w, h, title, subtitle=""):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="auto">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<text x="24" y="32" font-size="20" font-family="Menlo,monospace" fill="#0f172a">{title}</text>',
    ]
    if subtitle:
        parts.append(f'<text x="24" y="52" font-size="11" font-family="Menlo,monospace" fill="#64748b">{subtitle}</text>')
    return parts


def bar_chart(title, subtitle, labels, datasets, unit, width=1000):
    """Grouped horizontal bar chart. datasets = {"name": [values], ...}"""
    n_groups = len(labels)
    n_bars = len(datasets)
    bar_h = 14
    gap = 3
    group_gap = 12
    group_h = n_bars * (bar_h + gap) + group_gap
    left = 200
    top = 70
    bottom = 50
    chart_w = width - left - 100
    height = top + n_groups * group_h + bottom

    max_val = max(v for ds in datasets.values() for v in ds if v > 0)
    max_val = max(max_val, 1.0)

    parts = svg_header(width, height, title, subtitle)

    # Legend
    for i, (name, _) in enumerate(datasets.items()):
        lx = left + i * 180
        parts.append(f'<rect x="{lx}" y="{top - 16}" width="10" height="10" rx="2" fill="{COLORS[name]}"/>')
        parts.append(f'<text x="{lx+14}" y="{top - 7}" font-size="11" font-family="Menlo,monospace" fill="#334155">{LABELS[name]}</text>')

    for gi, label in enumerate(labels):
        gy = top + gi * group_h
        parts.append(f'<text x="12" y="{gy + n_bars * (bar_h + gap) // 2 + 4}" '
                     f'font-size="11" font-family="Menlo,monospace" fill="#111827">{label}</text>')

        for bi, (name, values) in enumerate(datasets.items()):
            by = gy + bi * (bar_h + gap)
            v = values[gi]
            if v <= 0:
                continue
            bw = max(chart_w * (v / max_val), 1)
            parts.append(f'<rect x="{left}" y="{by}" width="{bw:.1f}" height="{bar_h}" rx="3" fill="{COLORS[name]}"/>')
            parts.append(f'<text x="{left + bw + 5:.1f}" y="{by + 11}" font-size="10" '
                         f'font-family="Menlo,monospace" fill="#334155">{v:.1f} {unit}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def line_chart(title, subtitle, x_labels, datasets, y_unit, width=1000, height=420):
    """Line chart with log-scale Y axis."""
    left = 80
    right = 40
    top = 70
    bottom = 60
    chart_w = width - left - right
    chart_h = height - top - bottom

    all_vals = [v for ds in datasets.values() for v in ds if v > 0]
    if not all_vals:
        return ""
    y_min = math.log10(min(all_vals) * 0.8)
    y_max = math.log10(max(all_vals) * 1.2)
    if y_max <= y_min:
        y_max = y_min + 1

    parts = svg_header(width, height, title, subtitle)

    # Legend
    for i, (name, _) in enumerate(datasets.items()):
        lx = left + i * 180
        parts.append(f'<rect x="{lx}" y="{top - 16}" width="10" height="10" rx="2" fill="{COLORS[name]}"/>')
        parts.append(f'<text x="{lx+14}" y="{top - 7}" font-size="11" font-family="Menlo,monospace" fill="#334155">{LABELS[name]}</text>')

    # Axes
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#cbd5e1"/>')

    # Y gridlines
    y_ticks = []
    low = math.floor(y_min)
    high = math.ceil(y_max)
    for exp in range(low, high + 1):
        for mantissa in [1, 2, 5]:
            val = mantissa * (10 ** exp)
            lv = math.log10(val)
            if y_min <= lv <= y_max:
                y_ticks.append((val, lv))

    for val, lv in y_ticks:
        y = top + chart_h - chart_h * (lv - y_min) / (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#f1f5f9"/>')
        label = f"{val:.0f}" if val >= 1 else f"{val:.2f}"
        parts.append(f'<text x="{left-6}" y="{y+4:.1f}" font-size="9" font-family="Menlo,monospace" '
                     f'fill="#64748b" text-anchor="end">{label}</text>')

    # X labels
    n = len(x_labels)
    for i, label in enumerate(x_labels):
        x = left + chart_w * i / max(n - 1, 1)
        parts.append(f'<text x="{x:.1f}" y="{top+chart_h+18}" font-size="9" font-family="Menlo,monospace" '
                     f'fill="#64748b" text-anchor="middle">{label}</text>')

    # Lines
    for name, values in datasets.items():
        points = []
        for i, v in enumerate(values):
            if v <= 0:
                continue
            x = left + chart_w * i / max(n - 1, 1)
            lv = math.log10(v)
            y = top + chart_h - chart_h * (lv - y_min) / (y_max - y_min)
            points.append(f"{x:.1f},{y:.1f}")
        if len(points) >= 2:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" '
                         f'stroke="{COLORS[name]}" stroke-width="2.5"/>')
        for pt in points:
            x, y = pt.split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{COLORS[name]}"/>')

    parts.append(f'<text x="{left - 10}" y="{top + chart_h // 2}" font-size="10" font-family="Menlo,monospace" '
                 f'fill="#475569" text-anchor="middle" transform="rotate(-90 {left - 50} {top + chart_h // 2})">{y_unit}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    csv_files = sys.argv[1:] if len(sys.argv) > 1 else sorted(
        str(p) for p in ROOT.glob("benchmark-results/scalar-vs-amx-*.csv")
    )
    if not csv_files:
        print("No CSV files found. Run benchmark with BENCH_CSV=path.csv first.")
        sys.exit(1)

    rows = load_csv(csv_files)
    chip = rows[0]["chip"] if rows else "Unknown"
    FIGURES_DIR.mkdir(exist_ok=True)

    # --- 1. Square matmul GFLOPS (bar chart) ---
    sq_rows = [r for r in rows if r["op"] == "matmul" and r["m"] == r["k"] == r["n"]]
    sq_rows.sort(key=lambda r: r["m"])
    # Pick a subset for readability
    target_n = {8, 16, 32, 64, 128, 256, 512}
    sq_sub = [r for r in sq_rows if r["m"] in target_n]

    labels = [r["shape"] for r in sq_sub]
    svg = bar_chart(
        "Square Matmul GFLOPS — Scalar vs AMX vs Accelerate",
        f"{chip} · f32 · higher is better",
        labels,
        {"scalar": [r["scalar_gflops"] for r in sq_sub],
         "amx": [r["amx_gflops"] for r in sq_sub],
         "amx_par": [r["amx_par_gflops"] for r in sq_sub],
         "blas": [r["blas_gflops"] for r in sq_sub]},
        "GFLOPS",
    )
    p = FIGURES_DIR / "matmul-gflops-bar.svg"
    p.write_text(svg)
    print(f"Wrote {p}")

    # --- 2. Square matmul GFLOPS (line chart) ---
    labels = [str(r["m"]) for r in sq_rows]
    svg = line_chart(
        "Square Matmul GFLOPS vs Matrix Size",
        f"{chip} · f32 · log scale · higher is better",
        labels,
        {"scalar": [r["scalar_gflops"] for r in sq_rows],
         "amx": [r["amx_gflops"] for r in sq_rows],
         "amx_par": [r["amx_par_gflops"] for r in sq_rows],
         "blas": [r["blas_gflops"] for r in sq_rows]},
        "GFLOPS",
    )
    p = FIGURES_DIR / "matmul-gflops-line.svg"
    p.write_text(svg)
    print(f"Wrote {p}")

    # --- 3. Square matmul latency (line chart) ---
    svg = line_chart(
        "Square Matmul Latency vs Matrix Size",
        f"{chip} · f32 · log scale · lower is better",
        labels,
        {"scalar": [r["scalar_us"] for r in sq_rows],
         "amx": [r["amx_us"] for r in sq_rows],
         "amx_par": [r["amx_par_us"] for r in sq_rows],
         "blas": [r["blas_us"] for r in sq_rows]},
        "µs",
    )
    p = FIGURES_DIR / "matmul-latency-line.svg"
    p.write_text(svg)
    print(f"Wrote {p}")

    # --- 4. Dot product GFLOPS (line chart) ---
    dot_rows = [r for r in rows if r["op"] == "dot"]
    dot_rows.sort(key=lambda r: r["m"])
    if dot_rows:
        labels = [r["shape"].replace("len=", "") for r in dot_rows]
        svg = line_chart(
            "Dot Product GFLOPS vs Vector Length",
            f"{chip} · f32 · log scale · higher is better",
            labels,
            {"scalar": [r["scalar_gflops"] for r in dot_rows],
             "amx": [r["amx_gflops"] for r in dot_rows],
             "amx_par": [r["amx_par_gflops"] for r in dot_rows],
             "blas": [r["blas_gflops"] for r in dot_rows]},
            "GFLOPS",
        )
        p = FIGURES_DIR / "dot-gflops-line.svg"
        p.write_text(svg)
        print(f"Wrote {p}")

    # --- 5. Rectangular matmul GFLOPS (bar chart, selected shapes) ---
    rect_rows = [r for r in rows if r["op"] == "matmul" and r["m"] != r["k"]]
    if rect_rows:
        rect_sub = rect_rows[:min(12, len(rect_rows))]
        labels = [r["shape"].replace("x", "×") for r in rect_sub]
        svg = bar_chart(
            "Rectangular Matmul GFLOPS",
            f"{chip} · f32 · selected shapes · higher is better",
            labels,
            {"scalar": [r["scalar_gflops"] for r in rect_sub],
             "amx": [r["amx_gflops"] for r in rect_sub],
             "amx_par": [r["amx_par_gflops"] for r in rect_sub],
             "blas": [r["blas_gflops"] for r in rect_sub]},
            "GFLOPS",
        )
        p = FIGURES_DIR / "rect-matmul-gflops-bar.svg"
        p.write_text(svg)
        print(f"Wrote {p}")

    print("Done.")


if __name__ == "__main__":
    main()
