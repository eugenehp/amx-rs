#!/usr/bin/env python3

import csv
import math
import pathlib
from collections import defaultdict


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark-results"
FIGURES_DIR = ROOT / "figures"


def parse_csv(path: pathlib.Path):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    variants = defaultdict(list)
    for row in rows:
        variants[row["case_variant"]].append(float(row["throughput_work_units_per_ns"]))

    avg_variant = {
        variant: sum(values) / len(values)
        for variant, values in variants.items()
    }

    meta = {
        "path": path,
        "machine_slug": rows[0].get("machine_slug", "unknown"),
        "cpu_brand": rows[0].get("cpu_brand", "unknown"),
        "product_version": rows[0].get("product_version", "unknown"),
        "timestamp_utc": rows[0].get("timestamp_utc", "unknown"),
        "variant_count": len(avg_variant),
        "variants": avg_variant,
    }
    return meta


def choose_runs(runs):
    by_machine = defaultdict(list)
    for run in runs:
        by_machine[run["machine_slug"]].append(run)

    selected = []
    for machine, machine_runs in by_machine.items():
        machine_runs.sort(
            key=lambda item: (item["variant_count"], item["timestamp_utc"]),
            reverse=True,
        )
        selected.append(machine_runs[0])
    return sorted(selected, key=lambda item: item["cpu_brand"])


def color_for(value, min_log, max_log):
    if value is None or value <= 0.0:
        return "#f1f5f9"
    lv = math.log10(value)
    if max_log <= min_log:
        t = 0.5
    else:
        t = (lv - min_log) / (max_log - min_log)
    t = max(0.0, min(1.0, t))

    # Blue -> teal -> green ramp
    r = int(30 + (16 - 30) * t)
    g = int(64 + (185 - 64) * t)
    b = int(175 + (129 - 175) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_svg(selected_runs):
    chips = [f"{r['cpu_brand']}\nmacOS {r['product_version']}" for r in selected_runs]
    all_variants = sorted({variant for run in selected_runs for variant in run["variants"].keys()})

    # Order by geometric mean throughput across available chips (descending)
    def geo_mean(variant):
        vals = [run["variants"].get(variant) for run in selected_runs if run["variants"].get(variant)]
        if not vals:
            return 0.0
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    all_variants.sort(key=geo_mean, reverse=True)

    values = [
        run["variants"].get(variant)
        for variant in all_variants
        for run in selected_runs
        if run["variants"].get(variant) is not None and run["variants"].get(variant) > 0
    ]
    min_log = math.log10(min(values)) if values else 0.0
    max_log = math.log10(max(values)) if values else 1.0

    cell_w = 210
    cell_h = 22
    left = 360
    top = 110
    width = left + cell_w * len(selected_runs) + 80
    height = top + cell_h * len(all_variants) + 120

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="auto">')
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append('<text x="24" y="36" font-size="26" font-family="Menlo, monospace" fill="#0f172a">AMX Function Throughput Across Apple Silicon</text>')
    parts.append('<text x="24" y="58" font-size="13" font-family="Menlo, monospace" fill="#475569">Cells show average throughput_work_units_per_ns for each function variant. Darker means faster.</text>')

    # Column headers
    for idx, chip in enumerate(chips):
        x = left + idx * cell_w + 8
        label1, label2 = chip.split("\n", 1)
        parts.append(f'<text x="{x}" y="86" font-size="12" font-family="Menlo, monospace" fill="#111827">{label1}</text>')
        parts.append(f'<text x="{x}" y="102" font-size="11" font-family="Menlo, monospace" fill="#475569">{label2}</text>')

    # Rows and cells
    for r_idx, variant in enumerate(all_variants):
        y = top + r_idx * cell_h
        parts.append(f'<text x="20" y="{y + 15}" font-size="11" font-family="Menlo, monospace" fill="#111827">{variant}</text>')
        for c_idx, run in enumerate(selected_runs):
            x = left + c_idx * cell_w
            v = run["variants"].get(variant)
            fill = color_for(v, min_log, max_log)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 2}" rx="3" fill="{fill}"/>')
            if v is not None:
                parts.append(
                    f'<text x="{x + 6}" y="{y + 15}" font-size="10" font-family="Menlo, monospace" fill="#f8fafc">{v:.4f}</text>'
                )

    # Legend
    legend_x = left
    legend_y = height - 70
    steps = 8
    for i in range(steps):
        t = i / (steps - 1)
        lv = min_log + (max_log - min_log) * t
        val = 10 ** lv
        fill = color_for(val, min_log, max_log)
        x = legend_x + i * 40
        parts.append(f'<rect x="{x}" y="{legend_y}" width="36" height="14" rx="2" fill="{fill}"/>')
    parts.append(f'<text x="{legend_x}" y="{legend_y + 32}" font-size="11" font-family="Menlo, monospace" fill="#334155">low</text>')
    parts.append(f'<text x="{legend_x + 40 * (steps - 1)}" y="{legend_y + 32}" font-size="11" font-family="Menlo, monospace" fill="#334155">high</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    runs = []
    for path in sorted(RESULTS_DIR.glob("amx-benchmark-*.csv")):
        parsed = parse_csv(path)
        if parsed:
            runs.append(parsed)

    if not runs:
        raise SystemExit("No benchmark CSV files found in benchmark-results")

    selected = choose_runs(runs)
    svg = build_svg(selected)

    out_svg = FIGURES_DIR / "chip-function-throughput-heatmap.svg"
    out_svg.write_text(svg)

    selected_info = FIGURES_DIR / "chip-function-throughput-selected-runs.txt"
    with selected_info.open("w") as fh:
        for run in selected:
            fh.write(f"{run['cpu_brand']} | macOS {run['product_version']} | variants={run['variant_count']} | {run['path']}\n")

    print(f"chart={out_svg}")
    print(f"selected_runs={selected_info}")


if __name__ == "__main__":
    main()