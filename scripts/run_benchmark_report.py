#!/usr/bin/env python3

import csv
import datetime as dt
import html
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark-results"
BENCH_HEADER = "case_family,case_variant,precision,data_type,workload,m,n,k,tiles,io_only_ns,compute_only_ns,end_to_end_ns,io_split_pct,compute_split_pct"


def run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return proc.stdout


def shell_output(command: str) -> str:
    return run(["/bin/zsh", "-lc", command]).strip()


def shell_output_or(command: str, default: str) -> str:
    proc = subprocess.run(
        ["/bin/zsh", "-lc", command],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    value = (proc.stdout or "").strip()
    if proc.returncode != 0 or not value:
        return default
    return value


def machine_metadata() -> Dict[str, str]:
    meta = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "product_name": shell_output_or("sw_vers -productName", "unknown"),
        "product_version": shell_output_or("sw_vers -productVersion", "unknown"),
        "build_version": shell_output_or("sw_vers -buildVersion", "unknown"),
        "kernel": shell_output_or("uname -a", "unknown"),
        "cpu_brand": shell_output_or("sysctl -n machdep.cpu.brand_string", shell_output_or("uname -m", "unknown")),
        "memory_bytes": shell_output_or("sysctl -n hw.memsize", "unknown"),
        "core_count": shell_output_or("sysctl -n machdep.cpu.core_count", shell_output_or("sysctl -n hw.ncpu", "unknown")),
        "thread_count": shell_output_or("sysctl -n machdep.cpu.thread_count", shell_output_or("sysctl -n hw.ncpu", "unknown")),
        "hw_model": shell_output_or("sysctl -n hw.model", "unknown"),
    }
    meta["machine_slug"] = slugify(f"{meta['cpu_brand']}_{meta['product_version']}")
    return meta


def slugify(value: str) -> str:
    cleaned = []
    for ch in value.lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("-")
    text = "".join(cleaned)
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-")


def parse_bench_output(output: str, metadata: Dict[str, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line == BENCH_HEADER or line.startswith("amx_sys io-vs-compute benchmark"):
            continue
        if line.startswith("filter=") or line.startswith("samples="):
            continue
        if "," not in line:
            continue
        if not line.startswith(("extract,", "fma,", "fms,", "mac,", "vecint,", "vecfp,", "matint,", "matfp,", "genlut,")):
            continue
        fields = line.split(",")
        if len(fields) != 14:
            continue
        row = {
            "case_family": fields[0],
            "case_variant": fields[1],
            "precision": fields[2],
            "data_type": fields[3],
            "workload": fields[4],
            "m": int(fields[5]),
            "n": int(fields[6]),
            "k": int(fields[7]),
            "tiles": int(fields[8]),
            "io_only_ns": float(fields[9]),
            "compute_only_ns": float(fields[10]),
            "end_to_end_ns": float(fields[11]),
            "io_split_pct": float(fields[12]),
            "compute_split_pct": float(fields[13]),
            "throughput_work_units_per_ns": work_units(fields[0], int(fields[5]), int(fields[6]), int(fields[7])) / max(float(fields[11]), 1.0),
            **metadata,
        }
        rows.append(row)
    return rows


def work_units(family: str, m: int, n: int, k: int) -> float:
    if family in {"matfp", "matint", "fma", "fms", "mac"}:
        return float(max(m * n * k, 1))
    if family in {"vecfp", "vecint", "extract", "genlut"}:
        return float(max(n, 1))
    return 1.0


def write_csv(rows: List[Dict[str, object]], path: pathlib.Path) -> None:
    if not rows:
        raise RuntimeError("no benchmark rows parsed")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[Dict[str, object]], path: pathlib.Path) -> None:
    path.write_text(json.dumps(rows, indent=2))


def group_by(rows: List[Dict[str, object]], key: str) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def summarize(rows: List[Dict[str, object]]) -> Dict[str, List[Tuple[str, float]]]:
    family_summary = []
    variant_summary = []
    precision_summary = []

    for family, items in group_by(rows, "case_family").items():
        avg = statistics.mean(item["end_to_end_ns"] for item in items)
        family_summary.append((family, avg))

    for variant, items in group_by(rows, "case_variant").items():
        score = statistics.mean(item["throughput_work_units_per_ns"] for item in items)
        variant_summary.append((variant, score))

    for precision, items in group_by(rows, "precision").items():
        score = statistics.mean(item["throughput_work_units_per_ns"] for item in items)
        precision_summary.append((precision, score))

    family_summary.sort(key=lambda item: item[1])
    variant_summary.sort(key=lambda item: item[1], reverse=True)
    precision_summary.sort(key=lambda item: item[1], reverse=True)

    return {
        "fastest_families": family_summary,
        "best_variants": variant_summary,
        "best_precisions": precision_summary,
    }


def svg_bar_chart(title: str, subtitle: str, data: List[Tuple[str, float]], unit: str, width: int = 960, height: int = 420, color: str = "#0b7285") -> str:
    margin_left = 240
    margin_top = 50
    margin_bottom = 40
    bar_gap = 10
    chart_width = width - margin_left - 40
    count = max(len(data), 1)
    bar_height = max(18, (height - margin_top - margin_bottom - (count - 1) * bar_gap) // count)
    max_val = max((value for _, value in data), default=1.0)
    max_val = max(max_val, 1.0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" aria-label="{html.escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="28" font-size="22" font-family="Menlo, monospace" fill="#111827">{html.escape(title)}</text>',
        f'<text x="24" y="46" font-size="12" font-family="Menlo, monospace" fill="#6b7280">{html.escape(subtitle)}</text>',
    ]

    for idx, (label, value) in enumerate(data):
        y = margin_top + idx * (bar_height + bar_gap)
        bar_w = 0 if max_val == 0 else chart_width * (value / max_val)
        parts.append(f'<text x="20" y="{y + bar_height * 0.7:.1f}" font-size="12" font-family="Menlo, monospace" fill="#111827">{html.escape(label)}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{bar_w:.2f}" height="{bar_height}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{margin_left + bar_w + 8:.2f}" y="{y + bar_height * 0.7:.1f}" font-size="12" font-family="Menlo, monospace" fill="#111827">{value:.2f} {html.escape(unit)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_scatter_plot(title: str, rows: List[Dict[str, object]], width: int = 960, height: int = 480) -> str:
    plot_rows = sorted(rows, key=lambda item: (item["tiles"], item["end_to_end_ns"]))
    max_tiles = max((row["tiles"] for row in plot_rows), default=1)
    max_end = max((row["end_to_end_ns"] for row in plot_rows), default=1.0)
    max_end = max(max_end, 1.0)
    margin = 60
    colors = {
        "extract": "#c2410c",
        "fma": "#2563eb",
        "fms": "#7c3aed",
        "mac": "#059669",
        "vecint": "#b91c1c",
        "vecfp": "#0f766e",
        "matint": "#a16207",
        "matfp": "#4338ca",
        "genlut": "#be123c",
    }
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" aria-label="{html.escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="28" font-size="22" font-family="Menlo, monospace" fill="#111827">{html.escape(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#94a3b8"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#94a3b8"/>',
        f'<text x="{width / 2:.0f}" y="{height - 16}" font-size="12" font-family="Menlo, monospace" text-anchor="middle" fill="#475569">tiles</text>',
        f'<text x="18" y="{height / 2:.0f}" font-size="12" font-family="Menlo, monospace" transform="rotate(-90 18 {height / 2:.0f})" text-anchor="middle" fill="#475569">end-to-end ns</text>',
    ]

    for row in plot_rows:
        x = margin + (width - 2 * margin) * (math.log2(row["tiles"] + 1) / math.log2(max_tiles + 1))
        y = (height - margin) - (height - 2 * margin) * (row["end_to_end_ns"] / max_end)
        color = colors.get(row["case_family"], "#334155")
        label = f"{row['case_variant']} {row['workload']}"
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}"><title>{html.escape(label)}: {row["end_to_end_ns"]:.2f} ns</title></circle>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def html_table(rows: List[Dict[str, object]]) -> str:
    header = (
        "<tr>"
        "<th>variant</th><th>workload</th><th>m</th><th>n</th><th>k</th><th>tiles</th>"
        "<th>io_only_ns</th><th>compute_only_ns</th><th>end_to_end_ns</th>"
        "<th>io_split_pct</th><th>compute_split_pct</th>"
        "</tr>"
    )
    body = []
    for row in sorted(rows, key=lambda item: item["end_to_end_ns"]):
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['case_variant']))}</td>"
            f"<td>{html.escape(str(row['workload']))}</td>"
            f"<td>{row['m']}</td>"
            f"<td>{row['n']}</td>"
            f"<td>{row['k']}</td>"
            f"<td>{row['tiles']}</td>"
            f"<td>{row['io_only_ns']:.2f}</td>"
            f"<td>{row['compute_only_ns']:.2f}</td>"
            f"<td>{row['end_to_end_ns']:.2f}</td>"
            f"<td>{row['io_split_pct']:.2f}</td>"
            f"<td>{row['compute_split_pct']:.2f}</td>"
            "</tr>"
        )
    return f"<table><thead>{header}</thead><tbody>{''.join(body)}</tbody></table>"


def build_report(rows: List[Dict[str, object]], metadata: Dict[str, str], csv_name: str) -> str:
    summary = summarize(rows)
    family_chart = svg_bar_chart(
        "Average End-to-End Latency by Family",
        "Lower is better. Values are mean end-to-end nanoseconds across all workloads in the family.",
        summary["fastest_families"],
        "ns",
        color="#1d4ed8",
    )
    variant_chart = svg_bar_chart(
        "Highest Throughput Variants",
        "Higher is better. Throughput is logical work units per nanosecond.",
        summary["best_variants"][:12],
        "work/ns",
        color="#047857",
    )
    precision_chart = svg_bar_chart(
        "Precision-Level Throughput",
        "Higher is better. Average logical work units per nanosecond across variants.",
        summary["best_precisions"],
        "work/ns",
        color="#b45309",
    )
    scatter = svg_scatter_plot("Latency vs Tile Count", rows)
    table = html_table(rows)

    bullets = []
    fastest_family = summary["fastest_families"][0] if summary["fastest_families"] else None
    best_variant = summary["best_variants"][0] if summary["best_variants"] else None
    best_precision = summary["best_precisions"][0] if summary["best_precisions"] else None
    if fastest_family:
        bullets.append(f"Lowest average end-to-end latency family: {fastest_family[0]} ({fastest_family[1]:.2f} ns)")
    if best_variant:
        bullets.append(f"Highest average throughput variant: {best_variant[0]} ({best_variant[1]:.4f} work/ns)")
    if best_precision:
        bullets.append(f"Highest average throughput precision bucket: {best_precision[0]} ({best_precision[1]:.4f} work/ns)")

    meta_list = "".join(
        f"<li><strong>{html.escape(key)}</strong>: {html.escape(str(value))}</li>"
        for key, value in metadata.items()
    )
    insight_list = "".join(f"<li>{html.escape(text)}</li>" for text in bullets)

    return f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>AMX Benchmark Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    section {{ margin-bottom: 28px; padding: 20px; background: white; border-radius: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    h1, h2 {{ margin-top: 0; }}
    code {{ background: #e2e8f0; padding: 2px 6px; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ position: sticky; top: 0; background: #f8fafc; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
  </style>
</head>
<body>
  <section>
    <h1>AMX IO vs Compute Benchmark Report</h1>
    <p>Result set generated from <code>{html.escape(csv_name)}</code>.</p>
    <ul>{meta_list}</ul>
  </section>
  <section>
    <h2>Key Findings</h2>
    <ul>{insight_list}</ul>
  </section>
  <section class=\"grid\">{family_chart}</section>
  <section class=\"grid\">{variant_chart}</section>
  <section class=\"grid\">{precision_chart}</section>
  <section class=\"grid\">{scatter}</section>
  <section>
    <h2>Raw Results</h2>
    {table}
  </section>
</body>
</html>
"""


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    metadata = machine_metadata()
    samples = os.environ.get("AMX_BENCH_SAMPLES", "5")
    bench_filter = os.environ.get("AMX_BENCH_FILTER")

    env = os.environ.copy()
    env.setdefault("AMX_BENCH_SAMPLES", samples)
    if bench_filter:
        env["AMX_BENCH_FILTER"] = bench_filter

    bench_cmd = [
        "cargo",
        "bench",
        "-p",
        "amx-sys",
        "--bench",
        "io_vs_compute",
        "--",
        "--nocapture",
    ]
    try:
        output = run(bench_cmd, env=env)
    except subprocess.CalledProcessError as exc:
        failure_log = RESULTS_DIR / "amx-benchmark-last-failure.log"
        failure_log.write_text(exc.stdout or "")
        sys.stderr.write("Benchmark execution failed.\n")
        sys.stderr.write(f"Command: {' '.join(bench_cmd)}\n")
        sys.stderr.write(f"Exit code: {exc.returncode}\n")
        sys.stderr.write(f"Captured output: {failure_log}\n")
        return exc.returncode or 1
    rows = parse_bench_output(output, metadata)
    if not rows:
        sys.stderr.write("No benchmark rows parsed.\n")
        return 1

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"amx-benchmark-{metadata['machine_slug']}-{stamp}"
    csv_path = RESULTS_DIR / f"{prefix}.csv"
    json_path = RESULTS_DIR / f"{prefix}.json"
    html_path = RESULTS_DIR / f"{prefix}.html"
    log_path = RESULTS_DIR / f"{prefix}.log"

    write_csv(rows, csv_path)
    write_json(rows, json_path)
    html_path.write_text(build_report(rows, metadata, csv_path.name))
    log_path.write_text(output)

    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"html={html_path}")
    print(f"log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())