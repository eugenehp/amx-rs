#!/usr/bin/env python3
"""Parse text-format AMX instruction benchmark results and generate comparison SVGs."""

import math
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark-results"
FIGURES_DIR = ROOT / "figures"


def parse_txt_benchmark(path: pathlib.Path) -> dict:
    """Parse a text benchmark file into structured data."""
    text = path.read_text()
    lines = text.splitlines()

    # Extract header metadata
    chip_name = ""
    date = ""
    kernel = ""
    cores = ""
    perflevels = ""
    for line in lines:
        if line.startswith("# ") and not chip_name:
            chip_name = line[2:].strip()
        elif line.startswith("# 20"):
            date = line[2:].strip()
        elif line.startswith("# Darwin"):
            kernel = line[2:].strip()
        elif line.startswith("# cores:"):
            cores = line.split(":")[1].strip()
        elif line.startswith("# perflevels:"):
            perflevels = line.split(":")[1].strip()

    result = {
        "path": str(path),
        "chip_name": chip_name,
        "date": date,
        "kernel": kernel,
        "cores": cores,
        "perflevels": perflevels,
        "sections": {},
    }

    # Parse sections
    current_section = None
    instruction_re = re.compile(
        r'^\s+'
        r'((?:set \+ clr|ldx \+ stx|fma16 vector|fma16 matrix|fma32 vector|fma32 matrix|'
        r'fma64 vector|fma64 matrix|mac16 vector|mac16 matrix))'
        r'\s+\(([^)]+)\)?\s+'
        r'([\d.]+)\s+([\d.]+)\s+([\d.]+|-)'
    )

    section_headers = [
        "Single P-core",
        "Single E-core",
        "All P-cores parallel",
        "All E-cores parallel",
        "Whole chip",
    ]

    for line in lines:
        stripped = line.strip()
        for hdr in section_headers:
            if stripped.startswith(hdr):
                current_section = hdr
                result["sections"][current_section] = []
                break

        if current_section and current_section != "P-core vs E-core":
            m = instruction_re.match(line)
            if m:
                instr = m.group(1).strip()
                size_info = m.group(2) if m.group(2) else ""
                col1 = float(m.group(3))
                col2 = float(m.group(4))
                col3_str = m.group(5)
                gflops = float(col3_str) if col3_str != "-" else None

                result["sections"][current_section].append({
                    "instruction": instr,
                    "size": size_info,
                    "ns_per_op": col1,
                    "ginstr_per_s": col2,
                    "gflops": gflops,
                })

    # Also parse "Whole chip" which has different columns (no ns/op)
    # Re-parse with different regex for whole chip
    whole_chip_re = re.compile(
        r'^\s+'
        r'((?:fma16 vector|fma16 matrix|fma32 vector|fma32 matrix|'
        r'fma64 vector|fma64 matrix|mac16 vector|mac16 matrix))'
        r'\s+\(([^)]+)\)\s+'
        r'([\d.]+)\s+([\d.]+)'
    )

    in_whole_chip = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Whole chip"):
            in_whole_chip = True
            result["sections"]["Whole chip"] = []
            continue
        if in_whole_chip:
            m = whole_chip_re.match(line)
            if m:
                instr = m.group(1).strip()
                size_info = m.group(2)
                ginstr = float(m.group(3))
                gflops = float(m.group(4))
                result["sections"]["Whole chip"].append({
                    "instruction": instr,
                    "size": size_info,
                    "ns_per_op": None,
                    "ginstr_per_s": ginstr,
                    "gflops": gflops,
                })

    return result


def average_runs(runs: List[dict]) -> dict:
    """Average multiple runs of the same chip."""
    if len(runs) == 1:
        return runs[0]

    base = dict(runs[0])
    base["date"] = f"{runs[0]['date']} (avg of {len(runs)} runs)"
    base["sections"] = {}

    all_sections = set()
    for run in runs:
        all_sections.update(run["sections"].keys())

    for section in all_sections:
        section_data = [run["sections"].get(section, []) for run in runs if section in run["sections"]]
        if not section_data:
            continue

        # Use first run as template
        template = section_data[0]
        averaged = []
        for i, entry in enumerate(template):
            avg_entry = dict(entry)
            # Average numeric fields
            for key in ["ns_per_op", "ginstr_per_s", "gflops"]:
                vals = []
                for sd in section_data:
                    if i < len(sd) and sd[i].get(key) is not None:
                        vals.append(sd[i][key])
                avg_entry[key] = sum(vals) / len(vals) if vals else None
            averaged.append(avg_entry)
        base["sections"][section] = averaged

    return base


def load_all_benchmarks() -> List[dict]:
    """Load and group benchmark text files, averaging runs per chip."""
    txt_files = sorted(RESULTS_DIR.glob("*.txt"))
    if not txt_files:
        raise SystemExit("No benchmark txt files found")

    parsed = [parse_txt_benchmark(f) for f in txt_files]

    # Group by chip name
    by_chip = defaultdict(list)
    for p in parsed:
        by_chip[p["chip_name"]].append(p)

    return [average_runs(runs) for runs in by_chip.values()]


def color_ramp(t: float) -> str:
    """Blue (#1e40af) to teal (#0d9488) to green (#10b981) ramp."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        s = t / 0.5
        r = int(30 + (13 - 30) * s)
        g = int(64 + (148 - 64) * s)
        b = int(175 + (136 - 175) * s)
    else:
        s = (t - 0.5) / 0.5
        r = int(13 + (16 - 13) * s)
        g = int(148 + (185 - 148) * s)
        b = int(136 + (129 - 136) * s)
    return f"#{r:02x}{g:02x}{b:02x}"


def bar_color(t: float) -> str:
    """Color for grouped bars: chip index -> color."""
    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]
    return colors[int(t) % len(colors)]


def build_grouped_bar_svg(
    title: str,
    subtitle: str,
    instructions: List[str],
    chips: List[str],
    data: Dict[str, Dict[str, Optional[float]]],  # instruction -> chip -> value
    unit: str,
    width: int = 1100,
) -> str:
    """Build a grouped horizontal bar chart SVG."""
    n_instr = len(instructions)
    n_chips = len(chips)
    bar_h = 16
    group_gap = 14
    bar_gap = 2
    group_h = n_chips * (bar_h + bar_gap) + group_gap
    margin_left = 260
    margin_top = 80
    margin_bottom = 60
    chart_w = width - margin_left - 120
    height = margin_top + n_instr * group_h + margin_bottom + 40

    # Find max value
    max_val = 0
    for instr in instructions:
        for chip in chips:
            v = data.get(instr, {}).get(chip)
            if v is not None and v > max_val:
                max_val = v
    max_val = max(max_val, 1.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="auto">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="32" font-size="22" font-family="Menlo, monospace" fill="#0f172a">{title}</text>',
        f'<text x="24" y="52" font-size="12" font-family="Menlo, monospace" fill="#475569">{subtitle}</text>',
    ]

    # Legend
    for ci, chip in enumerate(chips):
        lx = margin_left + ci * 220
        parts.append(f'<rect x="{lx}" y="62" width="12" height="12" rx="2" fill="{bar_color(ci)}"/>')
        parts.append(f'<text x="{lx + 16}" y="73" font-size="11" font-family="Menlo, monospace" fill="#334155">{chip}</text>')

    for ii, instr in enumerate(instructions):
        gy = margin_top + ii * group_h
        parts.append(f'<text x="16" y="{gy + (n_chips * (bar_h + bar_gap)) // 2 + 5}" font-size="12" font-family="Menlo, monospace" fill="#111827">{instr}</text>')

        for ci, chip in enumerate(chips):
            v = data.get(instr, {}).get(chip)
            by = gy + ci * (bar_h + bar_gap)
            color = bar_color(ci)

            if v is not None and v > 0:
                bw = chart_w * (v / max_val)
                parts.append(f'<rect x="{margin_left}" y="{by}" width="{bw:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
                parts.append(f'<text x="{margin_left + bw + 6:.1f}" y="{by + 12}" font-size="10" font-family="Menlo, monospace" fill="#334155">{v:.1f} {unit}</text>')
            else:
                parts.append(f'<text x="{margin_left + 4}" y="{by + 12}" font-size="10" font-family="Menlo, monospace" fill="#94a3b8">n/a</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def build_heatmap_svg(
    title: str,
    subtitle: str,
    sections: List[Tuple[str, str]],  # (section_name, display_name)
    chips: List[dict],
    metric: str,  # "gflops" or "ginstr_per_s"
    unit: str,
) -> str:
    """Build a heatmap comparing chips across sections and instructions."""
    # Collect all instructions across all chips and sections
    all_instrs = []
    seen = set()
    for sec_name, _ in sections:
        for chip in chips:
            for entry in chip["sections"].get(sec_name, []):
                key = f"{entry['instruction']} ({entry['size']})" if entry.get('size') else entry['instruction']
                if key not in seen:
                    seen.add(key)
                    all_instrs.append(key)

    n_cols = len(chips)
    n_sections = len(sections)
    cell_w = 180
    cell_h = 22
    section_gap = 30
    left = 260
    top = 110
    col_header_h = 50

    # Count total rows
    total_rows = 0
    section_rows = []
    for sec_name, sec_display in sections:
        instrs_in_section = []
        seen_sec = set()
        for chip in chips:
            for entry in chip["sections"].get(sec_name, []):
                key = f"{entry['instruction']} ({entry['size']})" if entry.get('size') else entry['instruction']
                if key not in seen_sec:
                    seen_sec.add(key)
                    instrs_in_section.append(key)
        section_rows.append((sec_name, sec_display, instrs_in_section))
        total_rows += len(instrs_in_section)

    width = left + cell_w * n_cols + 60
    height = top + total_rows * cell_h + n_sections * section_gap + 80

    # Collect all values for color scaling
    all_vals = []
    for chip in chips:
        for sec_name, _ in sections:
            for entry in chip["sections"].get(sec_name, []):
                v = entry.get(metric)
                if v is not None and v > 0:
                    all_vals.append(v)

    if all_vals:
        min_log = math.log10(min(all_vals))
        max_log = math.log10(max(all_vals))
    else:
        min_log, max_log = 0, 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="auto">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="36" font-size="22" font-family="Menlo, monospace" fill="#0f172a">{title}</text>',
        f'<text x="24" y="56" font-size="12" font-family="Menlo, monospace" fill="#475569">{subtitle}</text>',
    ]

    # Column headers
    for ci, chip in enumerate(chips):
        x = left + ci * cell_w + 4
        parts.append(f'<text x="{x}" y="82" font-size="12" font-family="Menlo, monospace" fill="#111827">{chip["chip_name"]}</text>')
        parts.append(f'<text x="{x}" y="96" font-size="10" font-family="Menlo, monospace" fill="#6b7280">{chip["cores"]} cores</text>')

    y = top
    for sec_name, sec_display, instrs in section_rows:
        # Section header
        parts.append(f'<text x="16" y="{y + 14}" font-size="13" font-family="Menlo, monospace" font-weight="bold" fill="#0f172a">{sec_display}</text>')
        y += 20

        for instr_key in instrs:
            parts.append(f'<text x="24" y="{y + 15}" font-size="10" font-family="Menlo, monospace" fill="#334155">{instr_key}</text>')

            for ci, chip in enumerate(chips):
                x = left + ci * cell_w
                # Find matching entry
                v = None
                for entry in chip["sections"].get(sec_name, []):
                    ek = f"{entry['instruction']} ({entry['size']})" if entry.get('size') else entry['instruction']
                    if ek == instr_key:
                        v = entry.get(metric)
                        break

                if v is not None and v > 0:
                    lv = math.log10(v)
                    t = (lv - min_log) / (max_log - min_log) if max_log > min_log else 0.5
                    t = max(0.0, min(1.0, t))
                    fill = color_ramp(t)
                    parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 6}" height="{cell_h - 2}" rx="3" fill="{fill}"/>')
                    parts.append(f'<text x="{x + 6}" y="{y + 14}" font-size="10" font-family="Menlo, monospace" fill="#f8fafc">{v:.1f} {unit}</text>')
                else:
                    parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 6}" height="{cell_h - 2}" rx="3" fill="#f1f5f9"/>')
                    parts.append(f'<text x="{x + 6}" y="{y + 14}" font-size="10" font-family="Menlo, monospace" fill="#94a3b8">—</text>')

            y += cell_h
        y += section_gap

    # Legend
    ly = y + 10
    steps = 8
    for i in range(steps):
        t = i / (steps - 1)
        fill = color_ramp(t)
        lx = left + i * 40
        parts.append(f'<rect x="{lx}" y="{ly}" width="36" height="14" rx="2" fill="{fill}"/>')
    parts.append(f'<text x="{left}" y="{ly + 28}" font-size="11" font-family="Menlo, monospace" fill="#334155">low</text>')
    parts.append(f'<text x="{left + 40 * (steps - 1)}" y="{ly + 28}" font-size="11" font-family="Menlo, monospace" fill="#334155">high</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    benchmarks = load_all_benchmarks()

    if not benchmarks:
        raise SystemExit("No benchmark data found")

    print(f"Loaded {len(benchmarks)} chip(s):")
    for b in benchmarks:
        secs = list(b["sections"].keys())
        print(f"  {b['chip_name']} — {b['cores']} cores — sections: {secs}")

    # 1. Single P-core GFLOPS comparison (grouped bar chart)
    instructions = [
        "fma16 matrix",
        "mac16 matrix",
        "fma32 matrix",
        "fma16 vector",
        "mac16 vector",
        "fma64 matrix",
        "fma32 vector",
        "fma64 vector",
    ]
    chip_names = [b["chip_name"] for b in benchmarks]

    pcore_data = {}
    for instr in instructions:
        pcore_data[instr] = {}
        for b in benchmarks:
            for entry in b["sections"].get("Single P-core", []):
                if entry["instruction"] == instr and entry["gflops"] is not None:
                    pcore_data[instr][b["chip_name"]] = entry["gflops"]

    svg1 = build_grouped_bar_svg(
        "Single P-core GFLOPS",
        "Peak throughput per P-core for each AMX instruction. Higher is better.",
        instructions,
        chip_names,
        pcore_data,
        "GFLOPS",
    )
    p1 = FIGURES_DIR / "single-pcore-gflops.svg"
    p1.write_text(svg1)
    print(f"Wrote {p1}")

    # 2. Whole chip GFLOPS comparison (grouped bar chart)
    whole_data = {}
    for instr in instructions:
        whole_data[instr] = {}
        for b in benchmarks:
            for entry in b["sections"].get("Whole chip", []):
                if entry["instruction"] == instr and entry["gflops"] is not None:
                    whole_data[instr][b["chip_name"]] = entry["gflops"]

    svg2 = build_grouped_bar_svg(
        "Whole Chip GFLOPS",
        "Aggregate throughput using all cores. Higher is better.",
        instructions,
        chip_names,
        whole_data,
        "GFLOPS",
    )
    p2 = FIGURES_DIR / "whole-chip-gflops.svg"
    p2.write_text(svg2)
    print(f"Wrote {p2}")

    # 3. All-P-cores parallel GFLOPS
    allp_data = {}
    for instr in instructions:
        allp_data[instr] = {}
        for b in benchmarks:
            for entry in b["sections"].get("All P-cores parallel", []):
                if entry["instruction"] == instr and entry["gflops"] is not None:
                    allp_data[instr][b["chip_name"]] = entry["gflops"]

    svg3 = build_grouped_bar_svg(
        "All P-cores Parallel GFLOPS",
        "Throughput with all P-cores running in parallel. Higher is better.",
        instructions,
        chip_names,
        allp_data,
        "GFLOPS",
    )
    p3 = FIGURES_DIR / "all-pcores-parallel-gflops.svg"
    p3.write_text(svg3)
    print(f"Wrote {p3}")

    # 4. Heatmap across all sections
    section_list = [
        ("Single P-core", "Single P-core"),
        ("Single E-core", "Single E-core"),
        ("All P-cores parallel", "All P-cores parallel"),
        ("All E-cores parallel", "All E-cores parallel"),
        ("Whole chip", "Whole chip"),
    ]

    svg4 = build_heatmap_svg(
        "AMX Instruction Throughput Heatmap",
        "GFLOPS across all sections and chips. Darker green = faster.",
        section_list,
        benchmarks,
        "gflops",
        "GF",
    )
    p4 = FIGURES_DIR / "instruction-throughput-heatmap.svg"
    p4.write_text(svg4)
    print(f"Wrote {p4}")

    # 5. Update the chip-function-throughput-heatmap.svg with instruction data
    # (keeping the original name for backward compat)
    whole_and_pcore = [
        ("Single P-core", "Single P-core"),
        ("Whole chip", "Whole chip (all cores)"),
    ]
    svg5 = build_heatmap_svg(
        "AMX Function Throughput Across Apple Silicon",
        "GFLOPS for each instruction. Darker green = higher throughput.",
        whole_and_pcore,
        benchmarks,
        "gflops",
        "GF",
    )
    p5 = FIGURES_DIR / "chip-function-throughput-heatmap.svg"
    p5.write_text(svg5)
    print(f"Wrote {p5} (updated)")


if __name__ == "__main__":
    main()
