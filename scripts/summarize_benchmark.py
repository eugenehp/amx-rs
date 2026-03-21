#!/usr/bin/env python3

import csv
import pathlib
import statistics
from collections import defaultdict


def main() -> int:
    results_dir = pathlib.Path(__file__).resolve().parents[1] / "benchmark-results"
    csv_files = sorted(results_dir.glob("amx-benchmark-*.csv"))
    if not csv_files:
        raise SystemExit("no benchmark csv files found")

    path = csv_files[-1]
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key in ["m", "n", "k", "tiles"]:
            row[key] = int(row[key])
        for key in [
            "io_only_ns",
            "compute_only_ns",
            "end_to_end_ns",
            "io_split_pct",
            "compute_split_pct",
            "throughput_work_units_per_ns",
        ]:
            row[key] = float(row[key])

    families = defaultdict(list)
    variants = defaultdict(list)
    for row in rows:
        families[row["case_family"]].append(row)
        variants[row["case_variant"]].append(row)

    print(f"CSV {path}")
    print()
    print("FAMILY_AVG_END_TO_END_NS")
    for family, items in sorted(
        families.items(),
        key=lambda item: statistics.mean(row["end_to_end_ns"] for row in item[1]),
    ):
        print(f"{family},{statistics.mean(row['end_to_end_ns'] for row in items):.2f}")

    print()
    print("TOP_VARIANTS_BY_THROUGHPUT")
    for variant, items in sorted(
        variants.items(),
        key=lambda item: statistics.mean(row["throughput_work_units_per_ns"] for row in item[1]),
        reverse=True,
    )[:12]:
        print(f"{variant},{statistics.mean(row['throughput_work_units_per_ns'] for row in items):.6f}")

    print()
    print("TOP_ROWS_BY_THROUGHPUT")
    for row in sorted(rows, key=lambda item: item["throughput_work_units_per_ns"], reverse=True)[:12]:
        print(
            f"{row['case_variant']},{row['workload']},{row['throughput_work_units_per_ns']:.6f},"
            f"{row['end_to_end_ns']:.2f},{row['io_split_pct']:.2f},{row['compute_split_pct']:.2f}"
        )

    print()
    print("TOP_IO_SPLIT")
    for row in sorted(rows, key=lambda item: item["io_split_pct"], reverse=True)[:10]:
        print(
            f"{row['case_variant']},{row['workload']},{row['io_split_pct']:.2f},"
            f"{row['compute_split_pct']:.2f},{row['end_to_end_ns']:.2f}"
        )

    print()
    print("TOP_COMPUTE_SPLIT")
    for row in sorted(rows, key=lambda item: item["compute_split_pct"], reverse=True)[:10]:
        print(
            f"{row['case_variant']},{row['workload']},{row['compute_split_pct']:.2f},"
            f"{row['io_split_pct']:.2f},{row['end_to_end_ns']:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())