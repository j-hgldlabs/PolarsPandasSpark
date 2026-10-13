#!/usr/bin/env python3
"""
Scale sweep: runs all three benchmarks at multiple dataset sizes to show how
each framework scales as data grows beyond single-machine comfort zones.

Usage:
    .venv/bin/python scale_sweep.py                    # 10M, 25M, 50M rows
    .venv/bin/python scale_sweep.py --sizes 10 25      # custom sizes (millions)
    .venv/bin/python scale_sweep.py --repeats 3        # fewer repeats (faster)
    .venv/bin/python scale_sweep.py --isolate          # each framework in its own process

With --isolate, every benchmark runs once per framework (--framework polars,
then pandas, then spark), each in a fresh Python process, so no framework
starts with memory another one left behind. The three results are then
combined and ranked as usual.

Memory guide (estimated peak, Python process + Spark JVM combined):
    10M rows  ~3.7 GB   fits comfortably on 8 GB
    25M rows  ~9.2 GB   pandas likely OOMs on 8 GB — shown as OOM in the table
    50M rows  ~18.5 GB  needs 24+ GB; only Polars and Spark are likely to finish

OOM is caught per-framework: the failed framework is skipped and labelled OOM
while the other two continue. A subprocess crash is labelled "fail".

Results are saved under results/sweep_<timestamp>/.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

PYTHON = sys.executable

BENCHMARKS = [
    ("benchmark_1_ingest.py",    "Ingest"),
    ("benchmark_2_transform.py", "Transform"),
    ("benchmark_3_aggregate.py", "Aggregate"),
]

FRAMEWORKS = ["Polars", "Pandas", "Spark (local[*])"]
FW_SHORT   = {"Polars": "Polars", "Pandas": "Pandas", "Spark (local[*])": " Spark"}
FW_KEY     = {"Polars": "polars", "Pandas": "pandas", "Spark (local[*])": "spark"}  # --framework values


def _parquet_path(rows_m):
    # Reuse the default 10M dataset rather than duplicating it.
    return "transactions.parquet" if rows_m == 10 else f"transactions_{rows_m}M.parquet"


def _generate_if_needed(rows_m, parquet):
    if os.path.exists(parquet):
        print(f"  {parquet} already exists, skipping generation")
        return True
    print(f"  Generating {rows_m}M rows → {parquet} …", flush=True)
    r = subprocess.run(
        [PYTHON, "generate_data.py",
         "--rows", str(rows_m * 1_000_000),
         "--output", parquet],
        check=False,
    )
    if r.returncode != 0:
        print(f"  ERROR: generate_data.py failed (exit {r.returncode})")
        return False
    return True


def _run_benchmark(script, parquet, out_dir, repeats, warmup, framework=None):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [PYTHON, script,
           "--parquet", parquet,
           "--repeats", str(repeats),
           "--warmup",  str(warmup),
           "--results-dir", out_dir]
    if framework:
        cmd += ["--framework", framework]
    r = subprocess.run(cmd, check=False)
    return r.returncode == 0


def _load_summary(out_dir, framework=None):
    # Isolated runs are saved as <benchmark>_<framework>_<timestamp>.json
    pattern = f"*_{framework}_*.json" if framework else "*.json"
    jsons = glob.glob(os.path.join(out_dir, pattern))
    if not jsons:
        return None
    with open(sorted(jsons)[-1]) as f:
        return json.load(f).get("summary")


def _rank(entries):
    """Recompute times_slower_than_fastest across separately run frameworks."""
    ok = [e for e in entries if e.get("status") == "ok" and e.get("median_s") is not None]
    if ok:
        best = min(e["median_s"] for e in ok)
        for e in ok:
            e["times_slower_than_fastest"] = round(e["median_s"] / best, 2)
    return entries


def _cell(entry):
    """Format one summary entry as a fixed-width cell."""
    if entry is None:
        return "  fail"
    if entry.get("status") == "oom":
        return "   OOM"
    m = entry.get("median_s")
    return f"{m:5.2f}s" if m is not None else "   OOM"


def _print_table(sizes_m, results, repeats, isolate=False):
    col     = 7                        # chars per framework column
    block_w = len(FRAMEWORKS) * col    # chars for all three frameworks
    total_w = 14 + len(sizes_m) * (block_w + 2)

    bar = "=" * total_w
    print()
    print(bar)
    print("  SCALE SWEEP — MEDIAN ELAPSED TIME (seconds)")
    print(f"  {repeats} timed run(s) per framework  "
          f"|  OOM = out of memory  |  fail = subprocess error")
    print("  " + ("each framework in its own process (--isolate)" if isolate
                  else "all three frameworks share one process per benchmark"))
    print(bar)

    # Size labels centred over each framework block
    line = f"  {'':12}"
    for m in sizes_m:
        line += f"  {f'{m}M rows':^{block_w}}"
    print(line)

    # Framework name header
    line = f"  {'Benchmark':12}"
    for _ in sizes_m:
        for fw in FRAMEWORKS:
            line += f" {FW_SHORT[fw]:>{col - 1}}"
        line += "  "
    print(line.rstrip())

    print(f"  {'-' * 12}" + (f"  {'-' * block_w}" * len(sizes_m)))

    # Data rows
    for _, label in BENCHMARKS:
        line = f"  {label:12}"
        for m in sizes_m:
            summary = results.get((m, label))
            for fw in FRAMEWORKS:
                entry = (next((s for s in summary if s["framework"] == fw), None)
                         if summary is not None else None)
                line += f" {_cell(entry):>{col - 1}}"
            line += "  "
        print(line.rstrip())

    print(bar)
    print()


def _save_csv(sizes_m, results, path):
    rows = []
    for m in sizes_m:
        for _, label in BENCHMARKS:
            summary = results.get((m, label))
            for fw in FRAMEWORKS:
                entry = (next((s for s in summary if s["framework"] == fw), None)
                         if summary is not None else None)
                status = "fail" if entry is None else entry.get("status", "ok")
                rows.append({
                    "size_m":      m,
                    "benchmark":   label,
                    "framework":   fw,
                    "status":      status,
                    "median_s":    entry.get("median_s", "") if entry else "",
                    "rows_per_sec": entry.get("input_rows_per_sec_median", "") if entry else "",
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["size_m", "benchmark", "framework",
                                           "status", "median_s", "rows_per_sec"])
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50], metavar="M",
                   help="dataset sizes in millions of rows (default: 10 25 50)")
    p.add_argument("--repeats", type=int, default=3,
                   help="timed runs per framework per benchmark (default: 3)")
    p.add_argument("--warmup", type=int, default=1,
                   help="warm-up runs per framework (default: 1)")
    p.add_argument("--isolate", action="store_true",
                   help="run each framework in its own process (--framework); "
                        "default: all three share one process per benchmark")
    args = p.parse_args()

    stamp     = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = os.path.join("results", f"sweep_{stamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"\nScale sweep: sizes={args.sizes}M  repeats={args.repeats}  "
          f"warmup={args.warmup}  isolate={args.isolate}")
    print(f"Results: {sweep_dir}\n")

    results = {}

    for size_m in args.sizes:
        parquet = _parquet_path(size_m)
        print(f"\n{'─' * 60}")
        print(f"  {size_m}M rows  →  {parquet}")
        print(f"{'─' * 60}")

        if not _generate_if_needed(size_m, parquet):
            for _, label in BENCHMARKS:
                results[(size_m, label)] = None
            continue

        for script, label in BENCHMARKS:
            bench_dir = os.path.join(sweep_dir, f"{size_m}M_{label.lower()}")
            if args.isolate:
                entries = []
                for fw in FRAMEWORKS:
                    print(f"\n==> {label} — {fw} (own process) …")
                    ok = _run_benchmark(script, parquet, bench_dir, args.repeats,
                                        args.warmup, FW_KEY[fw])
                    part = _load_summary(bench_dir, FW_KEY[fw]) if ok else None
                    if part:
                        entries.extend(part)
                    else:
                        print(f"  WARNING: {script} --framework {FW_KEY[fw]} "
                              f"exited non-zero — recorded as fail")
                results[(size_m, label)] = _rank(entries)
            else:
                print(f"\n==> {label} …")
                ok = _run_benchmark(script, parquet, bench_dir, args.repeats, args.warmup)
                results[(size_m, label)] = _load_summary(bench_dir) if ok else None
                if not ok:
                    print(f"  WARNING: {script} exited non-zero — recorded as fail")

    _print_table(args.sizes, results, args.repeats, args.isolate)

    csv_path = os.path.join(sweep_dir, "sweep_summary.csv")
    _save_csv(args.sizes, results, csv_path)
    print(f"  Sweep CSV: {csv_path}")
    print(f"  Full results: {sweep_dir}/\n")


if __name__ == "__main__":
    main()
