#!/usr/bin/env python3
"""
Shared harness for the three benchmark scripts.

What it adds to each benchmark:
  * Warm-up: every framework runs once untimed before measurement starts.
  * Repeats: N timed runs per framework (default 10), with the framework order
    rotated on every repeat so no framework always runs first.
  * Records: the exact settings, library versions, hardware and dataset details
    are saved with the results in results/<benchmark>_<timestamp>.json, and
    every individual run is saved in results/<benchmark>_<timestamp>_runs.csv.

Spark's settings live here, in one place (SPARK_MASTER and SPARK_CONF), and
are used by all three benchmarks.
"""
import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata

import psutil

RESULTS_DIR = "results"

# ── Spark settings used by every benchmark ───────────────────────────────────
SPARK_MASTER = "local[*]"
SPARK_CONF = {
    # 2g leaves headroom on an 8 GB machine; the 10M-row dataset needs far less.
    # (The original 50M-row runs used 4g.)
    "spark.driver.memory": "2g",
    "spark.sql.shuffle.partitions": "8",
}

# Recorded with their effective values even when left at Spark's default.
SPARK_KEYS_TO_REPORT = [
    "spark.master",
    "spark.driver.memory",
    "spark.default.parallelism",
    "spark.sql.shuffle.partitions",
    "spark.sql.adaptive.enabled",
    "spark.sql.adaptive.coalescePartitions.enabled",
    "spark.sql.files.maxPartitionBytes",
    "spark.sql.files.openCostInBytes",
    "spark.sql.parquet.filterPushdown",
    "spark.sql.parquet.enableVectorizedReader",
    "spark.sql.parquet.columnarReaderBatchSize",
]

# Environment variables that can change how many threads or how much memory
# a framework uses. Recorded as null when not set.
ENV_VARS_TO_REPORT = [
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "PYSPARK_SUBMIT_ARGS",
    "PYSPARK_PYTHON",
    "SPARK_HOME",
    "SPARK_CONF_DIR",
    "JAVA_HOME",
    "JAVA_TOOL_OPTIONS",
]

LIBRARIES = ["polars", "pandas", "pyarrow", "numpy", "pyspark", "py4j", "psutil"]

_spark = None
_spark_info = None


# ── Measurement helpers ──────────────────────────────────────────────────────
def mem_mb():
    """Resident memory of this Python process in MB (excludes Spark's JVM)."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def get_spark(app_name):
    """Start Spark once per script, outside any timed block, and reuse it."""
    global _spark, _spark_info
    if _spark is None:
        from pyspark.sql import SparkSession
        builder = SparkSession.builder.appName(app_name).master(SPARK_MASTER)
        for key, value in SPARK_CONF.items():
            builder = builder.config(key, value)
        _spark = builder.getOrCreate()
        _spark.sparkContext.setLogLevel("ERROR")
        _spark_info = _describe_spark(_spark)
    return _spark


def stop_spark():
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None


# ── Environment capture ──────────────────────────────────────────────────────
def _sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _describe_machine(data_path):
    vm = psutil.virtual_memory()
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_gib": round(vm.total / 1024**3, 1),
        "available_ram_gib_at_start": round(vm.available / 1024**3, 1),
    }
    if platform.system() == "Darwin":
        info["macos_version"] = platform.mac_ver()[0]
        info["cpu_model"] = _sysctl("machdep.cpu.brand_string")
        # Apple Silicon: performance cores (perflevel0) and efficiency cores (perflevel1)
        info["performance_cores"] = _sysctl("hw.perflevel0.physicalcpu")
        info["efficiency_cores"] = _sysctl("hw.perflevel1.physicalcpu")
    else:
        info["cpu_model"] = platform.processor() or None

    # Which disk the dataset is on (for example an external SSD)
    try:
        real = os.path.realpath(data_path)
        parts = psutil.disk_partitions(all=True)
        best = max((p for p in parts if real.startswith(p.mountpoint)),
                   key=lambda p: len(p.mountpoint), default=None)
        if best is not None:
            info["dataset_volume"] = {"mountpoint": best.mountpoint,
                                      "device": best.device,
                                      "filesystem": best.fstype}
    except Exception:
        pass
    return info


def _versions():
    out = {}
    for name in LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _describe_polars():
    try:
        import polars as pl
        size = getattr(pl, "thread_pool_size", None) or getattr(pl, "threadpool_size")
        return {"thread_pool_size": size()}
    except Exception as e:
        return {"error": str(e)}


def _describe_pandas_and_arrow():
    info = {}
    try:
        import pandas as pd
        info["pandas_parquet_engine_option"] = pd.get_option("io.parquet.engine")
    except Exception as e:
        info["pandas_error"] = str(e)
    try:
        import pyarrow as pa
        info["pyarrow_cpu_threads"] = pa.cpu_count()
        info["pyarrow_io_threads"] = pa.io_thread_count()
    except Exception as e:
        info["pyarrow_error"] = str(e)
    return info


def describe_dataset(path):
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(path).metadata
    info = {
        "path": os.path.abspath(path),
        "size_bytes": os.path.getsize(path),
        "rows": meta.num_rows,
        "columns": meta.num_columns,
        "row_groups": meta.num_row_groups,
        "created_by": meta.created_by,
    }
    if meta.num_row_groups:
        info["rows_per_row_group_first"] = meta.row_group(0).num_rows
        info["compression"] = meta.row_group(0).column(0).compression
    return info


def _describe_spark(spark):
    sc = spark.sparkContext
    info = {
        "spark_version": spark.version,
        "master": sc.master,
        "default_parallelism": sc.defaultParallelism,
        "set_by_benchmark": dict(SPARK_CONF, **{"spark.master": SPARK_MASTER}),
        "effective": {},
        # Everything Spark reports as explicitly set, including values Spark
        # fills in itself at startup (app ID, driver host and port, and so on).
        "all_explicitly_set": dict(sorted(sc.getConf().getAll())),
    }
    for key in SPARK_KEYS_TO_REPORT:
        try:
            info["effective"][key] = spark.conf.get(key)
        except Exception:
            info["effective"][key] = None  # not set: Spark's built-in default applies
    try:
        jvm = sc._jvm
        info["jvm_max_heap_gib"] = round(
            jvm.java.lang.Runtime.getRuntime().maxMemory() / 1024**3, 2)
        info["java_version"] = jvm.java.lang.System.getProperty("java.version")
        info["java_vendor"] = jvm.java.lang.System.getProperty("java.vendor")
    except Exception as e:
        info["jvm_error"] = str(e)
    return info


def capture_environment(data_path):
    return {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable] + sys.argv),
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "machine": _describe_machine(data_path),
        "libraries": _versions(),
        "polars": _describe_polars(),
        "pandas_pyarrow": _describe_pandas_and_arrow(),
        "environment_variables": {k: os.environ.get(k) for k in ENV_VARS_TO_REPORT},
        "dataset": describe_dataset(data_path),
        "spark": None,  # filled in once Spark has started
    }


# ── Runner ───────────────────────────────────────────────────────────────────
def _parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--repeats", type=int,
                   default=int(os.environ.get("BENCH_REPEATS", 10)),
                   help="timed runs per framework (default 10, or $BENCH_REPEATS)")
    p.add_argument("--warmup", type=int,
                   default=int(os.environ.get("BENCH_WARMUP", 1)),
                   help="untimed warm-up runs per framework (default 1, or $BENCH_WARMUP)")
    p.add_argument("--parquet", default="transactions.parquet",
                   help="input file (default transactions.parquet)")
    p.add_argument("--results-dir", default=RESULTS_DIR,
                   help="where to save results (default results/)")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    if args.warmup < 0:
        p.error("--warmup cannot be negative")
    return args


def run_benchmark(script_file, title, frameworks):
    """
    frameworks: list of (name, fn). Each fn(parquet_path) runs one measurement
    and returns {"elapsed": s, "mem_mb": MB, "rows": output rows,
    optional "sample": printable text}.
    """
    args = _parse_args(title)
    if not os.path.exists(args.parquet):
        print(f"ERROR: {args.parquet} not found — run generate_data.py first.")
        raise SystemExit(1)

    env = capture_environment(args.parquet)
    input_rows = env["dataset"]["rows"]
    names = [name for name, _ in frameworks]
    fns = dict(frameworks)
    runs, samples = [], {}

    def _record(result, name, phase, repeat, position):
        runs.append({
            "phase": phase,
            "repeat": repeat,
            "position": position,
            "framework": name,
            "elapsed_s": round(result["elapsed"], 4),
            "mem_delta_mb": round(result["mem_mb"], 1),
            "rows_out": result["rows"],
        })

    print(f"\n  {title}")
    print(f"  {args.warmup} warm-up + {args.repeats} timed run(s) per framework, "
          f"order rotated each repeat\n")
    failed = set()
    try:
        for w in range(1, args.warmup + 1):
            for pos, name in enumerate(names, 1):
                if name in failed:
                    continue
                print(f"  warm-up {w}/{args.warmup}  {name:<18}", end=" ", flush=True)
                try:
                    r = fns[name](args.parquet)
                except MemoryError:
                    failed.add(name)
                    print("OOM — skipped for remaining runs", flush=True)
                    gc.collect()
                    continue
                print(f"{r['elapsed']:7.2f}s  (not counted)", flush=True)
                _record(r, name, "warmup", w, pos)
        for i in range(args.repeats):
            k = i % len(names)
            order = names[k:] + names[:k]
            for pos, name in enumerate(order, 1):
                if name in failed:
                    continue
                print(f"  repeat  {i + 1}/{args.repeats}  {name:<18}", end=" ", flush=True)
                try:
                    r = fns[name](args.parquet)
                except MemoryError:
                    failed.add(name)
                    print("OOM — skipped for remaining runs", flush=True)
                    gc.collect()
                    continue
                print(f"{r['elapsed']:7.2f}s", flush=True)
                _record(r, name, "timed", i + 1, pos)
                if "sample" in r:
                    samples[name] = r["sample"]
    finally:
        stop_spark()

    env["spark"] = _spark_info
    summary = _summarize(runs, names, input_rows)
    _print_summary(title, summary, input_rows, args, samples)
    json_path, csv_path = _write_results(script_file, title, args, env, summary, runs)
    print(f"  Saved: {json_path}")
    print(f"         {csv_path}\n")


def _summarize(runs, names, input_rows):
    summary = []
    for name in names:
        timed = [r for r in runs if r["framework"] == name and r["phase"] == "timed"]
        if not timed:
            summary.append({
                "framework": name, "status": "oom", "timed_runs": 0,
                "median_s": None, "mean_s": None, "stdev_s": None,
                "min_s": None, "max_s": None,
                "input_rows_per_sec_median": None,
                "rows_out": None, "first_run_mem_delta_mb": None,
                "times_slower_than_fastest": None,
            })
            continue
        all_fw = [r for r in runs if r["framework"] == name]
        first  = all_fw[0]
        t      = [r["elapsed_s"] for r in timed]
        median = statistics.median(t)
        summary.append({
            "framework": name, "status": "ok",
            "timed_runs": len(t),
            "median_s": round(median, 4),
            "mean_s": round(statistics.mean(t), 4),
            "stdev_s": round(statistics.stdev(t), 4) if len(t) > 1 else 0.0,
            "min_s": min(t),
            "max_s": max(t),
            "input_rows_per_sec_median": round(input_rows / median),
            "rows_out": timed[-1]["rows_out"],
            "first_run_mem_delta_mb": first["mem_delta_mb"],
        })
    ok = [s for s in summary if s["status"] == "ok"]
    if ok:
        best = min(s["median_s"] for s in ok)
        for s in ok:
            s["times_slower_than_fastest"] = round(s["median_s"] / best, 2)
    return summary


def _print_summary(title, summary, input_rows, args, samples):
    ok       = [s for s in summary if s.get("status") == "ok"]
    best     = min(s["median_s"] for s in ok) if ok else None
    first_ok = ok[0] if ok else None
    print()
    print("=" * 84)
    print(f"  {title.upper()}")
    out_str = f"  |  Output: {first_ok['rows_out']:,} rows" if first_ok else ""
    print(f"  Input: {input_rows:,} rows{out_str}  |  {args.repeats} timed run(s) each")
    print("=" * 84)
    print(f"  {'Framework':<20} {'Median':>8}  {'Min–Max':>15}  {'Rows/sec':>13}"
          f"  {'Mem Δ MB*':>9}  Speedup")
    print(f"  {'-'*20} {'-'*8}  {'-'*15}  {'-'*13}  {'-'*9}  {'-'*12}")
    for s in summary:
        if s.get("status") == "oom":
            print(f"  {s['framework']:<20} {'OOM':>8}  {'—':>15}  {'—':>13}  {'—':>9}  —")
            continue
        label = ("◀ fastest" if s["median_s"] == best
                 else f"{s['times_slower_than_fastest']:.1f}x slower")
        span = f"{s['min_s']:.2f}–{s['max_s']:.2f}s"
        print(f"  {s['framework']:<20} {s['median_s']:>7.2f}s  {span:>15}"
              f"  {s['input_rows_per_sec_median']:>11,}/s  {s['first_run_mem_delta_mb']:>9.0f}"
              f"  {label}")
    print("=" * 84)
    print("  * Change in Python process memory during each framework's first run.")
    print("    Excludes Spark's JVM. Later runs reuse memory, so only the first is shown.")

    if ok:
        outs = {s["rows_out"] for s in ok}
        if len(outs) > 1:
            print("  WARNING: frameworks returned different row counts: "
                  + ", ".join(f"{s['framework']}={s['rows_out']:,}" for s in ok))

    if samples and ok:
        winner = min(ok, key=lambda s: s["median_s"])["framework"]
        if winner in samples:
            print(f"\n  Sample output (top 5 rows, from {winner}):")
            print(samples[winner])
    print()


def _write_results(script_file, title, args, env, summary, runs):
    os.makedirs(args.results_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(script_file))[0]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"{stem}_{stamp}.json")
    csv_path = os.path.join(args.results_dir, f"{stem}_{stamp}_runs.csv")

    record = {
        "benchmark": title,
        "script": os.path.basename(script_file),
        "method": {
            "warmup_runs_per_framework": args.warmup,
            "timed_runs_per_framework": args.repeats,
            "order": "rotated each repeat (repeat 1: as listed; repeat 2: shifted by one; ...)",
            "timing": "time.perf_counter() around the whole operation, including the "
                      "Parquet read; Spark startup excluded (Spark starts once per script)",
            "spark_action": "df.write.format('noop').mode('overwrite').save() — computes "
                            "every output column and discards the rows; nothing is written",
            "memory": "Python process RSS change during each framework's first run; "
                      "excludes Spark's JVM",
        },
        "environment": env,
        "summary": summary,
        "runs": runs,
    }
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    if runs:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
            w.writeheader()
            w.writerows(runs)
    return json_path, csv_path
