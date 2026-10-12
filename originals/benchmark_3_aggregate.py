#!/usr/bin/env python3
"""
Benchmark 3 — GroupBy Aggregation
-----------------------------------
Simulates the analytical/reporting phase of ETL:
  • Read Parquet from disk
  • Group by region × status (50 groups)
  • Compute: total_revenue, avg_order_value, order_count, max_amount, min_amount
  • Sort results by total_revenue descending

This is a full shuffle/aggregation — the most CPU & memory intensive stage.

Run:  python benchmark_3_aggregate.py
"""
import os, time, gc
import psutil

PARQUET = "transactions.parquet"
ROWS    = 50_000_000


def _mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def print_results(results):
    best_time = min(r["elapsed"] for r in results)
    print()
    print("=" * 72)
    print("  BENCHMARK 3 — GroupBy Aggregation (region × status)")
    print(f"  Input: {ROWS:,} rows  →  ~50 output groups")
    print("=" * 72)
    print(f"  {'Framework':<24} {'Time':>8}  {'Rows/sec':>12}  {'Mem Δ MB':>10}  Speedup")
    print(f"  {'-'*24} {'-'*8}  {'-'*12}  {'-'*10}  {'-'*22}")
    for r in results:
        rps     = ROWS / r["elapsed"]
        speedup = r["elapsed"] / best_time
        label   = "◀ fastest" if r["elapsed"] == best_time else f"{speedup:.1f}x slower"
        print(f"  {r['name']:<24} {r['elapsed']:>7.2f}s  {rps:>10,.0f}/s  {r['mem_mb']:>9.0f}   {label}")
    print("=" * 72)

    # Print the actual aggregated output (from Polars, smallest repr)
    print()
    print("  Sample output (top 5 rows, from fastest framework):")
    winner = min(results, key=lambda r: r["elapsed"])
    print(winner["sample"])
    print()


# ── Polars ────────────────────────────────────────────────────────────────────
def run_polars():
    import polars as pl

    gc.collect(); mem_before = _mem_mb()
    t0 = time.perf_counter()

    df = (
        pl.scan_parquet(PARQUET)
        .group_by(["region", "status"])
        .agg([
            pl.col("amount").sum().alias("total_revenue"),
            pl.col("amount").mean().alias("avg_order_value"),
            pl.len().alias("order_count"),
            pl.col("amount").max().alias("max_amount"),
            pl.col("amount").min().alias("min_amount"),
        ])
        .sort("total_revenue", descending=True)
        .collect()
    )
    _ = df.shape

    elapsed = time.perf_counter() - t0
    mem_delta = _mem_mb() - mem_before

    lines = str(df.head(5)).split("\n")
    sample = "\n".join("    " + l for l in lines)
    return {"name": "Polars", "elapsed": elapsed, "mem_mb": mem_delta,
            "rows": len(df), "sample": sample}


# ── Pandas ────────────────────────────────────────────────────────────────────
def run_pandas():
    import pandas as pd

    gc.collect(); mem_before = _mem_mb()
    t0 = time.perf_counter()

    df = (pd.read_parquet(PARQUET)
            .groupby(["region", "status"], as_index=False)
            .agg(
                total_revenue  = ("amount", "sum"),
                avg_order_value= ("amount", "mean"),
                order_count    = ("amount", "count"),
                max_amount     = ("amount", "max"),
                min_amount     = ("amount", "min"),
            )
            .sort_values("total_revenue", ascending=False)
            .reset_index(drop=True))

    elapsed = time.perf_counter() - t0
    mem_delta = _mem_mb() - mem_before

    pd.set_option("display.float_format", "{:.2f}".format)
    lines = str(df.head(5)).split("\n")
    sample = "\n".join("    " + l for l in lines)
    return {"name": "Pandas", "elapsed": elapsed, "mem_mb": mem_delta,
            "rows": len(df), "sample": sample}


# ── Spark ─────────────────────────────────────────────────────────────────────
def run_spark():
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

    spark = (SparkSession.builder
             .appName("benchmark_3_aggregate")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    gc.collect(); mem_before = _mem_mb()
    t0 = time.perf_counter()

    df = (spark.read.parquet(PARQUET)
          .groupBy("region", "status")
          .agg(
              F.sum("amount").alias("total_revenue"),
              F.mean("amount").alias("avg_order_value"),
              F.count("amount").alias("order_count"),
              F.max("amount").alias("max_amount"),
              F.min("amount").alias("min_amount"),
          )
          .orderBy(F.col("total_revenue").desc()))
    count = df.count()

    elapsed = time.perf_counter() - t0
    mem_delta = _mem_mb() - mem_before

    rows = df.limit(5).collect()
    header = f"    {'region':<12} {'status':<12} {'total_revenue':>14} {'avg_order':>10} {'count':>10}"
    sep    = "    " + "-" * 62
    lines  = [header, sep]
    for row in rows:
        lines.append(f"    {row.region:<12} {row.status:<12} "
                     f"{row.total_revenue:>14,.2f} {row.avg_order_value:>10.2f} {row.order_count:>10,}")
    sample = "\n".join(lines)

    spark.stop()
    return {"name": "Spark (local[*])", "elapsed": elapsed, "mem_mb": mem_delta,
            "rows": count, "sample": sample}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(PARQUET):
        print(f"ERROR: {PARQUET} not found — run generate_data.py first.")
        raise SystemExit(1)

    results = []
    for name, fn in [("Polars", run_polars), ("Pandas", run_pandas), ("Spark", run_spark)]:
        print(f"  Running {name}...", flush=True)
        results.append(fn())

    print_results(results)
