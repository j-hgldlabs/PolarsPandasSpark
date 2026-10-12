#!/usr/bin/env python3
"""
Benchmark 2 — Filtering & Column Derivation
--------------------------------------------
Simulates mid-pipeline transformation work:
  • Read Parquet from disk
  • Filter to completed transactions above a revenue threshold
  • Derive net_amount after discount
  • Classify rows into a segment label based on net_amount
  • Clean a string column (strip + lowercase)

The filtered output is ~20% of the input (~10M rows).

Run:  python benchmark_2_transform.py
"""
import os, time, gc
import psutil

PARQUET    = "transactions.parquet"
ROWS       = 50_000_000
MIN_AMOUNT = 50.0


def _mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def _bar(ratio, width=30):
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def print_results(results):
    best_time = min(r["elapsed"] for r in results)
    print()
    print("=" * 72)
    print("  BENCHMARK 2 — Filtering & Column Derivation")
    print(f"  Input: {ROWS:,} rows  |  Output: ~{results[0]['rows']:,} rows")
    print("=" * 72)
    print(f"  {'Framework':<24} {'Time':>8}  {'Rows/sec':>12}  {'Mem Δ MB':>10}  Speedup")
    print(f"  {'-'*24} {'-'*8}  {'-'*12}  {'-'*10}  {'-'*22}")
    for r in results:
        rps     = ROWS / r["elapsed"]
        speedup = r["elapsed"] / best_time
        label   = "◀ fastest" if r["elapsed"] == best_time else f"{speedup:.1f}x slower"
        print(f"  {r['name']:<24} {r['elapsed']:>7.2f}s  {rps:>10,.0f}/s  {r['mem_mb']:>9.0f}   {label}")
    print("=" * 72)
    print()


# ── Polars ────────────────────────────────────────────────────────────────────
def run_polars():
    import polars as pl

    gc.collect(); mem_before = _mem_mb()
    t0 = time.perf_counter()

    df = (
        pl.scan_parquet(PARQUET)
        .filter(
            (pl.col("status") == "completed") &
            (pl.col("amount") > MIN_AMOUNT)
        )
        .with_columns([
            (pl.col("amount") * (1 - pl.col("discount_pct") / 100)).alias("net_amount"),
            pl.col("raw_category").str.strip_chars().str.to_lowercase().alias("category"),
        ])
        .with_columns(
            pl.when(pl.col("net_amount") > 300).then(pl.lit("premium"))
             .when(pl.col("net_amount") > 150).then(pl.lit("standard"))
             .otherwise(pl.lit("budget"))
             .alias("segment")
        )
        .drop("raw_category")
        .collect()
    )
    _ = df.shape

    elapsed = time.perf_counter() - t0
    mem_delta = _mem_mb() - mem_before
    return {"name": "Polars", "elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df)}


# ── Pandas ────────────────────────────────────────────────────────────────────
def run_pandas():
    import pandas as pd
    import numpy as np

    gc.collect(); mem_before = _mem_mb()
    t0 = time.perf_counter()

    df = pd.read_parquet(PARQUET)
    df = df[(df["status"] == "completed") & (df["amount"] > MIN_AMOUNT)].copy()
    df["net_amount"] = df["amount"] * (1 - df["discount_pct"] / 100)
    df["category"]   = df["raw_category"].str.strip().str.lower()
    df["segment"]    = np.where(df["net_amount"] > 300, "premium",
                       np.where(df["net_amount"] > 150, "standard", "budget"))
    df.drop(columns=["raw_category"], inplace=True)

    elapsed = time.perf_counter() - t0
    mem_delta = _mem_mb() - mem_before
    return {"name": "Pandas", "elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df)}


# ── Spark ─────────────────────────────────────────────────────────────────────
def run_spark():
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

    spark = (SparkSession.builder
             .appName("benchmark_2_transform")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    gc.collect(); mem_before = _mem_mb()
    t0 = time.perf_counter()

    df = (spark.read.parquet(PARQUET)
          .filter((F.col("status") == "completed") & (F.col("amount") > MIN_AMOUNT))
          .withColumn("net_amount",
              F.col("amount") * (1 - F.col("discount_pct") / 100))
          .withColumn("category",
              F.lower(F.trim(F.col("raw_category"))))
          .withColumn("segment",
              F.when(F.col("net_amount") > 300, "premium")
               .when(F.col("net_amount") > 150, "standard")
               .otherwise("budget"))
          .drop("raw_category"))
    count = df.count()

    elapsed = time.perf_counter() - t0
    mem_delta = _mem_mb() - mem_before
    spark.stop()
    return {"name": "Spark (local[*])", "elapsed": elapsed, "mem_mb": mem_delta, "rows": count}


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
