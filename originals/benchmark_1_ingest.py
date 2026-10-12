#!/usr/bin/env python3
"""
Benchmark 1 — Ingestion & Schema Normalization
-----------------------------------------------
Simulates the first stage of an ETL pipeline:
  • Read Parquet from disk
  • Cast an integer offset column to a proper date type
  • Normalize a dirty string column (strip whitespace, lowercase)
  • Compute a derived numeric column  (total_value = amount × quantity)
  • Project to a clean final schema

Run:  python benchmark_1_ingest.py
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
    print("  BENCHMARK 1 — Ingestion & Schema Normalization")
    print(f"  Dataset: {ROWS:,} rows  |  File: {PARQUET}")
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

    # 2020-01-01 is 18262 days after the Unix epoch (1970-01-01)
    EPOCH_OFFSET = 18262

    gc.collect(); mem_before = _mem_mb()
    t0 = time.perf_counter()

    df = (
        pl.scan_parquet(PARQUET)   # lazy — lets the query optimizer work
        .with_columns([
            (pl.col("date_offset").cast(pl.Int32) + EPOCH_OFFSET)
              .cast(pl.Date).alias("txn_date"),
            pl.col("raw_category").str.strip_chars().str.to_lowercase().alias("category"),
            (pl.col("amount") * pl.col("quantity")).alias("total_value"),
        ])
        .select(["transaction_id", "customer_id", "product_id",
                 "region", "status", "category", "txn_date",
                 "amount", "quantity", "discount_pct", "total_value"])
        .collect()   # materialize
    )
    _ = df.shape

    elapsed = time.perf_counter() - t0
    mem_delta = _mem_mb() - mem_before
    return {"name": "Polars", "elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df)}


# ── Pandas ────────────────────────────────────────────────────────────────────
def run_pandas():
    import pandas as pd

    base = pd.Timestamp("2020-01-01")
    gc.collect(); mem_before = _mem_mb()
    t0 = time.perf_counter()

    df = pd.read_parquet(PARQUET)
    df["txn_date"]    = base + pd.to_timedelta(df["date_offset"], unit="D")
    df["category"]    = df["raw_category"].str.strip().str.lower()
    df["total_value"] = df["amount"] * df["quantity"]
    df = df[["transaction_id", "customer_id", "product_id",
             "region", "status", "category", "txn_date",
             "amount", "quantity", "discount_pct", "total_value"]]

    elapsed = time.perf_counter() - t0
    mem_delta = _mem_mb() - mem_before
    return {"name": "Pandas", "elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df)}


# ── Spark ─────────────────────────────────────────────────────────────────────
def run_spark():
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

    spark = (SparkSession.builder
             .appName("benchmark_1_ingest")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    gc.collect(); mem_before = _mem_mb()
    t0 = time.perf_counter()

    df = (spark.read.parquet(PARQUET)
          .withColumn("txn_date",
              F.date_add(F.lit("2020-01-01").cast("date"), F.col("date_offset").cast("int")))
          .withColumn("category",
              F.lower(F.trim(F.col("raw_category"))))
          .withColumn("total_value",
              F.col("amount") * F.col("quantity"))
          .select("transaction_id", "customer_id", "product_id",
                  "region", "status", "category", "txn_date",
                  "amount", "quantity", "discount_pct", "total_value"))
    # force full execution
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
