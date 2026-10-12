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

The filtered output is ~18% of the input (≈1.8M rows from the default 10M).

Run:  python benchmark_2_transform.py              (1 warm-up + 10 timed runs each)
      python benchmark_2_transform.py --repeats 3  (quicker check)
Settings, versions and every run are saved in results/.
"""
import time, gc

import bench_common as common

MIN_AMOUNT = 50.0


# ── Polars ────────────────────────────────────────────────────────────────────
def run_polars(parquet):
    import polars as pl

    gc.collect(); mem_before = common.mem_mb()
    t0 = time.perf_counter()

    df = (
        pl.scan_parquet(parquet)
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
    mem_delta = common.mem_mb() - mem_before
    return {"elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df)}


# ── Pandas ────────────────────────────────────────────────────────────────────
def run_pandas(parquet):
    import pandas as pd
    import numpy as np

    gc.collect(); mem_before = common.mem_mb()
    t0 = time.perf_counter()

    df = pd.read_parquet(parquet)
    df = df[(df["status"] == "completed") & (df["amount"] > MIN_AMOUNT)].copy()
    df["net_amount"] = df["amount"] * (1 - df["discount_pct"] / 100)
    df["category"]   = df["raw_category"].str.strip().str.lower()
    df["segment"]    = np.where(df["net_amount"] > 300, "premium",
                       np.where(df["net_amount"] > 150, "standard", "budget"))
    df.drop(columns=["raw_category"], inplace=True)

    elapsed = time.perf_counter() - t0
    mem_delta = common.mem_mb() - mem_before
    return {"elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df)}


# ── Spark ─────────────────────────────────────────────────────────────────────
def run_spark(parquet):
    import pyspark.sql.functions as F

    # Started once per script with the settings in bench_common.py (not timed)
    spark = common.get_spark("benchmark_2_transform")

    gc.collect(); mem_before = common.mem_mb()
    t0 = time.perf_counter()

    df = (spark.read.parquet(parquet)
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
    # Force full execution. The "noop" writer computes every output column and
    # discards the rows (nothing is written). count() is not used here because
    # Spark would skip net_amount, category and segment for it.
    df.write.format("noop").mode("overwrite").save()

    elapsed = time.perf_counter() - t0
    mem_delta = common.mem_mb() - mem_before
    count = df.count()   # row check only, outside the timed block
    return {"elapsed": elapsed, "mem_mb": mem_delta, "rows": count}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    common.run_benchmark(
        __file__,
        "Benchmark 2 — Filtering & Column Derivation",
        [("Polars", run_polars), ("Pandas", run_pandas), ("Spark (local[*])", run_spark)],
    )
