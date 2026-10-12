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

Run:  python benchmark_1_ingest.py                 (1 warm-up + 10 timed runs each)
      python benchmark_1_ingest.py --repeats 3     (quicker check)
Settings, versions and every run are saved in results/.
"""
import time, gc

import bench_common as common


# ── Polars ────────────────────────────────────────────────────────────────────
def run_polars(parquet):
    import polars as pl

    # 2020-01-01 is 18262 days after the Unix epoch (1970-01-01)
    EPOCH_OFFSET = 18262

    gc.collect(); mem_before = common.mem_mb()
    t0 = time.perf_counter()

    df = (
        pl.scan_parquet(parquet)   # lazy — lets the query optimizer work
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
    mem_delta = common.mem_mb() - mem_before
    return {"elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df)}


# ── Pandas ────────────────────────────────────────────────────────────────────
def run_pandas(parquet):
    import pandas as pd

    base = pd.Timestamp("2020-01-01")
    gc.collect(); mem_before = common.mem_mb()
    t0 = time.perf_counter()

    df = pd.read_parquet(parquet)
    df["txn_date"]    = base + pd.to_timedelta(df["date_offset"], unit="D")
    df["category"]    = df["raw_category"].str.strip().str.lower()
    df["total_value"] = df["amount"] * df["quantity"]
    df = df[["transaction_id", "customer_id", "product_id",
             "region", "status", "category", "txn_date",
             "amount", "quantity", "discount_pct", "total_value"]]

    elapsed = time.perf_counter() - t0
    mem_delta = common.mem_mb() - mem_before
    return {"elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df)}


# ── Spark ─────────────────────────────────────────────────────────────────────
def run_spark(parquet):
    import pyspark.sql.functions as F

    # Started once per script with the settings in bench_common.py (not timed)
    spark = common.get_spark("benchmark_1_ingest")

    gc.collect(); mem_before = common.mem_mb()
    t0 = time.perf_counter()

    df = (spark.read.parquet(parquet)
          .withColumn("txn_date",
              F.date_add(F.lit("2020-01-01").cast("date"), F.col("date_offset").cast("int")))
          .withColumn("category",
              F.lower(F.trim(F.col("raw_category"))))
          .withColumn("total_value",
              F.col("amount") * F.col("quantity"))
          .select("transaction_id", "customer_id", "product_id",
                  "region", "status", "category", "txn_date",
                  "amount", "quantity", "discount_pct", "total_value"))
    # Force full execution. The "noop" writer computes every output column and
    # discards the rows (nothing is written). count() is not used here because
    # Spark would skip the date cast, string cleanup and total_value for it.
    df.write.format("noop").mode("overwrite").save()

    elapsed = time.perf_counter() - t0
    mem_delta = common.mem_mb() - mem_before
    count = df.count()   # row check only, outside the timed block
    return {"elapsed": elapsed, "mem_mb": mem_delta, "rows": count}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    common.run_benchmark(
        __file__,
        "Benchmark 1 — Ingestion & Schema Normalization",
        [("Polars", run_polars), ("Pandas", run_pandas), ("Spark (local[*])", run_spark)],
    )
