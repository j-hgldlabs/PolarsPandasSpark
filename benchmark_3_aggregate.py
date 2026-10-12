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

Run:  python benchmark_3_aggregate.py              (1 warm-up + 10 timed runs each)
      python benchmark_3_aggregate.py --repeats 3  (quicker check)
Settings, versions and every run are saved in results/.
"""
import time, gc

import bench_common as common


# ── Polars ────────────────────────────────────────────────────────────────────
def run_polars(parquet):
    import polars as pl

    gc.collect(); mem_before = common.mem_mb()
    t0 = time.perf_counter()

    df = (
        pl.scan_parquet(parquet)
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
    mem_delta = common.mem_mb() - mem_before

    lines = str(df.head(5)).split("\n")
    sample = "\n".join("    " + l for l in lines)
    return {"elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df), "sample": sample}


# ── Pandas ────────────────────────────────────────────────────────────────────
def run_pandas(parquet):
    import pandas as pd

    gc.collect(); mem_before = common.mem_mb()
    t0 = time.perf_counter()

    df = (pd.read_parquet(parquet)
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
    mem_delta = common.mem_mb() - mem_before

    pd.set_option("display.float_format", "{:.2f}".format)
    lines = str(df.head(5)).split("\n")
    sample = "\n".join("    " + l for l in lines)
    return {"elapsed": elapsed, "mem_mb": mem_delta, "rows": len(df), "sample": sample}


# ── Spark ─────────────────────────────────────────────────────────────────────
def run_spark(parquet):
    import pyspark.sql.functions as F

    # Started once per script with the settings in bench_common.py (not timed)
    spark = common.get_spark("benchmark_3_aggregate")

    gc.collect(); mem_before = common.mem_mb()
    t0 = time.perf_counter()

    df = (spark.read.parquet(parquet)
          .groupBy("region", "status")
          .agg(
              F.sum("amount").alias("total_revenue"),
              F.mean("amount").alias("avg_order_value"),
              F.count("amount").alias("order_count"),
              F.max("amount").alias("max_amount"),
              F.min("amount").alias("min_amount"),
          )
          .orderBy(F.col("total_revenue").desc()))
    # Force full execution. The "noop" writer computes every output column and
    # discards the rows (nothing is written). count() is not used here because
    # Spark would skip the five aggregates and the sort for it.
    df.write.format("noop").mode("overwrite").save()

    elapsed = time.perf_counter() - t0
    mem_delta = common.mem_mb() - mem_before

    # Row check and sample output, outside the timed block
    count = df.count()
    rows = df.limit(5).collect()
    header = f"    {'region':<12} {'status':<12} {'total_revenue':>14} {'avg_order':>10} {'count':>10}"
    sep    = "    " + "-" * 62
    lines  = [header, sep]
    for row in rows:
        lines.append(f"    {row.region:<12} {row.status:<12} "
                     f"{row.total_revenue:>14,.2f} {row.avg_order_value:>10.2f} {row.order_count:>10,}")
    sample = "\n".join(lines)

    return {"elapsed": elapsed, "mem_mb": mem_delta, "rows": count, "sample": sample}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    common.run_benchmark(
        __file__,
        "Benchmark 3 — GroupBy Aggregation (region × status)",
        [("Polars", run_polars), ("Pandas", run_pandas), ("Spark (local[*])", run_spark)],
    )
