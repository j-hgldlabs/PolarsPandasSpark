# Polars / Pandas / Spark — ETL Performance Benchmarks

Three end-to-end benchmarks that measure how each framework performs across the
typical stages of an ETL pipeline at 50 million rows. Every script produces a
printed comparison table so results can be read directly in the terminal.

---

## Requirements

Python 3.14 · Java 17 (for Spark) · all dependencies isolated in `.venv/`

```bash
python3 -m venv .venv
.venv/bin/pip install polars pandas pyspark pyarrow psutil
```

---

## Quick start

```bash
bash run_all.sh
```

`run_all.sh` generates the dataset on the first run, then executes all three
benchmarks in sequence. Each benchmark can also be run individually:

```bash
.venv/bin/python generate_data.py       # one-time setup
.venv/bin/python benchmark_1_ingest.py
.venv/bin/python benchmark_2_transform.py
.venv/bin/python benchmark_3_aggregate.py
```

---

## Dataset — `generate_data.py`

Generates a synthetic retail transactions table and writes it to
`transactions.parquet` (~1.2 GB, Snappy-compressed, 500k rows per row group).

| Column | Type | Description |
|---|---|---|
| `transaction_id` | int64 | Sequential unique key |
| `customer_id` | int32 | 0–999,999 (1 M customers) |
| `product_id` | int32 | 0–9,999 (10 k products) |
| `region` | string | 10 geographic labels |
| `status` | string | pending / completed / failed / refunded / processing |
| `raw_category` | string | Intentionally dirty: mixed case, leading/trailing spaces |
| `amount` | float32 | Order value, uniform 1–500 |
| `quantity` | int32 | Units ordered, 1–19 |
| `discount_pct` | float32 | Discount %, uniform 0–50 |
| `date_offset` | int32 | Days since 2020-01-01 (0–1,825) |

The row-group size of 500 k means Spark gets 100 independently readable chunks,
which it reads in parallel with `local[*]` (one thread per CPU core).

---

## Benchmark 1 — Ingestion & Schema Normalization (`benchmark_1_ingest.py`)

**What it tests:** the first stage of ETL — reading raw data from disk and
producing a clean, correctly-typed DataFrame ready for downstream work.

**Operations:**

1. Read `transactions.parquet` from disk in full.
2. Cast `date_offset` (int32 days-since-epoch offset) to a native `Date` type.
3. Normalize `raw_category`: strip leading/trailing whitespace, lowercase.
4. Derive `total_value = amount × quantity`.
5. Project to an 11-column clean schema, dropping the original raw column.

**Per-framework approach:**

- **Polars** — `scan_parquet` (lazy scan) chains all transformations into a
  single optimized physical plan before calling `.collect()`. Date casting uses
  the formula `(date_offset + 18262).cast(Date)` where 18262 is the number of
  days between the Unix epoch and 2020-01-01 — fully vectorized, no Python loop.
- **Pandas** — `read_parquet` loads eagerly; `pd.to_timedelta` handles the date
  offset vectorially; string normalization uses the `.str` accessor.
- **Spark** — `spark.read.parquet` distributes row-group reads across cores;
  `F.date_add`, `F.lower`, and `F.trim` run as JVM bytecode; `.count()` forces
  full execution of the plan.

**Key insight:** Spark wins here because it reads the 100 Parquet row groups in
parallel across all CPU cores, saturating I/O from the drive. Polars and Pandas
read sequentially in a single thread at this scale.

---

## Benchmark 2 — Filtering & Column Derivation (`benchmark_2_transform.py`)

**What it tests:** mid-pipeline transformation — narrowing the dataset and
enriching surviving rows with computed columns.

**Operations:**

1. Read `transactions.parquet`.
2. Filter to rows where `status == 'completed'` AND `amount > 50.0`
   (~20% of rows survive → ≈10 M output rows).
3. Derive `net_amount = amount × (1 − discount_pct / 100)`.
4. Classify each row into a `segment` label:
   - `net_amount > 300` → `"premium"`
   - `net_amount > 150` → `"standard"`
   - otherwise → `"budget"`
5. Normalize `raw_category` (strip + lowercase) → `category`.
6. Drop the original `raw_category` column.

**Per-framework approach:**

- **Polars** — `scan_parquet` + `.filter()` allows predicate pushdown so only
  matching rows are decoded from the Parquet pages. `pl.when/then/otherwise`
  compiles to a vectorized branch expression.
- **Pandas** — boolean mask filter followed by `.copy()` to avoid
  SettingWithCopyWarning; `np.where` for the branching segment assignment.
- **Spark** — `.filter()` with compound boolean condition; `F.when` chain for
  segmentation; all executed as a single fused stage.

**Key insight:** Polars and Spark finish in near-identical time because both
exploit predicate pushdown (reading only the rows that pass the filter). Pandas
reads the full dataset into memory before filtering.

---

## Benchmark 3 — GroupBy Aggregation (`benchmark_3_aggregate.py`)

**What it tests:** the analytical/reporting phase — collapsing 50 M rows into a
summary table by two categorical keys.

**Operations:**

1. Read `transactions.parquet`.
2. Group by `region` × `status` (10 regions × 5 statuses = 50 groups).
3. For each group compute:
   - `total_revenue` = `SUM(amount)`
   - `avg_order_value` = `MEAN(amount)`
   - `order_count` = `COUNT(*)`
   - `max_amount`, `min_amount`
4. Sort descending by `total_revenue`.

The winner's top-5 output rows are printed below the timing table so you can
confirm correctness across frameworks.

**Per-framework approach:**

- **Polars** — `scan_parquet` + `.group_by().agg()` + `.sort()` + `.collect()`.
  Polars uses a hash-aggregate with SIMD-accelerated sum/mean over contiguous
  Arrow arrays. No shuffle needed — all data is in-process.
- **Pandas** — `groupby(..., as_index=False).agg(...)`. Single-threaded;
  internally builds a hash table in Python/NumPy.
- **Spark** — `.groupBy().agg().orderBy()`. Even in `local[*]` mode, Spark
  performs an in-process shuffle (hash-partition by group key) before
  aggregating, adding coordination overhead absent in the single-process tools.

**Key insight:** Polars wins decisively. Aggregating 50 M rows into 50 groups
is a compute-bound task where SIMD vectorization and zero-copy Arrow memory
dominate. Spark's shuffle overhead is a liability at this scale; Pandas' lack
of parallelism makes it ~9× slower than Polars.

---

## Metrics explained

| Metric | How it is measured |
|---|---|
| **Time** | `time.perf_counter()` wall-clock around the entire operation including read |
| **Rows/sec** | `50,000,000 / elapsed_seconds` (input rows, not output rows) |
| **Mem Δ MB** | Change in process RSS (`psutil`) before vs. after; reflects Python heap only — Spark's JVM heap is not captured |
| **Speedup** | `elapsed / best_elapsed`; "◀ fastest" marks the winner |

> Spark startup (~10 s) is excluded from all timings. `SparkSession` is
> initialized before the timed block in each benchmark.

---

## Observed results (Apple Silicon, external SSD, 10-core)

| Benchmark | Polars | Pandas | Spark |
|---|---|---|---|
| 1 — Ingest + normalize | 33 s | 32 s | **4 s** |
| 2 — Filter + derive | **18 s** | 26 s | 17 s |
| 3 — GroupBy aggregate | **3 s** | 28 s | 8 s |

Spark is the fastest I/O loader; Polars is the fastest for in-memory compute;
Pandas is the slowest in all three scenarios.
