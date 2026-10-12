# Polars / Pandas / Spark — ETL Performance Benchmarks

Three end-to-end benchmarks that measure how each framework performs across the
typical stages of an ETL pipeline on a single machine. Each script runs every
framework repeatedly, prints a comparison table, and saves the timings together
with the exact settings and environment they came from. A scale sweep script
runs all three benchmarks at multiple dataset sizes to show where each
framework hits its single-machine limits — and where distributing compute
becomes necessary.

---

## Requirements

Python 3.14 · Java 17 (for Spark) · all dependencies isolated in `.venv/`

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the versions used: Polars 1.44.2, pandas 3.0.6,
PySpark 4.2.0, pyarrow 25.0.1, NumPy 2.5.3, psutil 7.2.2.

> PySpark 4.2.0 prints a one-time warning that it does not yet fully support
> pandas 3.x. It does not affect these benchmarks, which never pass data
> between Spark and pandas.

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

With the default 10 repeats, a full `run_all.sh` takes about 5–10 minutes.
Options:

```bash
.venv/bin/python benchmark_1_ingest.py --repeats 3     # quicker check
.venv/bin/python benchmark_1_ingest.py --warmup 0      # skip the warm-up run
BENCH_REPEATS=5 bash run_all.sh                        # 5 repeats for all three
```

To run all three benchmarks across multiple dataset sizes (the scale sweep):

```bash
.venv/bin/python scale_sweep.py                    # 10M, 25M, 50M rows
.venv/bin/python scale_sweep.py --sizes 10 25      # safer first test on 8 GB
.venv/bin/python scale_sweep.py --repeats 3        # faster; 3 repeats per size
```

The sweep reuses `transactions.parquet` for 10 M rows and generates
`transactions_<N>M.parquet` for each additional size. If a framework runs out
of memory it is skipped and labelled OOM; the other two continue.

---

## Project files

| File | Purpose |
|---|---|
| `generate_data.py` | Creates a Parquet dataset; `--rows` sets the size, `--output` sets the path |
| `benchmark_1_ingest.py` | Benchmark 1 — ingestion and schema normalization |
| `benchmark_2_transform.py` | Benchmark 2 — filtering and column derivation |
| `benchmark_3_aggregate.py` | Benchmark 3 — group-by aggregation |
| `scale_sweep.py` | Runs all three benchmarks at multiple dataset sizes; prints a unified table and saves a `sweep_summary.csv` |
| `bench_common.py` | Shared code: Spark settings, warm-up and repeats, OOM handling, environment capture, results table and files |
| `run_all.sh` | Generates the data if missing, then runs all three benchmarks at the default size |
| `results/` | Created on the first run; one JSON and one CSV per benchmark run; sweep results land in `results/sweep_<timestamp>/` |
| `originals/` | The earlier single-run versions of the scripts, kept for reference |

---

## Method

**Repeats and order.** Each framework runs once untimed (warm-up), then 10
timed times. The order rotates on every repeat (Polars → pandas → Spark, then
pandas → Spark → Polars, and so on), so no framework always runs first. The
results table reports the median and the fastest and slowest run.

**Warm cache.** Because of the warm-up run, the Parquet file is normally
already in the operating system's file cache during the timed runs, so the
timings are warm-cache numbers. They include reading and decoding the file but
mostly not the disk itself.

**Timing.** `time.perf_counter()` around the whole operation, including the
Parquet read. Each framework's result is fully computed inside the timed block:

| Framework | What ends the timed block |
|---|---|
| Polars | `.collect()` — builds the full result table |
| pandas | Eager; each step runs as it is called |
| Spark | `df.write.format("noop").mode("overwrite").save()` — computes every output column and discards the rows; nothing is written |

Spark is not timed with `df.count()`. For a row count, Spark drops every step
the count doesn't depend on: it would skip the derived columns in benchmarks 1
and 2, and the five aggregates and the sort in benchmark 3. The original
scripts (in `originals/`) used `count()`, which is why their Spark times were
too low. Output row counts are still checked after the timer stops, and the
script warns if the frameworks disagree.

**Spark startup.** `SparkSession` is started once per script, before any timed
block, and reused for every run, so Spark's ~10 s startup is excluded.

### Settings

Spark's settings are defined once, in `bench_common.py` (`SPARK_MASTER` and
`SPARK_CONF`), and used by all three benchmarks:

| Setting | Value | Spark default |
|---|---|---|
| `spark.master` | `local[*]` (one worker thread per CPU core) | — |
| `spark.driver.memory` | `2g` (the original 50M-row runs used `4g`) | `1g` |
| `spark.sql.shuffle.partitions` | `8` | `200` |

Everything else is at PySpark 4.2.0 defaults, including adaptive query
execution (on). Polars and pandas use their defaults: Polars uses all cores,
and pandas reads Parquet through pyarrow, which reads with multiple threads.

### What is saved

Every run of a benchmark saves two files in `results/`:

- `<benchmark>_<timestamp>.json` — the method, the summary table, every
  individual run, and the environment:
  - Spark settings actually in effect, the JVM heap size and the Java version
  - library versions
  - Polars and pyarrow thread counts
  - CPU model, performance and efficiency core counts, and RAM
  - the disk the dataset is on
  - the dataset's size, row count, row groups and compression
  - relevant environment variables such as `POLARS_MAX_THREADS` and
    `PYSPARK_SUBMIT_ARGS` (recorded as null when not set)
- `<benchmark>_<timestamp>_runs.csv` — one row per run (warm-up and timed),
  for your own analysis.

---

## Dataset — `generate_data.py`

Generates a synthetic retail transactions table and writes it to
`transactions.parquet`: by default 10 million rows (about 240 MB,
Snappy-compressed, 500k rows per row group, so 20 row groups). The random seed
is fixed (42), so the file is the same every time it is generated with the same
number of rows. `--rows` changes the size (see "Dataset size and memory" below).

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

### Dataset size and memory

The default is 10 million rows so that every benchmark fits comfortably on an
8 GB machine (MacBook Air M3) with room left for macOS. Peak memory measured
at 10 million rows, using the pinned library versions:

| Benchmark | Polars | pandas | Whole run, including Spark |
|---|---|---|---|
| 1 — Ingest + normalize | 1.2 GB | 1.8 GB | 3.7 GB |
| 2 — Filter + derive | 0.5 GB | 1.7 GB | 3.3 GB |
| 3 — GroupBy aggregate | 0.5 GB | 1.5 GB | 2.8 GB |

The "whole run" column is the peak of the Python process plus Spark's JVM
during a complete benchmark script. It is higher than any single framework
because the Python process keeps memory from earlier runs, and Spark runs in
its own process (it peaked at 0.9–1.4 GB). Spark's limit is set to
`spark.driver.memory = 2g`.

At larger sizes, frameworks that exceed available RAM raise a `MemoryError`.
`bench_common.py` catches this per-framework: the affected framework is skipped
for all remaining runs and shown as OOM in the summary table; the other two
continue. Estimated crossover points on an 8 GB machine:

| Size | Expected behaviour |
|---|---|
| 10 M rows | All three frameworks complete |
| 25 M rows | pandas likely OOMs during ingest (~4.5 GB peak for pandas alone) |
| 50 M rows | pandas OOMs; Polars and Spark may survive depending on available RAM |

To generate a dataset at a specific size:

```bash
.venv/bin/python generate_data.py                                           # 10M rows → transactions.parquet
.venv/bin/python generate_data.py --rows 5000000                            # 5M rows, if memory is tight
.venv/bin/python generate_data.py --rows 25000000 --output transactions_25M.parquet
.venv/bin/python generate_data.py --rows 50000000 --output transactions_50M.parquet
```

`scale_sweep.py` handles dataset generation automatically for each size it
runs. The benchmarks read the row count from the file, so they need no changes,
and each results file records the row count used.

**How the file is read in parallel.** All three frameworks read the file with
multiple threads:

- **Spark** splits the file into roughly one chunk per core (about 8 on an
  8-core machine, each around 30 MB with the default dataset) and reads the
  chunks in parallel. A row group is the smallest unit a chunk can contain.
- **Polars** reads in parallel by default, across row groups or columns
  (`scan_parquet(parallel="auto")`).
- **pandas** reads through pyarrow, which uses multiple threads. The pandas
  operations that follow run on a single thread.

**Note on filters.** Because every column is randomly distributed, each row
group contains the full range of values (for example, every row group's
`status` spans `completed` to `refunded`, and `amount` spans about 1 to 500).
Parquet's row-group statistics therefore can't rule out any row group for the
benchmark 2 filter; every row group must be read.

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
  single optimized plan before calling `.collect()`. Date casting uses the
  formula `(date_offset + 18262).cast(Date)`, where 18262 is the number of
  days between the Unix epoch and 2020-01-01 — fully vectorized, no Python loop.
- **pandas** — `read_parquet` loads eagerly; `pd.to_timedelta` handles the date
  offset vectorially; string normalization uses the `.str` accessor.
- **Spark** — `spark.read.parquet` reads the file's chunks in parallel across
  cores; `F.date_add`, `F.lower` and `F.trim` run in the JVM.

**What to look for:** every framework must read all 10 columns and rewrite a
string column for all 10 M rows, so this benchmark is dominated by Parquet
decoding and string processing.

---

## Benchmark 2 — Filtering & Column Derivation (`benchmark_2_transform.py`)

**What it tests:** mid-pipeline transformation — narrowing the dataset and
enriching surviving rows with computed columns.

**Operations:**

1. Read `transactions.parquet`.
2. Filter to rows where `status == 'completed'` AND `amount > 50.0`
   (~18% of rows survive → ≈1.8 M output rows).
3. Derive `net_amount = amount × (1 − discount_pct / 100)`.
4. Classify each row into a `segment` label:
   - `net_amount > 300` → `"premium"`
   - `net_amount > 150` → `"standard"`
   - otherwise → `"budget"`
5. Normalize `raw_category` (strip + lowercase) → `category`.
6. Drop the original `raw_category` column.

**Per-framework approach:**

- **Polars** — the `.filter()` is pushed into the Parquet scan, so rows are
  filtered while the file is read rather than after the whole table is built.
  `pl.when/then/otherwise` compiles to a vectorized branch expression.
- **pandas** — reads the whole file into memory, then applies a boolean mask
  followed by `.copy()` to avoid SettingWithCopyWarning; `np.where` for the
  branching segment assignment.
- **Spark** — the filter is pushed down to the Parquet reader; the `F.when`
  chain and string cleanup run in the same stage as the scan.

**What to look for:** the cost of filtering early (Polars and Spark) versus
building the full 10 M-row table first (pandas). Row-group statistics can't
skip any data here (see the note under Dataset), so the savings come from
filtering during the scan, not from reading less of the file.

---

## Benchmark 3 — GroupBy Aggregation (`benchmark_3_aggregate.py`)

**What it tests:** the analytical/reporting phase — collapsing 10 M rows into a
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

The fastest framework's top-5 output rows are printed below the timing table so
you can confirm correctness across frameworks.

**Per-framework approach:**

- **Polars** — `scan_parquet` + `.group_by().agg()` + `.sort()` + `.collect()`.
  The lazy plan reads only the 3 columns it needs (`region`, `status`,
  `amount`) and aggregates with multiple threads in one process.
- **pandas** — `read_parquet` reads all 10 columns, then
  `groupby(..., as_index=False).agg(...)` aggregates on a single thread.
- **Spark** — `.groupBy().agg().orderBy()`. Spark also reads only the 3 needed
  columns. Each task first aggregates its own chunk, then the partial results
  (at most 50 rows per task) are shuffled and combined, and the 50 groups are
  sorted.

**What to look for:** this is compute-bound once the data is read. Polars and
Spark both read 3 of the 10 columns and aggregate in parallel; pandas reads all
10 and aggregates on one thread. Spark's shuffle here moves very little data,
so its extra cost is mainly scheduling the additional stages.

---

## Fairness notes

- **Column reading differs by design.** Polars (lazy) and Spark read only the
  columns a query needs; pandas `read_parquet` reads every column unless told
  otherwise. This is part of what the benchmarks compare (lazy versus eager
  execution), and it matters most in benchmarks 2 and 3.
- **Spark discards its result; Polars and pandas keep theirs in memory.**
  `noop` processes each chunk and drops it. Collecting 10 M rows into Python
  instead would mainly measure the cost of moving data from the JVM to Python.
- **Dataset size affects which frameworks can complete.** The single-benchmark
  scripts default to whatever Parquet file is present. The scale sweep runs
  across multiple sizes and labels any framework that runs out of memory as OOM,
  so comparisons at larger sizes are between whichever frameworks survived.
- **Memory is measured for the Python process only.** Spark does its work in a
  separate JVM, so its memory column is near zero and is not comparable.

---

## Metrics explained

| Metric | How it is measured |
|---|---|
| **Median** | Median of the timed runs |
| **Min–Max** | Fastest and slowest timed run |
| **Rows/sec** | Input rows (from the Parquet metadata) / median seconds |
| **Mem Δ MB** | Change in process RSS (`psutil`) during each framework's first run. Reflects the Python process only — Spark's JVM heap is not captured. Later runs reuse memory already held by the process, so only the first run is shown (all runs are in the CSV) |
| **Speedup** | `median / best_median`; "◀ fastest" marks the winner |

---

## Scale sweep — `scale_sweep.py`

Runs all three benchmarks at multiple dataset sizes in a single command and
prints a unified comparison table. Useful for the core research question: at
what point does a single machine stop being enough, and what does each
framework do when it runs out of memory?

```bash
.venv/bin/python scale_sweep.py                    # default: 10M, 25M, 50M rows
.venv/bin/python scale_sweep.py --sizes 10 25      # two sizes only
.venv/bin/python scale_sweep.py --repeats 3        # 3 timed runs instead of 10
```

**How it works:**

1. Generates `transactions_<N>M.parquet` for each size that isn't already on
   disk (`transactions.parquet` is reused for the 10 M case).
2. Runs each of the three benchmark scripts as a subprocess, passing
   `--parquet`, `--repeats`, `--warmup`, and a per-size results directory.
3. If a benchmark subprocess exits non-zero it is recorded as `fail`.
   Within a subprocess, any framework that raises `MemoryError` is caught,
   skipped for the rest of that benchmark, and recorded as `oom`.
4. Prints the unified table and saves `sweep_summary.csv` to
   `results/sweep_<timestamp>/` for plotting.

The sweep is slower than `run_all.sh` because it generates additional datasets
and runs each benchmark at every size. With `--repeats 3` and three sizes, expect
roughly 3× the time of a single `run_all.sh`.

---

## Results (MacBook Air M3, 8 GB RAM, external SSD)

Median elapsed time in seconds, 10 timed runs each, 1 warm-up, 10 million rows.

| Benchmark | Polars | pandas | Spark (local[*]) |
|---|---|---|---|
| 1 — Ingest + normalize | 1.62 s | 1.56 s | **0.70 s** |
| 2 — Filter + derive | **0.11 s** | 0.96 s | 0.64 s |
| 3 — GroupBy aggregate | **0.18 s** | 0.90 s | 0.50 s |

Source files: `benchmark_1_ingest_20261003-114335.json`,
`benchmark_2_transform_20261003-114529.json`,
`benchmark_3_aggregate_20261003-114713.json`.

**Reading the results.** Spark wins on ingestion because its JVM reads Parquet
chunks in parallel with lower per-row overhead than the Python process. For
compute-bound work (transform, aggregate) Polars is 3–6× faster than Spark on
a single machine — Spark's task-scheduling and stage-planning overhead has no
distributed benefit to offset it here. This is the core finding: Spark is not
an efficient single-machine ETL tool; it becomes worthwhile when data
genuinely outgrows one machine and workers can be added.

Scale sweep results to be added after running `scale_sweep.py` at 10M, 25M,
and 50M rows.

### Superseded results from the original scripts

50 million rows, single runs, Spark timed with `count()` (so it skipped most
of its work), and Polars always run first. Kept only for comparison; do not
cite these.

| Benchmark | Polars | pandas | Spark (`count()`) |
|---|---|---|---|
| 1 — Ingest + normalize | 33 s | 32 s | 4 s |
| 2 — Filter + derive | 18 s | 26 s | 17 s |
| 3 — GroupBy aggregate | 3 s | 28 s | 8 s |
