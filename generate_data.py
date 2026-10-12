#!/usr/bin/env python3
"""
Generate a synthetic transactions Parquet file for benchmarking.

Default: 10 million rows (about 240 MB on disk). This size is chosen so all
three frameworks fit comfortably in 8 GB of RAM; at 50 million rows pandas
needs roughly 9–10 GB. Use --rows to change it, for example on a machine with
more memory:

    python generate_data.py                    # 10,000,000 rows
    python generate_data.py --rows 50000000    # original 50M-row dataset

The benchmarks read the row count from the file, so they need no changes.
"""
import argparse
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROWS   = 10_000_000
DEFAULT_OUTPUT = "transactions.parquet"

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                    help=f"number of rows to generate (default {DEFAULT_ROWS:,})")
parser.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"output file path (default {DEFAULT_OUTPUT})")
_args  = parser.parse_args()
ROWS   = _args.rows
OUTPUT = _args.output
if ROWS < 1:
    parser.error("--rows must be at least 1")

REGIONS = np.array(['North', 'South', 'East', 'West', 'Central',
                    'Pacific', 'Atlantic', 'Mountain', 'Midwest', 'Southeast'])
STATUSES = np.array(['pending', 'completed', 'failed', 'refunded', 'processing'])
RAW_CATS = np.array([
    '  Electronics ', 'CLOTHING', 'food & beverage', 'Home & Garden ',
    '  SPORTS', 'books', 'BEAUTY ', 'Automotive', 'toys', 'Office Supplies  ',
])

print(f"Generating {ROWS:,} rows → {OUTPUT}")
print(f"Expect about {ROWS * 25 / 1e9:.1f} GB on disk "
      f"(about 25 bytes per row, Snappy-compressed).\n")
t0 = time.perf_counter()

rng = np.random.default_rng(42)

table = pa.table({
    'transaction_id': pa.array(np.arange(ROWS, dtype=np.int64)),
    'customer_id':    pa.array(rng.integers(0, 1_000_000, ROWS, dtype=np.int32)),
    'product_id':     pa.array(rng.integers(0, 10_000,    ROWS, dtype=np.int32)),
    'region':         pa.array(REGIONS[rng.integers(0, len(REGIONS),   ROWS)]),
    'status':         pa.array(STATUSES[rng.integers(0, len(STATUSES), ROWS)]),
    'raw_category':   pa.array(RAW_CATS[rng.integers(0, len(RAW_CATS), ROWS)]),
    'amount':         pa.array(rng.uniform(1.0, 500.0, ROWS).astype(np.float32)),
    'quantity':       pa.array(rng.integers(1, 20, ROWS, dtype=np.int32)),
    'discount_pct':   pa.array(rng.uniform(0.0, 50.0, ROWS).astype(np.float32)),
    # stored as int32 (days since 2020-01-01); benchmarks cast this to a proper date
    'date_offset':    pa.array(rng.integers(0, 1826, ROWS, dtype=np.int32)),
})

pq.write_table(table, OUTPUT, compression='snappy', row_group_size=500_000)

elapsed = time.perf_counter() - t0
import os
size_mb = os.path.getsize(OUTPUT) / 1024**2
print(f"Done in {elapsed:.1f}s  |  {ROWS:,} rows  |  {size_mb:.0f} MB on disk")
