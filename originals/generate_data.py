#!/usr/bin/env python3
"""Generate a 50M-row synthetic transactions Parquet file for benchmarking."""
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROWS = 50_000_000
OUTPUT = "transactions.parquet"

REGIONS = np.array(['North', 'South', 'East', 'West', 'Central',
                    'Pacific', 'Atlantic', 'Mountain', 'Midwest', 'Southeast'])
STATUSES = np.array(['pending', 'completed', 'failed', 'refunded', 'processing'])
RAW_CATS = np.array([
    '  Electronics ', 'CLOTHING', 'food & beverage', 'Home & Garden ',
    '  SPORTS', 'books', 'BEAUTY ', 'Automotive', 'toys', 'Office Supplies  ',
])

print(f"Generating {ROWS:,} rows → {OUTPUT}")
print("This will take ~30 seconds and produce ~900 MB on disk.\n")
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
