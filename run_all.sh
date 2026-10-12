#!/usr/bin/env bash
# Run all benchmarks in sequence.
# Usage: bash run_all.sh
set -e

PYTHON=".venv/bin/python"
PARQUET="transactions.parquet"

cd "$(dirname "$0")"

if [ ! -f "$PARQUET" ]; then
  echo "==> Generating dataset..."
  $PYTHON generate_data.py
  echo
fi

echo "==> Benchmark 1 — Ingestion & Schema Normalization"
$PYTHON benchmark_1_ingest.py

echo "==> Benchmark 2 — Filtering & Column Derivation"
$PYTHON benchmark_2_transform.py

echo "==> Benchmark 3 — GroupBy Aggregation"
$PYTHON benchmark_3_aggregate.py
