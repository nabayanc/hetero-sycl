#!/usr/bin/env bash
set -euo pipefail

# ─── CONFIG ───────────────────────────────────────────────────────────────

BUILD_DIR=build
CREATECSR=${BUILD_DIR}/createcsr
SPMV_CLI=${BUILD_DIR}/apps/spmv_cli/spmv_cli

DATA_DIR=benchmarks/datasets
RESULT_DIR=benchmarks_new/results

mkdir -p RESULT_DIR

ITERATIONS=10

SIZES=(1000 5000 10000 50000 100000)
DENSITIES=(0.0001 0.001 0.01)
SCHEDULERS=(static static_block chunked_rr locality_block feedback workstealing dynamic bandit)

# ─── PREP ─────────────────────────────────────────────────────────────────

command -v "${CREATECSR}" >/dev/null 2>&1 || { echo >&2 "Error: ${CREATECSR} not found."; exit 1; }
command -v "${SPMV_CLI}" >/dev/null 2>&1 || { echo >&2 "Error: ${SPMV_CLI} not found."; exit 1; }

mkdir -p "${DATA_DIR}" "${RESULT_DIR}"

# ─── EXPERIMENTS ────────────────────────────────────────────────────────────

for SIZE in "${SIZES[@]}"; do
  for D in "${DENSITIES[@]}"; do
    DSTR=$(printf "%s" "$D" | sed 's/\./_/')
    MTX="${DATA_DIR}/csr${SIZE}_${DSTR}.mtx"

    # 1) generate matrix once
    if [[ ! -f "${MTX}" ]]; then
      echo "==> Generating ${MTX} (rows=${SIZE}, density=${D})"
      "${CREATECSR}" \
        --rows    "${SIZE}" \
        --cols    "${SIZE}" \
        --density "${D}" \
        --output  "${MTX}"
    fi

    # 2) run each scheduler, but skip if already done
    for SCHED in "${SCHEDULERS[@]}"; do
      SUM="${RESULT_DIR}/summary_${SIZE}_${DSTR}_${SCHED}.csv"
      DEV="${RESULT_DIR}/devices_${SIZE}_${DSTR}_${SCHED}.csv"

      if [[ -s "${SUM}" ]]; then
        echo "==> [${SCHED}] Already benchmarked for ${MTX}, skipping."
        continue
      fi

      echo "==> [${SCHED}] Benchmarking ${MTX}"
        "${SPMV_CLI}" \
          "${MTX}" \
          "${SUM}" \
          "${DEV}" \
          "${ITERATIONS}" \
          "${SCHED}"

      echo "    → ${SUM}, ${DEV}"
    done
  done
done

echo "All experiments completed."
