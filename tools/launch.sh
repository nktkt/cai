#!/usr/bin/env bash
# launch.sh - one process per GPU launcher (roadmap M4-M6 run).
#
# UNVERIFIED on real hardware: there are no GPUs / SLURM in the authoring env.
# Sets RANK / WORLD_SIZE / LOCAL_RANK per process (the trainer reads these) and
# runs the per-rank command. Supports SLURM (srun sets SLURM_PROCID/NPROCS) and a
# single-node local fallback over visible GPUs. At 220k GPUs you would drive this
# under SLURM/MPI across the cluster; the offline plans + topology.bin are shared
# (read-only) so every rank just loads its own plan.rankNNNNNN.bin.
set -euo pipefail

PLAN_DIR="${1:?usage: launch.sh <plan_dir> [cmd...]}"
shift || true
CMD=("${@:-bin/trainer}")
TOPO="$PLAN_DIR/topology.bin"

run_one() {  # $1=rank
  local rank="$1"
  local plan
  plan="$(printf '%s/plan.rank%06d.bin' "$PLAN_DIR" "$rank")"
  RANK="$rank" WORLD_SIZE="$WORLD_SIZE" LOCAL_RANK="$LOCAL_RANK" \
    "${CMD[@]}" "$plan" "$TOPO"
}

if [[ -n "${SLURM_PROCID:-}" ]]; then
  # SLURM: this script is invoked once per task by srun.
  export WORLD_SIZE="${SLURM_NPROCS:?}"
  export LOCAL_RANK="${SLURM_LOCALID:-0}"
  run_one "$SLURM_PROCID"
else
  # local fallback: launch one process per visible GPU.
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU="$(nvidia-smi -L | wc -l)"
  else
    echo "no nvidia-smi: launching a single CPU process (analytic backend)" >&2
    NGPU=1
  fi
  export WORLD_SIZE="$NGPU"
  pids=()
  for ((g = 0; g < NGPU; g++)); do
    LOCAL_RANK="$g" CUDA_VISIBLE_DEVICES="$g" run_one "$g" &
    pids+=("$!")
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  exit "$rc"
fi
