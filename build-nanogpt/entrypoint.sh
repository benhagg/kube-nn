#!/bin/sh
# Run torchrun inside each container so each container can host multiple workers.
set -eu

# Keep values supplied by Kubernetes/Docker available as runtime overrides.
runtime_node_rank="${CONTAINER_NODE_RANK:-}"
runtime_master_addr="${TRAIN_MASTER_ADDR:-}"
runtime_master_port="${TRAIN_MASTER_PORT:-}"

# StatefulSet pod names end in their stable ordinal (for example, -0, -1). Set runtime node env var to the stateful set ordinal.
if [ -z "$runtime_node_rank" ] && [ -n "${POD_NAME:-}" ]; then
    runtime_node_rank="${POD_NAME##*-}"
fi

set -a
. /app/.env
set +a

NODE_RANK="${runtime_node_rank:-$NODE_RANK}"
MASTER_ADDR="${runtime_master_addr:-$MASTER_ADDR}"
MASTER_PORT="${runtime_master_port:-$MASTER_PORT}"

export NODE_RANK MASTER_ADDR MASTER_PORT
OMP_NUM_THREADS="${OMP_NUM_THREADS:-${TORCH_NUM_THREADS:-1}}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-${TORCH_NUM_THREADS:-1}}"
export OMP_NUM_THREADS MKL_NUM_THREADS

exec torchrun \
    --nnodes="$NNODES" \
    --nproc-per-node="$NPROC_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    /app/train_gpt2.py
