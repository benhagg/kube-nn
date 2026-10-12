# we need to run the file with torchrun so we can run multiple processes per container. It simulates a distributed training environment where we have multiple GPUs per node.
#!/bin/sh
set -eu

# Keep values supplied by Kubernetes/Docker available as runtime overrides.
runtime_node_rank="${CONTAINER_NODE_RANK:-}"
runtime_master_addr="${TRAIN_MASTER_ADDR:-}"
runtime_master_port="${TRAIN_MASTER_PORT:-}"

set -a
. /app/.env
set +a

NODE_RANK="${runtime_node_rank:-$NODE_RANK}"
MASTER_ADDR="${runtime_master_addr:-$MASTER_ADDR}"
MASTER_PORT="${runtime_master_port:-$MASTER_PORT}"

export NODE_RANK MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS MKL_NUM_THREADS

exec torchrun \
    --nnodes="$NNODES" \
    --nproc-per-node="$NPROC_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    /app/train_gpt2.py
