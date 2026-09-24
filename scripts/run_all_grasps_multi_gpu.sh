#!/bin/bash

set -eu

# Run grasp planning for all robot/grasp_type combinations with example_grasp/multi_gpu.py.
# Usage:
#   ./scripts/run_all_grasps_multi_gpu.sh --hand <shadow|leap_sp|leap> --exp-name NAME --parallel-env 100 --gpus 0 1 2 3 [-k] [--start 0] [--end 10]
# Examples:
#   ./scripts/run_all_grasps_multi_gpu.sh --hand shadow --exp-name minitest --parallel-env 10 --gpus 0 1 2 3
#   ./scripts/run_all_grasps_multi_gpu.sh --hand leap --exp-name minitest --parallel-env 10 --gpus 0 1 2 3 -k --start 0 --end 10

usage() {
    echo "Usage: ./scripts/run_all_grasps_multi_gpu.sh --hand <shadow|leap_sp|leap> --exp-name NAME --parallel-env N --gpus GPU0 [GPU1 ...] [-k] [--start START] [--end END]"
    echo "-k disables skipping existing files."
}

EXP_NAME="default"
NUM_PARALLEL_ENV=10
START=""
END=""
SKIP=true
GPU_IDS=()
HAND_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --hand)
            HAND_NAME="$2"
            shift 2
            ;;
        --parallel-env)
            NUM_PARALLEL_ENV="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"
            shift 2
            ;;
        --gpus)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                GPU_IDS+=("$1")
                shift
            done
            ;;
        -k)
            SKIP=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$HAND_NAME" ]; then
    echo "--hand must be provided."
    usage
    exit 1
fi

case "$HAND_NAME" in
    shadow|leap_sp|leap)
        ;;
    *)
        echo "Unsupported hand name: $HAND_NAME"
        usage
        exit 1
        ;;
esac

if [ ${#GPU_IDS[@]} -eq 0 ]; then
    echo "At least one GPU id must be provided with --gpus."
    exit 1
fi

case "$HAND_NAME" in
    shadow)
        declare -a CONFIGS=(
            "sim_shadow/tabletop_two"
            "sim_shadow/tabletop_three"
            "sim_shadow/tabletop_full"
            "sim_dual_dummy_arm_shadow/tabletop_three"
            "sim_dual_dummy_arm_shadow/tabletop_full"
        )
        ;;
    leap)
        declare -a CONFIGS=(
            "sim_leap/tabletop_two"
            "sim_leap/tabletop_three"
            "sim_leap/tabletop_full"
            "sim_dual_dummy_arm_leap/tabletop_three"
            "sim_dual_dummy_arm_leap/tabletop_full"
        )
        ;;
    leap_sp)
        declare -a CONFIGS=(
            "sim_leap_sp/tabletop_two"
            "sim_leap_sp/tabletop_three"
            "sim_leap_sp/tabletop_full"
            "sim_dual_dummy_arm_leap_sp/tabletop_three"
            "sim_dual_dummy_arm_leap_sp/tabletop_full"
        )
        ;;
esac

echo "Starting multi-GPU grasp planning experiments..."
echo "Hand: $HAND_NAME"
echo "Experiment Name: $EXP_NAME"
echo "GPUs: ${GPU_IDS[*]}, Parallel Envs: $NUM_PARALLEL_ENV"
echo "Skip Existing: $SKIP"
if [ -n "$START" ] || [ -n "$END" ]; then
    echo "World Range Override: start=${START:-<config>}, end=${END:-<config>}"
fi
echo "================================"

for config in "${CONFIGS[@]}"; do
    config_name=$(echo "$config" | tr '/' '_')

    echo ""
    echo "Running: $EXP_NAME"
    echo "Config: $config"
    echo "--------------------------------"

    cmd=(
        python example_grasp/multi_gpu.py
        -c "$config.yml"
        -w "$NUM_PARALLEL_ENV"
        -t grasp
        -g "${GPU_IDS[@]}"
        --exp_name "$EXP_NAME"
    )

    if [ "$SKIP" = false ]; then
        cmd+=(-k)
    fi

    if [ -n "$START" ]; then
        cmd+=(--start "$START")
    fi

    if [ -n "$END" ]; then
        cmd+=(--end "$END")
    fi

    if ! "${cmd[@]}"; then
        echo "Failed: ${config_name}/${EXP_NAME}"
        exit 1
    fi
    echo "Completed: ${config_name}/${EXP_NAME}"
done

echo ""
echo "================================"
echo "All experiments completed!"
