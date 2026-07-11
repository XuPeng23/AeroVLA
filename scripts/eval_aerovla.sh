#!/bin/bash

# Configuration ------------------------------------
PROJECT_ROOT="."
MODEL_DIR="$PROJECT_ROOT/checkpoints/aero_vla"
EXP_NAME=$(basename "$MODEL_DIR")

# [SEEN] ----------------------------------
# TASK_ID="seen_valset/Carla_Town01"
# TASK_ID="seen_valset/Carla_Town04"
# TASK_ID="seen_valset/Carla_Town05"
# TASK_ID="seen_valset/Carla_Town06"
# TASK_ID="seen_valset/Carla_Town07"
# TASK_ID="seen_valset/Carla_Town10HD"
# TASK_ID="seen_valset/Carla_Town15"
# TASK_ID="seen_valset/ModernCityMap"
# TASK_ID="seen_valset/NewYorkCity"
TASK_ID="seen_valset/NYCEnvironmentMegapa"
# TASK_ID="seen_valset/TropicalIsland"

# [UNSEEN OBJECT] ----------------------------------
# TASK_ID="unseen_object_valset/Carla_Town04"
# TASK_ID="unseen_object_valset/Carla_Town05"
# TASK_ID="unseen_object_valset/Carla_Town06"
# TASK_ID="unseen_object_valset/Carla_Town07"
# TASK_ID="unseen_object_valset/Carla_Town15"
# TASK_ID="unseen_object_valset/NewYorkCity"
# TASK_ID="unseen_object_valset/NYCEnvironmentMegapa"
# TASK_ID="unseen_object_valset/TropicalIsland"

# [UNSEEN MAP] ----------------------------------
# TASK_ID="unseen_map_valset/ModularPark"


CATEGORY=$(echo "$TASK_ID" | cut -d'/' -f1)
MAP_NAME=$(echo "$TASK_ID" | cut -d'/' -f2)

TEST_JSON="$PROJECT_ROOT/data/uav_dataset/${CATEGORY}_splits/${MAP_NAME}.json"
SAVE_DIR="$PROJECT_ROOT/eval_results/${EXP_NAME}/${CATEGORY}/${MAP_NAME}"

GPU_ID=0
PORT=$((30000 + GPU_ID * 5000))

echo "========================================================"
echo "Category:      $CATEGORY"
echo "Map Name:      $MAP_NAME"
echo "GPU ID:        $GPU_ID"
echo "Sim Port:      $PORT"
echo "Config File:   $TEST_JSON"
echo "Save Dir:      $SAVE_DIR"
echo "========================================================"

# --------------------------------------------------
CUDA_VISIBLE_DEVICES=$GPU_ID python -u $PROJECT_ROOT/src/vlnce_src/eval_aerovla.py \
    --run_type eval \
    --name AerialVLA_Eval \
    --gpu_id $GPU_ID \
    --simulator_tool_port $PORT \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --maxWaypoints 200 \
    --dataset_path $PROJECT_ROOT/dataset_raw/ \
    --eval_save_path $SAVE_DIR \
    --model_path $MODEL_DIR \
    --eval_json_path $TEST_JSON \
    --map_spawn_area_json_path $PROJECT_ROOT/data/meta/map_spawnarea_info.json \
    --object_name_json_path $PROJECT_ROOT/data/meta/object_description.json
