#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_single_scale
PY_ARGS=${@:1 }
COCO_PATH=/mnt/lihui.gu/coco

python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    --coco_path ${COCO_PATH} \
    ${PY_ARGS}
