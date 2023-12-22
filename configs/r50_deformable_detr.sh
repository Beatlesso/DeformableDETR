#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr
PY_ARGS=${@:1}
COCO_PATH=/mnt/lihui.gu/coco

python -u main.py \
    --output_dir ${EXP_DIR} \
    --coco_path ${COCO_PATH} \
    ${PY_ARGS}
