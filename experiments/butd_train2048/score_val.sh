#!/bin/bash

MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF="_train2048"
SPLIT="val"

DATA_DIR="data/coco2014"
EXP_DIR="experiments/${MODEL_ABBR}${MODEL_SUFF}"

RES_FN="$EXP_DIR/outputs/${SPLIT}.beam_100.json"
TOP_FN="$EXP_DIR/outputs/${SPLIT}.beam_100.top_5.json"
TGT_FN="$EXP_DIR/outputs/${SPLIT}.targets.json"
OUT_DIR="$EXP_DIR/results"

#ANNS_DIR="$DATA_DIR"

args="""
  --results-fn $RES_FN \
	--top-results-fn $TOP_FN \
	--targets-fn $TGT_FN \
	--output-dir $OUT_DIR \
	--metrics coco \
	--annotations-dir $DATA_DIR \
	--annotations-split val2014 \
"""

#source activate /envs/syncap

export PYTHONWARNINGS="ignore"

python src/score.py $args

#conda deactivate