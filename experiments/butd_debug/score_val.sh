#!/bin/bash

MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF="_debug"
SPLIT="val"

DATA_DIR="data/coco2014"
EXP_DIR="$experiments"

RES_FN="$EXP_DIR/_outputs/${MODEL_ABBR}${MODEL_SUFF}${SPLIT}.beam_100.json"
TOP_FN="$EXP_DIR/_outputs/${MODEL_ABBR}${MODEL_SUFF}${SPLIT}.beam_100.top_5.json"
TGT_FN="$EXP_DIR/_outputs/${MODEL_ABBR}${MODEL_SUFF}${SPLIT}.targets.json"
OUT_DIR="$EXP_DIR/results"

ANNS_DIR="$DATA_DIR/annotations"

args="""
  --results-fn $RES_FN \
	--top-results-fn $TOP_FN \
	--targets-fn $TGT_FN \
	--output-dir $OUT_DIR \
	--metrics coco \
	--annotations-dir $ANN_DIR \
	--annotations-split train2014 \
"""

#source activate /envs/syncap

export PYTHONWARNINGS="ignore"

python3 src/score.py $args

#conda deactivate