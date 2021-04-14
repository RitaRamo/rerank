IMAGE_NAME="image_features.hdf5"
MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF="_train2048"

DATA_DIR="data/coco2014"
IMGS_DIR="$DATA_DIR/images"
EXP_DIR="experiments/${MODEL_ABBR}${MODEL_SUFF}" #VER ESTE!!!
CAPS_DIR="$DATA_DIR/datasets"
LOG_DIR="myfirstlog"
CKPT="experiments/${MODEL_ABBR}${MODEL_SUFF}/checkpoints/checkpoint.best.pth.tar"

args="""
	--image-features-filename $IMGS_DIR/$IMAGE_NAME \
  --dataset-splits-dir $CAPS_DIR \
  --checkpoint $CKPT \
	--logging-dir $LOG_DIR \
	--output-path $EXP_DIR \
	--split val \
	--max-caption-len 20 \
	--beam-size 5 \
	--eval-beam-size 5 \
"""

#source activate /envs/syncap

export PYTHONWARNINGS="ignore"

time python3 src/eval.py $args

#conda deactivate


