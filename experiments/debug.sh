DATA_SPLIT="coco_heldout_1"
IMAGE_NAME="image_features.hdf5"
MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF=""

DATA_DIR="data/coco2014"
IMGS_DIR="$DATA_DIR/coco2014/images"
CAPS_DIR="$DATA_DIR/datasets"
CKPT_DIR="/checkpoints/${MODEL_NAME}${MODEL_SUFF}"
LOG_DIR="myfirstlog"

mkdir -p $CKPT_DIR $LOG_DIR

train_args="""
	--image-features-filename $IMGS_DIR/$IMAGE_NAME \
  --dataset-splits-dir $CAPS_DIR \
  --checkpoints-dir $CKPT_DIR \
	--logging-dir $LOG_DIR \
	--objective GENERATION \
  --batch-size 2 \
	--max-epochs 120 \
	--epochs-early-stopping 5 \
	--max-caption-len 20 \
	--print-freq 10 \
"""

model_args="""
butd
"""

#source activate /envs/syncap

export PYTHONWARNINGS="ignore"

time python3 src/train.py $train_args $model_args
