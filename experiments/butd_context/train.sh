#Retrieval 1 caption avg learned emb
IMAGE_NAME="image_features.hdf5"
MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF="_context"

DATA_DIR="data/coco2014"
IMGS_DIR="$DATA_DIR/images"
CAPS_DIR="$DATA_DIR/datasets"
CKPT_DIR="experiments/${MODEL_ABBR}${MODEL_SUFF}/checkpoints/"
LOG_DIR="myfirstlog"

mkdir -p $CKPT_DIR $LOG_DIR

train_args="""
	--image-features-filename $IMGS_DIR/$IMAGE_NAME \
  --dataset-splits-dir $CAPS_DIR \
  --checkpoints-dir $CKPT_DIR \
	--logging-dir $LOG_DIR \
	--objective GENERATION \
  --batch-size 5 \
	--max-epochs 120 \
	--epochs-early-stopping 5 \
	--max-caption-len 20 \
	--print-freq 10 \
	--debug \
"""

model_args="""
butd_context
--embeddings-dim 300 \
--attention-dim 1024 \
--attention-lstm-dim 1024 \
--language-lstm-dim 1024 \
--dropout 0.5
"""

#source activate /envs/syncap

export PYTHONWARNINGS="ignore"

time python3 src/train.py $train_args $model_args