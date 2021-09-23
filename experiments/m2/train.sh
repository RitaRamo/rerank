IMAGE_NAME="image_features.hdf5"
MODEL_NAME="m2_transformer"
MODEL_ABBR="m2_transformer"
MODEL_SUFF=""

DATA_DIR="data/coco2014"
CAPS_DIR="$DATA_DIR/datasets"
CKPT_DIR="experiments/${MODEL_ABBR}${MODEL_SUFF}/checkpoints/"
LOG_DIR="myfirstlog"

IMGS_DIR="$DATA_DIR/images/coco_detections.hdf5"
ANNS_DIR="$DATA_DIR/annotations"

mkdir -p $CKPT_DIR $LOG_DIR

train_args="""
	--exp_name ${MODEL_NAME}${MODEL_SUFF} \
  --batch_size 2 \
  --m 40 \
  --head 8 \
  --warmup 10000 \
	--max_len 20 \
  --features_path $IMGS_DIR \
  --annotation_folder $ANNS_DIR \
  --id_folder $CAPS_DIR \
  --checkpoint_path $CKPT_DIR \
  --logs_folder $LOG_DIR \
  --resume_last
"""


time python3 src/meshed-memory-transformer/train.py $train_args

