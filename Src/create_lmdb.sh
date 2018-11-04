#!/usr/bin/en sh
LABEL=../data/Clothes
LMDB_PATH=../data/Clothes
TOOL_PATH=../build/tools
TRAIN_DATA_ROOT=../data/Clothes/train_imgs/
VAL_DATA_ROOT=../data/Clothes/test_imgs/

rm -rf $LMDB_PATH/img_train_lmdb
rm -rf $LMDB_PATH/img_test_lmdb

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=150
  RESIZE_WIDTH=150
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Create train lmdb.."
$TOOL_PATH/convert_imageset \
  --shuffle=true \
	--resize_height=$RESIZE_HEIGHT \
	--resize_width=$RESIZE_WIDTH \
	$TRAIN_DATA_ROOT \
 	$LABEL/train_unorder.txt \
 	$LMDB_PATH/img_train_lmdb

echo "Create test lmdb.."
$TOOL_PATH/convert_imageset \
  --shuffle=true \
  --resize_height=$RESIZE_HEIGHT \
  --resize_width=$RESIZE_WIDTH \
  $VAL_DATA_ROOT \
  $LABEL/test_unorder.txt \
  $LMDB_PATH/img_test_lmdb