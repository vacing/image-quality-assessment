# !/usr/bin/bash
# encoding=utf-8

set -x
BASE_MODEL_NAME='MobileNet'
# WEIGHTS_FILE='../models/MobileNet/weights_mobilenet_technical_0.11.hdf5'
WEIGHTS_FILE='../models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5'

# 不能直接使用已有的模型的参数，因为是基于现有模型做的更改
# BASE_MODEL_NAME='Xception'
# WEIGHTS_FILE='/home/vacingfang/.keras/models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

# IMAGE_SOURCE='../src/tests/test_images/42039.jpg'
# IMAGE_SOURCE='../src/img_cmp/fashion_02/'
# IMAGE_SOURCE='/data/home/vacingfang/mydata/video_download_gametype/csgo/csgo_4'
IMAGE_SOURCE='/data/home/vacingfang/mydata/temp/66_CSGO'
IMAGE_DST='/data/home/vacingfang/mydata/temp/66_CSGO_dst'
# IMAGE_SOURCE='../src/img_cmp/auto_05_picture/'

export CUDA_VISIBLE_DEVICES=4,5
python -m evaluater.predict \
--base-model-name $BASE_MODEL_NAME \
--weights-file $WEIGHTS_FILE \
--image-source $IMAGE_SOURCE \
--image-dst $IMAGE_DST
