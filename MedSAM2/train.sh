#!/bin/bash

# 定义变量
IMG_PATH="/jizhicfs/chengpenghu/meddataset/ILD/imagesTr"
IMG_NAME_SUFFIX="_0000.nii.gz"
GT_PATH="/jizhicfs/chengpenghu/meddataset/ILD/labelsTr"
GT_NAME_SUFFIX=".nii.gz"
OUTPUT_PATH="/jizhicfs/chengpenghu/meddataset/data"
NUM_WORKERS=4
MODALITY="CT"
ANATOMY="ILD"
WINDOW_LEVEL=40
WINDOW_WIDTH=400
SAVE_NII="--save_nii"

NPZ_DIR="/jizhicfs/chengpenghu/meddataset/data/npz_train/CT_ILD"
NPY_DIR="/jizhicfs/chengpenghu/meddataset/data/npy"

TASK_NAME="MedSAM2-Tiny-ILD"
WORK_DIR="./work_dir"
BATCH_SIZE=16
PRETRAIN_MODEL_PATH="./checkpoints/sam2_hiera_tiny.pt"
MODEL_CFG="sam2_hiera_t.yaml"

# 继续训练的模型路径
RESUME=false  # 将此值设置为true以继续训练
RESUME_PATH="./work_dir/MedSAM2-Tiny-Flare22-2024-08-22-07-00/medsam2_model_latest.pth"


if [ ! -e ${OUTPUT_PATH}/.ILD.npz.done ]; then
    python nii_to_npz.py \
        -img_path $IMG_PATH \
        -img_name_suffix $IMG_NAME_SUFFIX \
        -gt_path $GT_PATH \
        -gt_name_suffix $GT_NAME_SUFFIX \
        -output_path $OUTPUT_PATH \
        -num_workers $NUM_WORKERS \
        -modality $MODALITY \
        -anatomy $ANATOMY \
        -window_level $WINDOW_LEVEL \
        -window_width $WINDOW_WIDTH \
        $SAVE_NII
    touch  ${OUTPUT_PATH}/.ILD.npz.done
fi

echo "Nii to Npz Completed!"

if [ ! -e ${OUTPUT_PATH}/.ILD.npy.done ]; then
    python npz_to_npy.py \
        -npz_dir $NPZ_DIR \
        -npy_dir $NPY_DIR \
        -num_workers $NUM_WORKERS
    touch  ${OUTPUT_PATH}/.ILD.npy.done
fi

echo "Npz to Npy Completed!"

echo "Begin Training!"

if [ "$RESUME" = true ]; then
    python finetune_sam2_img.py \
        -i $NPY_DIR \
        -task_name $TASK_NAME \
        -work_dir $WORK_DIR \
        -batch_size $BATCH_SIZE \
        -pretrain_model_path $PRETRAIN_MODEL_PATH \
        -model_cfg $MODEL_CFG \
        -resume $RESUME_PATH
else
    python finetune_sam2_img.py \
        -i $NPY_DIR \
        -task_name $TASK_NAME \
        -work_dir $WORK_DIR \
        -batch_size $BATCH_SIZE \
        -pretrain_model_path $PRETRAIN_MODEL_PATH \
        -model_cfg $MODEL_CFG
fi