#!/bin/bash

# 定义变量
IMG_PATH="/home/huchengpeng/ILD_infer/imagesTr"
IMG_NAME_SUFFIX="_0000.nii.gz"
GT_PATH="/home/huchengpeng/ILD_infer/labelsTr"
GT_NAME_SUFFIX=".nii.gz"
OUTPUT_PATH="./data/ILD_infer"
NUM_WORKERS=4
MODALITY="CT"
ANATOMY="ILD"
WINDOW_LEVEL=40
WINDOW_WIDTH=400
SAVE_NII="--save_nii"

# Tiny checkpoint
TASK_NAME="MedSAM2-Tiny-ILD"
PRETRAIN_MODEL_PATH="./checkpoints/sam2_hiera_tiny.pt"
MODEL_CFG="sam2_hiera_t.yaml"
MEDSAM2CHECKPOINT="./work_dir/MedSAM2-Tiny-ILD-20240903-1601/medsam_model_best.pth"

NPZ_TEST_DIR="./data/ILD_infer/npz_test/CT_ILD"
PRED_SAVE_DIR="./inference/segs/medsam2"
EVALPATH="./inference/infer_eval/medsam2"


if [ ! -e ${OUTPUT_PATH}/.ILD.npz.done ]; then
    python ./inference/nii_to_npz_infer.py \
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


echo "Begin Inference!"

python ./inference/infer_medsam2_ILD_infer.py \
    -data_root $NPZ_TEST_DIR \
    -pred_save_dir $PRED_SAVE_DIR \
    -sam2_checkpoint $PRETRAIN_MODEL_PATH \
    -medsam2_checkpoint $MEDSAM2CHECKPOINT \
    -model_cfg $MODEL_CFG \
    -bbox_shift 5 \
    -num_workers 10 \
    --visualize ## Save segmentation, ground truth volume, and images in .nii.gz for visualization

python ./metrics/compute_metrics_ILD.py \
-s ./${PRED_SAVE_DIR} \
-g ./${NPZ_TEST_DIR} \
-csv_dir $EVALPATH