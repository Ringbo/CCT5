#!/bin/bash

CURRENT_DIR=`pwd`
NCCL_DEBUG=INFO
GPU_ID=0

usage() {
  echo "Usage: ${0} [-g|--gpuid] " 1>&2
  exit 1 
}

while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    -g|--gpuid)
      GPU_ID=${2}
      shift 2
      ;;
    *)
      usage
      shift
      ;;
  esac
done

function pretrain() {
    SCRIPT_PATH="src/pre_training/pretrain.py"
    python $SCRIPT_PATH \
        --task pretraining \
        --sub_task none \
        --model_type codet5_CC \
        --data_num -1 \
        --warmup_steps 500 \
        --learning_rate 3e-4 \
        --num_train_epochs 30 \
        --model_name_or_path Salesforce/codet5-base \
        --tokenizer_name Salesforce/codet5-base \
        --data_dir ${CURRENT_DIR}/Dataset/pre-training \
        --output_dir ${CURRENT_DIR}/outputs/models/pre-training \
        --always_save_model \
        --train_batch_size 32 \
        --gradient_accumulation_steps 4 \
        --eval_batch_size 4 \
        --max_source_length 512 \
        --max_target_length 128 \
        --gpu_id ${GPU_ID} \
        --mask_rate 0.15 \
        --save_steps 6000 \
        --log_steps 5 \
        --train_steps 800000 \
        --treesitter_path ${CURRENT_DIR}/myParser/my-languages.so
}


pretrain;