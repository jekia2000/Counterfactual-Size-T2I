#!/bin/bash
python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/gpt/gpt_train.py \
    --train_dataset_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/dataset/train.json" \
    --val_dataset_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/dataset/val.json" \
    --model_save_dir "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/sft_right"
