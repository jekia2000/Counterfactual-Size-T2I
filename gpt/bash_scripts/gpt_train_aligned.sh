#!/bin/bash
python /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/gpt/gpt_train.py \
    --train_dataset_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/gpt/dataset/train_aligned.json" \
    --val_dataset_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/gpt/dataset/val_aligned.json" \
    --model_save_dir "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/gpt/new_sft_aligned"
