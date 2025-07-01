#!/bin/bash
python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/dpo/dpo_train_plain.py \
 --conf_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/dpo/conf.yml" \
 --sft_model_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/sft_large/best_model/" \
 --train_dataset_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/new_dataset/train_same_model.json/" \
 --val_dataset_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/new_dataset/val_same_model.json/" \
