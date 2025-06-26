#!/bin/bash
python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/dpo/dpo_train_plain.py \
 --conf_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/dpo/conf.yml" \
 --sft_model_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/sft_right/best_model/" \
 --train_dataset_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/dataset/train_same_model.json/" \
 --val_dataset_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/dataset/val_same_model.json/" \
