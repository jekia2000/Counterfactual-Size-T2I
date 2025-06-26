#!/bin/bash
python /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/dpo/dpo_train_plain.py \
 --conf_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/dpo/conf.yml" \
 --sft_model_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/gpt/new_sft/best_model/" \
 --train_dataset_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/dpo/dataset/train.json/" \
 --val_dataset_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/dpo/dataset/val.json/" \
