#!/bin/bash
python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/dpo/dpo_train.py \
 --comat_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/comat.yml" \
 --gsam_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/gsam.yml" \
 --reward_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/reward_config.yml" \
 --sft_model_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/sft_right/best_model/" \
 --train_dataset_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/new_dataset/train_same_model.json/" \
 --val_dataset_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/new_dataset/val_same_model.json/" \
 --dpo_save_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/experiment_same_model_sigmoid/"
