#!/bin/bash
python /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/dpo/dpo_train.py \
 --comat_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/comat.yml" \
 --gsam_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/gsam.yml" \
 --reward_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/reward_config.yml" \
 --sft_model_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/gpt/new_sft/best_model/" \
 --train_dataset_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/dpo/dataset/train_same_model.json/" \
 --val_dataset_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/dpo/dataset/val_same_model.json/" \
 --dpo_save_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/dpo/experiment_same_model_hinge/" \
 --loss_type "hinge"
