#!/bin/bash

SEED=$(jq '.seed' /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/seed.json)

python /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/gpt/gpt_infer.py \
  --savepath "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/dpo_dd.json"  \
  --model_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/dpo/experiment_different_distribution/full_train_run/checkpoint-step119" \
  --filename "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/gpt/separation/test.json" \
  --inference_type "single" \


python /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/final_assessment/text2reward.py \
    --data  "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/dpo_dd.json"  \
    --comat_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/comat.yml" \
    --gsam_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/gsam.yml" \
    --reward_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/reward_config.yml" \
    --savepath "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/dpo_dd_evaluated.json" \
    --img_folder "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/dpo_dd_imgs"
