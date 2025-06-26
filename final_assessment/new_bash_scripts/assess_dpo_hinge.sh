#!/bin/bash

SEED=$(jq '.seed' /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/seed.json)

python /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/gpt/gpt_infer.py \
  --savepath "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/dpo_hinge.json"  \
  --model_path "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/dpo/experiment_same_model_hinge/full_train_run/checkpoint-step112" \
  --filename "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/gpt/separation/test.json" \
  --inference_type "single" \


python /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/final_assessment/text2reward.py \
    --data  "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/dpo_hinge.json"  \
    --comat_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/comat.yml" \
    --gsam_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/gsam.yml" \
    --reward_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/reward_config.yml" \
    --savepath "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/dpo_hinge_evaluated.json" \
    --img_folder "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/dpo_hinge_imgs"
