#!/bin/bash

SEED=$(jq '.seed' /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/seed.json)

python  /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/gpt/gpt_infer.py \
  --savepath "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/dpo_339.json"  \
  --model_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/experiment_same_model_sigmoid/full_train_run/checkpoint-step339" \
  --filename "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/separation/test.json" \
  --inference_type "single" \


python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/final_assessment/text2reward.py \
    --data  "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/dpo_339.json"  \
    --comat_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/comat.yml" \
    --gsam_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/gsam.yml" \
    --reward_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/reward_config.yml" \
    --savepath "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/dpo_339_evaluated.json" \
    --img_folder "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/dpo_339_evaluated_imgs"
