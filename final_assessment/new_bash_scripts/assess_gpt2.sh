#!/bin/bash

SEED=$(jq '.seed' /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/seed.json)
python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/gpt/gpt_infer.py \
  --savepath "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/gpt2.json"  \
  --model_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/sft_right/best_model" \
  --filename "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/separation/test.json" \
  --inference_type "single" \


python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/final_assessment/text2reward.py \
    --data  "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/gpt2.json"  \
    --comat_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/evaluator/comat.yml" \
    --gsam_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/evaluator/gsam.yml" \
    --reward_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/evaluator/reward_config.yml" \
    --savepath "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/gpt2_evaluated.json" \
    --img_folder "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/gpt2_imgs"
