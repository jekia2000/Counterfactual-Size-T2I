#!/bin/bash

SEED=$(jq '.seed' /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/seed.json)
python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/final_assessment/text2reward.py \
    --data "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/dataset/test.json" \
    --comat_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/comat.yml" \
    --gsam_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/gsam.yml" \
    --reward_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/reward_config.yml" \
    --savepath "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/chatgpt-4o_evaluated.json" \
    --img_folder "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/chatgpt-4o_evaluated_imgs"
   
