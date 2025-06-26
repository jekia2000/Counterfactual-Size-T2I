#!/bin/bash

SEED=$(jq '.seed' /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/seed.json)
python /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/final_assessment/input_becomes_output.py \
    --data "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/gpt/separation/test.json" \
    --savepath "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/template_test.json"


python /leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/final_assessment/text2reward.py \
    --data  "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/template_test.json"  \
     --comat_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/comat.yml" \
    --gsam_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/gsam.yml" \
    --reward_config "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/thesis/evaluator/reward_config.yml" \
    --savepath "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/template_test_evaluated.json" \
    --img_folder "/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/assessments/${SEED}/template_test_imgs"
