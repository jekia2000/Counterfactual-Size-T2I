#!/bin/bash

SEED=$(jq '.seed' /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/seed.json)
python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/gpt/gpt_infer.py \
  --savepath "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/big_reranker_aligned.json"  \
  --model_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/sft_right_aligned/best_model" \
  --reranker_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/experiment_gpt2_large/full_train_run/checkpoint-step113" \
  --filename "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/separation/test.json" \
  --inference_type "single" \
  --num_return 14 \
  --top_p 1.0 \
  --temperature 0.6 \


python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/final_assessment/text2reward.py \
    --data  "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/big_reranker_aligned.json"  \
    --comat_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/comat.yml" \
    --gsam_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/gsam.yml" \
    --reward_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/reward_config.yml" \
    --savepath "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/big_reranker_aligned_evaluated.json" \
    --img_folder "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/assessments/${SEED}/big_reranker_aligned_imgs"
