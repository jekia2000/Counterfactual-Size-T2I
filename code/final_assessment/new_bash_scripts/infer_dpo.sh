#!/bin/bash

python  /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/gpt/gpt_infer.py \
  --savepath "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/new_dataset/extended_prompts.json" \
  --model_path "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/sft_right/best_model" \
  --filename "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/gpt/dataset/train+val.json" \
  --num_return 14 \
  --top_p 1.0 \
  --temperature 0.6 \
  --keep_prev \


python /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/final_assessment/text2reward.py \
  --data "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/new_dataset/extended_prompts.json" \
  --comat_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/comat.yml" \
  --gsam_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/gsam.yml" \
  --reward_config "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/evaluator/reward_config.yml" \
  --savepath "/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/resource/dpo/new_dataset/evaluated_prompts.json"
