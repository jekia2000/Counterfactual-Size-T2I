python /data/leuven/370/vsc37045/thesis/gpt/gpt_infer.py \
  --savepath "/scratch/leuven/370/vsc37045/resource/dpo/dataset/extended_prompts.json" \
  --model_path "/scratch/leuven/370/vsc37045/resource/gpt/sft/best_model" \
  --filename "/scratch/leuven/370/vsc37045/resource/gpt/dataset/train+val.json" \
  --num_return 14 \
  --top_p 1.0 \
  --temperature 0.6 \
  --keep_prev \


python /data/leuven/370/vsc37045/thesis/final_assessment/text2reward.py \
  --data "/scratch/leuven/370/vsc37045/resource/dpo/dataset/extended_prompts.json" \
  --comat_config "/data/leuven/370/vsc37045/thesis/evaluator/comat.yml" \
  --gsam_config "/data/leuven/370/vsc37045/thesis/evaluator/gsam.yml" \
  --reward_config "/data/leuven/370/vsc37045/thesis/evaluator/reward_config.yml" \
  --savepath "/scratch/leuven/370/vsc37045/resource/dpo/dataset/evaluated_prompts.json"