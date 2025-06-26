
python /data/leuven/370/vsc37045/thesis/gpt/gpt_infer.py \
  --savepath "/scratch/leuven/370/vsc37045/resource/assessments/gpt2_dpo_full_1302.json"  \
  --model_path "/scratch/leuven/370/vsc37045/resource/dpo/experiment3/full_train_run/checkpoint-step1302" \
  --filename "/scratch/leuven/370/vsc37045/resource/gpt/separation/test.json" \
  --inference_type "single" \




python /data/leuven/370/vsc37045/thesis/final_assessment/generate_images.py \
    --data  "/scratch/leuven/370/vsc37045/resource/assessments/gpt2_dpo_full_1302.json" \
    --comat_config "/data/leuven/370/vsc37045/thesis/evaluator/comat.yml" \
    --gsam_config "/data/leuven/370/vsc37045/thesis/evaluator/gsam.yml" \
    --reward_config "/data/leuven/370/vsc37045/thesis/evaluator/reward_config.yml" \
    --savepath "/scratch/leuven/370/vsc37045/resource/assessments/gpt2_dpo_full_1302_evaluated.json" \
    --img_folder "/scratch/leuven/370/vsc37045/resource/assessments/gpt2_dpo_full_1302_evaluated_imgs"
