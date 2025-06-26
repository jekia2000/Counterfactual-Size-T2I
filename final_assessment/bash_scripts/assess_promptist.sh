python /data/leuven/370/vsc37045/thesis/final_assessment/promptist_infer.py \
    --filename "/scratch/leuven/370/vsc37045/resource/gpt/separation/test.json" \
    --savepath"/scratch/leuven/370/vsc37045/resource/assessments/promptist.json"


python /data/leuven/370/vsc37045/thesis/final_assessment/generate_images.py \
    --data "/scratch/leuven/370/vsc37045/resource/assessments/promptist.json" \
    --comat_config "/data/leuven/370/vsc37045/thesis/evaluator/comat.yml" \
    --gsam_config "/data/leuven/370/vsc37045/thesis/evaluator/gsam.yml" \
    --reward_config "/data/leuven/370/vsc37045/thesis/evaluator/reward_config.yml" \
    --savepath "/scratch/leuven/370/vsc37045/resource/assessments/promptist_evaluated.json" \
    --img_folder "/scratch/leuven/370/vsc37045/resource/assessments/promptist_evaluated_imgs"
   
