python /data/leuven/370/vsc37045/thesis/final_assessment/input_becomes_output.py \
    --data "/scratch/leuven/370/vsc37045/resource/gpt/separation/test.json" \
    --savepath "/scratch/leuven/370/vsc37045/resource/assessments/template_test.json"


python /data/leuven/370/vsc37045/thesis/final_assessment/generate_images.py \
    --data  "/scratch/leuven/370/vsc37045/resource/assessments/template_test.json"  \
    --comat_config "/data/leuven/370/vsc37045/thesis/evaluator/comat.yml" \
    --gsam_config "/data/leuven/370/vsc37045/thesis/evaluator/gsam.yml" \
    --reward_config "/data/leuven/370/vsc37045/thesis/evaluator/reward_config.yml" \
    --savepath "/scratch/leuven/370/vsc37045/resource/assessments/template_test_evaluated.json" \
    --img_folder "/scratch/leuven/370/vsc37045/resource/assessments/template_test_evaluated_imgs"