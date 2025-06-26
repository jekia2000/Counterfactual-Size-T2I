#!/bin/bash
python temp_store.py \
--config "/data/leuven/370/vsc37045/thesis/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
--grounded_checkpoint "groundingdino_swint_ogc.pth" \
--sam_version "vit_l" \
--sam_checkpoint "sam_vit_l_0b3195.pth" \
--embeddings "$VSC_SCRATCH/resource" \
--img_path "$VSC_SCRATCH/resource/images" \
--device "cuda" \
--box_threshold 0.35 \
--text_threshold 0.25 \
--iou_threshold 0.8 \
