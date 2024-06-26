#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=sqllava-med
#SBATCH --error=/home/gs4288/guohao/sqLLaVA/RC_error/err_%j.txt
#SBATCH --output=/home/gs4288/guohao/sqLLaVA/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0-20:20:00
#SBATCH --gpus-per-node=a100:4
#SBATCH --partition tier3
#SBATCH --mem=64g
#SBATCH --account=crossmodal
#SBATCH --partition=tier3


source ~/conda/etc/profile.d/conda.sh
conda activate lmm

deepspeed train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/sqllava-7B-v1.5-1200k \
    --version v1_sq \
    --data_path /home/gs4288/guohao/data/medData/instruct/mysqllava_med_instruct_60k_inline_mention.json \
    --image_folder /home/gs4288/guohao/data/sqllava-med/llavaMed \
    --vision_tower ZachSun/sqllava-7B-vit-L-336-1200k \
    --pretrain_mm_mlp_adapter ./checkpoints/sqllava-7B-v1.5-1200k/mm_projector.bin \
    --mm_projector_type cluster \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/medical/sqllava-med-7b-lora-v3-im \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --vision_tower_lr 5e-5 \
    --vit_lora_enable \
    --lora_alpha_vit 64 \
    --lora_r_vit 32 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --data_aug False \
    --sq_r 0.5\
    
    