#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=dpo
#SBATCH --error=/home/gs4288/guohao/sqLLaVA/RC_error/err_%j.txt
#SBATCH --output=/home/gs4288/guohao/sqLLaVA/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0-5:20:00
#SBATCH --gpus-per-node=a100:4
#SBATCH --partition tier3
#SBATCH --mem=64g
#SBATCH --account=crossmodal
#SBATCH --partition=tier3


source ~/conda/etc/profile.d/conda.sh
conda activate lmm

deepspeed train_mem_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/medical/sqllava-med-7b-70k \
    --version v1 \
    --data_path /home/gs4288/guohao/data/medData/instruct/sqllava_med_70k_dpo_10k.json \
    --image_folder /home/gs4288/guohao/data/sqllava-med/llavaMed \
    --vision_tower ./checkpoints/medical/sqllava-med-7b-70k-vit \
    --pretrain_mm_mlp_adapter ./checkpoints/medical/sqllava-med-7b-lora-70k-9epo/non_lora_trainables.bin \
    --mm_projector_type cluster \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/sqllava-med-lora-7b-70k-dpo-v3-1po\
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 5 \
    --learning_rate 5e-6 \
    --vision_tower_lr 2e-5 \
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
    --data_aug False