#!/bin/sh

source ~/.bashrc
conda activate videollm


torchrun --nproc_per_node=1 --standalone train.py --deepspeed configs/deepspeed/zero1.json \
    --live_version live1+ \
    --train_datasets vstibench_goalstep_livechat_trainval \
    --eval_datasets  vstibench_goalstep_livechat_val \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --eval_strategy no \
    --prediction_loss_only False \
    --save_strategy epoch \
    --save_steps 1 \
    --learning_rate 0.0001 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/vstibench/live1+ \
