torchrun --nproc_per_node=1 --standalone evaluate.py \
    --live_version live1+ \
    --eval_datasets vstibench_goalstep_livechat_trainval \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --prediction_loss_only False \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/vstibench/live1+/ \
    --resume_from_checkpoint /u/hyin2/videollm-online/outputs/vstibench/live1+/checkpoint-1050
    # --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus
    # --resume_from_checkpoint outputs/coin_benchmarks/live1+/
