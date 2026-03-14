# python -m data.preprocess.encode --num_gpus 8 --vision_pretrained google/siglip-large-patch16-384 --video_dir ??

export MASTER_ADDR=localhost
export MASTER_PORT=12345
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_TORCH_DEVICE_BACKEND=gloo \
python -m data.preprocess.encode \
    --num_gpus 1 \
    --vision_pretrained google/siglip-large-patch16-384 \
    --video_dir /u/hyin2/videollm-online/datasets/vlm3r_videos/scannet/videos_2fps_max384
    # --video_dir vsi-bench/scannet_2fps_max384