# sbatch -A bfhg-dtai-gh -partition= --mem=300g --nodes=1 --tasks=4 --tasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --time=48:00:00 ncsa_loop_layerwise.sh 
# sbatch --account=bfhg-delta-gpu --partition=gpuA40x4 --mem=128g --tasks=4 --nodes=1 --gpus-per-node=4 --tasks-per-node=4 --cpus-per-task=16 --time=12:00:00 ./run_scripts/teacher_finetune.sh
# sbatch --account=bfhg-delta-gpu --partition=gpuA100x4 --mem=240g --tasks=4 --nodes=1 --gpus-per-node=4 --tasks-per-node=4 --cpus-per-task=32 --time=2-00:00:00 ./run_scripts/parallel_distillation_train.sh
# sbatch --account=bfhg-delta-gpu --partition=gpuA40x4 --mem=240g --tasks=4 --nodes=1 --gpus-per-node=4 --tasks-per-node=4 --cpus-per-task=16 --time=12:00:00 ./run_scripts/canonicalizer_train.sh
sbatch --account=bfhg-delta-gpu --partition=gpuA100x4 --mem=240g --tasks=1 --nodes=1 --gpus-per-node=1 --tasks-per-node=1 --cpus-per-task=16 --time=2-00:00:00 ./scripts/vstibench/live1+.sh


# srun -A bfhg-dt-gh --partition=gpuA100x4 --gres=gpu:V100 --time=00:10:00 --tasks=4 --nodes=1 --ntasks-per-node=16 --pty ./run_scripts/teacher_finetune.sh
# srun --account=bfhg-delta-gpu --partition=gpuA100x4-interactive --mem=128g --tasks=1 --nodes=1 --gpus-per-node=1 --tasks-per-node=1 --cpus-per-task=8 --time=01:00:00 --pty bash
# srun --account=bfhg-delta-gpu --partition=gpuA40x4-interactive --mem=128g --tasks=1 --nodes=1 --gpus-per-node=1 --tasks-per-node=1 --cpus-per-task=8 --time=01:00:00 --pty bash
# srun --account=bfhg-delta-gpu --partition=cpu --mem=128g --tasks=1 --nodes=1 --gpus-per-node=1 --tasks-per-node=5 --cpus-per-task=16 --time=00:20:00 --pty bash

