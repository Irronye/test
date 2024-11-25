#!/bin/bash
#SBATCH --job-name=heatmap_visualization
#SBATCH --output=heatmap_output.log
#SBATCH --error=heatmap_error.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

module purge

singularity exec --nv \
    --overlay /scratch/mae9855/ContrastiveCrop/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "python3.10 visualize_heatmap.py"
   
#--checkpoint ./checkpoints/small/cifar10/moco_ccrop/last.pth --data ./data/cifar-10-batches-py/ --image ./sample_image.png"
