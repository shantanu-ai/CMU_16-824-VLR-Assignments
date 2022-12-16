#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/CMU-Visual-Learning-Recognition/Hw3-vqa-main/%j_bash_run_BB.out
pwd
hostname
date
CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/CMU-Visual-Learning-Recognition/Hw3-vqa-main/run_BB_$CURRENT.out
echo "Vlr-Hw3"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate vlr

# python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/BB/VIT/TransFG-master/train_awa2.py > $slurm_output

#python /ocean/projects/asc170022p/shg121/PhD/CMU-Visual-Learning-Recognition/Hw3-vqa-main/main.py > $slurm_output

python /ocean/projects/asc170022p/shg121/PhD/CMU-Visual-Learning-Recognition/Hw3-vqa-main/main.py --model deep_mlp >$slurm_output
