#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/CMU-Visual-Learning-Recognition/Hw2-generative-model/generative-modeling-master/gan/%j_bash_run_explainer.out
pwd; hostname; date

CURRENT=`date +"%Y-%m-%d_%T_GAN"`
echo $CURRENT
slurm_output=/ocean/projects/asc170022p/shg121/PhD/CMU-Visual-Learning-Recognition/Hw2-generative-model/generative-modeling-master/gan/run_explainer_$CURRENT.out
echo "GAN"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate vlr_python_3_8

# # iter 1
#---------------------------------
# Train explainer

python python q1_3.py > $slurm_output

















