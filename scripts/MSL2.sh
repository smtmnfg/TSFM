#!/bin/sh
#BATCH --job-name= anomaly_detection
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --gres=gpu:1 ## Run on 1 GPU
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p gpu-v100-16gb
##SBATCH -p v100-64gb-hiprio
##SBATCH -p AI_Center_L40S

##Load your modules and run code here

hostname

date



module load cuda/12.1

module load python3/anaconda/2023.9


python -u /work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/custom_compact_models/NeurIPS2023-One-Fits-All/Anomaly_Detection/run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /work/jayakodc/mcnair_data/smart-manufacturing/smart-manufacturing/custom_compact_models/NeurIPS2023-One-Fits-All/Anomaly_Detection/datasets/MSL \
  --model_id MSL \
  --model GPT4TS \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --gpt_layer 6 \
  --d_model 768 \
  --d_ff 8 \
  --patch_size 1 \
  --stride 1 \
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 2 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10
