#!/bin/bash
#CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --window_sliding 10 --num_epoch 10 --adaptive_learning 5 8 10 --learning_rate 0.001
#CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --window_sliding 100 --num_epoch 10 --adaptive_learning 5 8 10 --learning_rate 0.001
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --window_sliding 1000 --num_epoch 10 --adaptive_learning 5 8 10 --learning_rate 0.001 --batch_size 128



