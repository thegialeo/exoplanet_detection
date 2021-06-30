#!/bin/bash
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --window_sliding 100 --num_epoch 10 --adaptive_learning 5 8 10 --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --window_sliding 500 --num_epoch 10 --adaptive_learning 5 8 10 --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --window_sliding 1000 --num_epoch 10 --adaptive_learning 5 8 10 --cross_validation



