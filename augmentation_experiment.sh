#!/bin/bash

CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --extra_augmentation 0.1
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --extra_augmentation 0.3
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --extra_augmentation 0.8

