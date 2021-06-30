#!/bin/bash

CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --extra_augmentation 0.1 --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --extra_augmentation 0.3 --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --extra_augmentation 0.8 --cross_validation


