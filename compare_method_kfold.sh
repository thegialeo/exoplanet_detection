#!/bin/bash

CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --cross_validation --method 'KNN'
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --cross_validation --method 'SVC'
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --cross_validation --method 'RandomForest'

CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --cross_validation --method 'KNN' --no_fourier_transform --no_gaussian_smoothing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --cross_validation --method 'SVC' --no_fourier_transform --no_gaussian_smoothing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --cross_validation --method 'RandomForest' --no_fourier_transform --no_gaussian_smoothing

