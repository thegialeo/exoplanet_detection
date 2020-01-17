#!/bin/bash

CUDA_VISIBLE_DEVICES=${1} python train.py
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random'
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE'
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' 
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' 
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN'

CUDA_VISIBLE_DEVICES=${1} python train.py --no_preprocessing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random' --no_preprocessing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --no_preprocessing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' --no_preprocessing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' --no_preprocessing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --no_preprocessing

CUDA_VISIBLE_DEVICES=${1} python train.py --no_fourier_transform
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random' --no_fourier_transform
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --no_fourier_transform
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' --no_fourier_transform
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' --no_fourier_transform
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --no_fourier_transform

CUDA_VISIBLE_DEVICES=${1} python train.py --no_gaussian_smoothing 
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random' --no_gaussian_smoothing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --no_gaussian_smoothing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' --no_gaussian_smoothing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' --no_gaussian_smoothing
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --no_gaussian_smoothing

CUDA_VISIBLE_DEVICES=${1} python train.py --no_gaussian_smoothing --no_fourier_transform
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random' --no_gaussian_smoothing --no_fourier_transform
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --no_gaussian_smoothing --no_fourier_transform
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' --no_gaussian_smoothing --no_fourier_transform 
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' --no_gaussian_smoothing --no_fourier_transform
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --no_gaussian_smoothing --no_fourier_transform



