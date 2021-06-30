#!/bin/bash

CUDA_VISIBLE_DEVICES=${1} python train.py --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random' --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --cross_validation

CUDA_VISIBLE_DEVICES=${1} python train.py --no_preprocessing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random' --no_preprocessing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --no_preprocessing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' --no_preprocessing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' --no_preprocessing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --no_preprocessing --cross_validation

CUDA_VISIBLE_DEVICES=${1} python train.py --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random' --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --no_fourier_transform --cross_validation

CUDA_VISIBLE_DEVICES=${1} python train.py --no_gaussian_smoothing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random' --no_gaussian_smoothing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --no_gaussian_smoothing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' --no_gaussian_smoothing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' --no_gaussian_smoothing --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --no_gaussian_smoothing --cross_validation

CUDA_VISIBLE_DEVICES=${1} python train.py --no_gaussian_smoothing --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'random' --no_gaussian_smoothing --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SMOTE' --no_gaussian_smoothing --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'BorderSMOTE' --no_gaussian_smoothing --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'SVMSMOTE' --no_gaussian_smoothing --no_fourier_transform --cross_validation
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --no_gaussian_smoothing --no_fourier_transform --cross_validation



