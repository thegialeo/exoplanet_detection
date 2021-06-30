#!/bin/bash
CUDA_VISIBLE_DEVICES=${1} python train.py --oversampling 'ADASYN' --cross_validation

