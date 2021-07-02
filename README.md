# Exoplanet Detection

## Requirements
> pip install -r requirements.txt

## Dataset
For training, we used the following [Exoplanet Hunting in Deep Space](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data). Copy both files *exoTrain.csv* and *exoTest.csv* into the subfolder *dataset*.


## Usage

### Pre-processing and Oversampling Experiment
> ./preprocessing_experiment.sh *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*) 

For stratified k-fold cross-validation use:
> ./preprocessing_experiment_kfold.sh *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*) 

### Window Sliding Experiment
> ./window_experiment.sh  *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

For stratified k-fold cross-validation use:
> ./window_experiment_kfold.sh  *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

### Time Warping and Gaussian Noise Experiment
> ./augmentation_experiment.sh  *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

For stratified k-fold cross-validation use:
> ./augmentation_experiment_kfold.sh  *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

### Training
> ./train.sh *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

For stratified k-fold cross-validation use:
> ./train_kfold.sh *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

### Testing 
Copy the file containing the model weights to root and name it *net_SMOTE.params*. By default there is already model weights file in root. Note: The folder *Backup* contains a backup of the default model weights file. Then run:
> ./test.sh *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

### Other Machine Learning Methods Experiment
Other machine learning methods include K-Nearest Neighbor Classifier, Support Vector Machine Classifier and Random Forest Classifier.
> ./compare_method_kfold.sh *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

## Contact
Leo.Nguyen@gmx.de

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)





