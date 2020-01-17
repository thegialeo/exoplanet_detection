# Exoplanet Detection

## Requirements
> pip install -r requirements.txt

## Dataset
For training, we used the following [Exoplanet Hunting in Deep Space](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data). Copy both files *exoTrain.csv* and *exoTest.csv* into the subfolder *dataset*.


## Usage

### Pre-processing and Oversampling Experiment
> ./preprocessing_experiment.sh *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*) 

### Window Sliding Experiment
> ./window_experiment.sh  *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

### Time Warping and Gaussian Noise Experiment
> ./augmentation_experiment.sh  *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

### Training
> ./train.sh *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

### Testing 
Copy the file containing the model weights to root and name it *net_SMOTE.params*. By default there is already model weights file in root. Note: The folder *Backup* contains a backup of the default model weights file. Then run:
> ./test.sh *CUDA_VISIBLE_DEVICES* (if you only have one GPU: put 0 for *CUDA_VISIBLE_DEVICES*)

## Backup
Running train.py will overwrite pretrained models and plots. If this happens by accident and you want to retrieve the files, go the the folder *Backup* and copy the files back into the corresponding folders again.

## Contact
Leo.Nguyen@gmx.de

## License
MIT License





