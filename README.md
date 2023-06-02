# How to use this repo

## Setup the environment
~~This project works best with python 3.7 and tensorflow 1.15~~

Project works with a python 3.10 and tensorflow 2.12 environment

```
pip install -r requirements.txt
```

## Download the data
```python download_data.py```

data downloads as a zip file by default inside `data/images_train`

unzip the file and place it in a folder inside the `images_train` folder

## Create training data

`python makedata.py -images data/images_train/E2`

The script takes the the images within the original folder and splits it into three sets for training validation and testing.
An augmentation is then run on the training set of the Rings class using the augmentation script parameter.
The script used for augmentation can be found in the `helpers` directory.
The non-rings class is not augmented.

## Run the training

`python train_alexnet.py -train_data data/train_data/base/ -epochs 1 -batch_size 64`

The training script saves the outputs generated during training to a folder `outputs`

Each training session creates a different subfolder based on the process id 

## Evaluate the performance of the trained model on validation or test data

`python evaluate_alexnet.py  -test_dir "data/train_data/base/validation/" -model_path output/741649/741649.h5`

The prediction results are saved to a file eval_output.csv

## Predict on a large number of unlabelled images

`python predict_alexnet.py -pred_dir data/Panstarrs/dummy -model_path output/741649/741649.h5 -batch_size 64`

The predictions are saved to a file called pred_output.csv


### Predict single 

```python predict_single.py -model_path output/263861/263861.h5 -inputs nair_common_withpredict.csv```

Create a csv file containing just the filepaths of images for predictions using

```
find path/to/images/ -type f > images_for_prediction.csv
```

The predictions are saved to a file called `predictions.csv` 

The prediction script can also predict on single images by giving an image path instead of a csv path.


