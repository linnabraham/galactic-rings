# How to use this repo

## Setup the environment
~~This project works best with python 3.7 and tensorflow 1.15~~

Project works with a python 3.10 and tensorflow 2.12 environment

```
pip install -r requirements.txt
```

## Train AlexNet

`python train_alexnet.py -images data/images_train/E2 -batch_size 16`

The data is split into training and validations sets internally by tf data loaders. 
The model architecture is defined in the file `alexnet-utils/alexnet.py`. Default parameters are defined in `alexnet-utils/params.py`.
The training currently uses tensorflow "on-the-fly" augmentations done on both classes.

The augmentations currently employed are:
+ Random Flip (Horizontal and Vertical)
+ Random 90 degree rotations
+ Random Brightness Adjustment
+ Random Contrast Enhancement

The model is saved to a file `best_model.h5` in the base of the directory. If this file already exists, the model is loaded from this file when the script is run.
The best model is currently defined as the one which has the lowest `validation loss`.

The training script saves the outputs generated during training to a folder `outputs`. Each training session creates a different subfolder based on the process id.
It contains the following:

	List of filenames used for training and validation are separately saved as text files.
	The validation data along with filenames is saved as `tf.data.Dataset` object to a folder named `val_data`.
	The training and validation metrics are saved to a file named `history.json`. The file can be viewed using the command `python training-graphs.py output/pid/history.json`

## Train LeNet

` python train_lenet.py -images data/images_train/E2/`

The LeNet training script includes `augmentations`. These are on-the-fly augmentations done using tf.Image class. Current augmentations include (RandomFlips=(Horizontal and Vertical), Integral rotations by 90 deg)

## Evaluate the performance of the trained model on validation or test data

`python evaluate_alexnet.py  -test_dir "data/train_data/base/validation/" -model_path output/741649/741649.h5 -batch_size 64`

The prediction results are saved to a file eval_output.csv

## Predict on a large number of unlabelled images

`python predict_alexnet.py -pred_dir data/Panstarrs/dummy -model_path output/741649/741649.h5 -batch_size 64`

The predictions are saved to a file called pred_output.csv


# Deprecated Workflow
## Create training data

`python makedata.py -images data/images_train/E2`

The script takes the the images within the original folder and splits it into three sets for training validation and testing.
An augmentation is then run on the training set of the Rings class using the augmentation script parameter.
The script used for augmentation can be found in the `helpers` directory.
The non-rings class is not augmented.

## Download the data
```python download_data.py```

data downloads as a zip file by default inside `data/images_train`

unzip the file and place it in a folder inside the `images_train` folder

### Predict single 

```python predict_single.py -model_path output/263861/263861.h5 -inputs nair_common_withpredict.csv```

Create a csv file containing just the filepaths of images for predictions using

```
find path/to/images/ -type f > images_for_prediction.csv
```

The predictions are saved to a file called `predictions.csv` 

The prediction script can also predict on single images by giving an image path instead of a csv path.

# Things to do

+ Deal with the class imbalance in the training data. Some possibilities are:
	+ Oversampling the minority class using rejection sampling etc.
	+ Using a biased initial weights for faster convergence
	+ Bias the cost function
+ Create a new metric to track during model training that is a weighted sum of precision and recall. This metric should be based on our goals. Specifically the desired purity(precision) vs completness(recall) 
of the astronomical catalog
