import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-images', default="raw_train/E1", help="path containing images of two classes")
parser.add_argument('-traindata', default="train_data/base", help="path containing train validation and test directories")
parser.add_argument('-targetsize', default=(240,240), help="target size to resize images to before training")
parser.add_argument('-epochs', default=50, help="num epochs")
parser.add_argument('-batchsize', default=16, help="batch size for training")
parser.add_argument('-trainfrac', default=0.65, help="fraction to use for the train sample")
parser.add_argument('-valfrac', default=0.15, help="fraction to use for the validation sample")
parser.add_argument('-randomstate', default=42, help="seed for random processes for reproducibility")
parser.add_argument('-runtrain', action="store_true", help="switch to run the training")
parser.add_argument('-maketrain', action="store_true", help="switch to make the training data")
parser.add_argument('-numclasses', default=2, help="Number of classes")
parser.add_argument('-channels', default=3, help="Number of channels in the image data")
parser.add_argument('-output', default="output", help="Location to store the outputs generated during training")

