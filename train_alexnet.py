import os
from keras.preprocessing.image import ImageDataGenerator
import time
import pandas as pd
from alexnet_utils.params import parser
from keras.optimizers import Adam
from alexnet_utils.alexnet import AlexNet
from alexnet_utils.trainingmonitor import TrainingMonitor
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

    # create generators for loading the training and validation data
def create_data_gen(TARGET_SIZE, COLOR_MODE, BATCH_SIZE, args):
    base_dir = args.traindata
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # define the preprocessing
    aug = ImageDataGenerator(rescale=1./255)
    print("[INFO] Creating the train generator")
    #save_to_dir = '../data/augmented/',
    train_generator = aug.flow_from_directory(train_dir, target_size=TARGET_SIZE, color_mode=COLOR_MODE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    print("[INFO] Creating the validation generator")
    validation_generator = aug.flow_from_directory(validation_dir, target_size=TARGET_SIZE, color_mode=COLOR_MODE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    return train_generator, validation_generator

    # create an output directory to hold saved model, training graphs etc.
def make_output_dir(output):
    # create a separate output directory for each run
    pid = os.getpid()
    outdir = os.path.join(output,f"{pid}")
    print(f"Outputs are saved to {outdir}")
    os.makedirs(outdir)
    return pid, outdir

    # construct the set of callbacks
def create_callbacks(pid, outdir):
    jsonPath = os.path.sep.join([outdir, "{}.json".format(pid)])
    modelPath = os.path.sep.join([outdir, "{}.h5".format(pid)])

    callbacks = [TrainingMonitor(jsonPath),
    #EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'),
        ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True)]
    return callbacks

    # save a list of filenames used for training and validation
def save_filelists(train_generator, validation_generator, pid, outdir):

    print(f"Saving filenames used for training and validation to csv")

    filenames=train_generator.filenames
    results=pd.DataFrame({"Filename":filenames})
    results.to_csv(os.path.sep.join([outdir,f"{pid}_train_filenames.csv"]),index=False)

    filenames=validation_generator.filenames
    results=pd.DataFrame({"Filename":filenames})
    results.to_csv(os.path.sep.join([outdir,f"{pid}_validation_filenames.csv"]),index=False)


def train_model(train_generator, validation_generator, TARGET_SIZE, CHANNELS, NUM_CLASSES, EPOCHS, args):
    
    pid, outdir = make_output_dir(args.output)
    save_filelists(train_generator, validation_generator, pid, outdir)
    callbacks = create_callbacks(pid, outdir)

    # initialize the optimizer
    opt = Adam(lr=1e-3)
    model = AlexNet.build(width=TARGET_SIZE[0], height=TARGET_SIZE[1], depth=CHANNELS, classes=NUM_CLASSES, reg=0.0002)

    print("[INFO] compiling model...")
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # calculate the train and validation steps to be passed on to the generator
    train_steps = int(train_generator.samples/train_generator.batch_size)
    validation_steps = int(validation_generator.samples/validation_generator.batch_size)
    print("steps_per_epoch:","\ntraining:",train_steps,"\nvalidation:",validation_steps)

    #Fit the model using a batch generator
    history = model.fit_generator(train_generator, callbacks=callbacks, verbose=1, steps_per_epoch=train_steps, epochs=EPOCHS, validation_data=validation_generator, validation_steps=validation_steps)

    return history


if __name__=="__main__":

    args = parser.parse_args()
    print("Using the following parameters", args)

    CHANNELS = args.channels
    NUM_CLASSES = args.numclasses
    color_dict = {1:'grayscale',3:'rgb'}
    COLOR_MODE = color_dict[args.channels]
    TARGET_SIZE = args.targetsize
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batchsize)

    train_generator, validation_generator = create_data_gen(TARGET_SIZE, COLOR_MODE, BATCH_SIZE, args)

    start = time.time()
    history = train_model(train_generator, validation_generator, TARGET_SIZE, CHANNELS, NUM_CLASSES, EPOCHS, args)
    print("Total time taken for training: %d seconds" % (time.time()-start))
