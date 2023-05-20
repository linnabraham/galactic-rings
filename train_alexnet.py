from alexnet_utils.params import parser
import os
import pandas as pd
from alexnet_utils.alexnet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json,time
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from alexnet_utils.trainingmonitor import TrainingMonitor

def prepare(train_dir, validation_dir, TARGET_SIZE, COLOR_MODE, BATCH_SIZE):
    # define the preprocessing
    aug = ImageDataGenerator(rescale=1./255)

    # create generators for loading the training and validation data
    print("[INFO] Creating the train generator")
    train_generator = aug.flow_from_directory(
    train_dir,
    target_size=TARGET_SIZE,
    color_mode=COLOR_MODE,
    class_mode='categorical',
    #save_to_dir = '../data/augmented/',
    shuffle=True,
    batch_size=BATCH_SIZE)

    print("[INFO] Creating the validation generator")
    validation_generator = aug.flow_from_directory(
    validation_dir,
    target_size=TARGET_SIZE,
    color_mode=COLOR_MODE,
    class_mode='categorical',
    #save_to_dir='../data/augmented/',
    shuffle=True,
    batch_size=BATCH_SIZE)

    return train_generator, validation_generator


def train(pid, train_generator, validation_generator, TARGET_SIZE, CHANNELS, NUM_CLASSES , EPOCHS, OUTPUT_PATH):

        # initialize the optimizer
        opt = Adam(lr=1e-3)
        model = AlexNet.build(width=TARGET_SIZE[0], height=TARGET_SIZE[1], depth=CHANNELS, classes=NUM_CLASSES, reg=0.0002)

        print("[INFO] compiling model...")
        model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

        # construct the set of callbacks
        # path = os.path.sep.join([OUTPUT_PATH, "{}.png".format(pid)])
        jsonPath = os.path.sep.join([OUTPUT_PATH, "{}.json".format(pid)])
        modelPath = os.path.sep.join([OUTPUT_PATH, "{}.h5".format(pid)])
        print("[INFO] setting output paths...")
        print(f"Saving training history to {jsonPath}")
        # print(f"Writing training graphs to {path}")
        print(f"Saving model to {modelPath}")

        print(f"Saving filenames used for training and validation to csv")
        filenames=train_generator.filenames
        results=pd.DataFrame({"Filename":filenames})
        results.to_csv(os.path.sep.join([OUTPUT_PATH,f"{pid}_train_filenames.csv"]),index=False)

        filenames=validation_generator.filenames
        results=pd.DataFrame({"Filename":filenames})
        results.to_csv(os.path.sep.join([OUTPUT_PATH,f"{pid}_validation_filenames.csv"]),index=False)


        callbacks = [TrainingMonitor(jsonPath),
        #EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'),
            ModelCheckpoint(filepath=modelPath, monitor='val_loss', save_best_only=True)]

        # calculate the train and validation steps to be passed on to the generator
        train_steps = int(train_generator.samples/train_generator.batch_size)
        validation_steps = int(validation_generator.samples/validation_generator.batch_size)

        print("steps_per_epoch:","\ntraining:",train_steps,"\nvalidation:",validation_steps)

        start = time.time()

        #Fit the model using a batch generator
        history = model.fit_generator(
            train_generator,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=train_steps,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_steps)

        print("Total time taken for training: %d seconds" % (time.time()-start))

        return history



if __name__=="__main__":

    args = parser.parse_args()
    print("Using the following parameters", args)

    # define the constants

    NUM_CLASSES = 2
    CHANNELS = 3
    color_dict = {1:'grayscale',3:'rgb'}
    # define the path to the output directory used for storing plots,
    # classification reports, etc.
    OUTPUT_PATH = "output"
    COLOR_MODE = color_dict[CHANNELS]

    # obtain tweakable parameters from command line
    TARGET_SIZE = args.targetsize
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batchsize)
    base_dir = args.traindata
    IMAGES_PATH = args.images
    aug = args.noaugment # noaugment has default value of True
    AUGSCRIPT = args.augscript
    STEP = int(args.step)
    TRAIN_FRAC = float(args.trainfrac)
    VAL_FRAC   = float(args.valfrac)
    RANDOM_STATE = args.randomstate


    # obtain the process id to uniquely name outputs like saved model, training graphs etc.
    pid = os.getpid()

    # create a separate output directory for each run
    outdir = os.path.join(OUTPUT_PATH,f"{pid}")
    os.makedirs(outdir)

    train_dir           = os.path.join(base_dir, 'train')
    validation_dir      = os.path.join(base_dir, 'validation')

    train_gen, valid_gen = prepare(train_dir, validation_dir, TARGET_SIZE, COLOR_MODE, BATCH_SIZE)

    if args.runtrain:
        history = train(pid, train_gen, valid_gen, TARGET_SIZE, CHANNELS, NUM_CLASSES, EPOCHS, outdir)
