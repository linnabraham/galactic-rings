import os
import time
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from alexnet_utils.params import parser, print_arguments
from alexnet_utils.alexnet import AlexNet

class SaveHistoryCallback(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))

        with open(self.file_path, 'w') as f:
            json.dump(self.history, f)


# create an output directory to hold saved model, training graphs etc.
def make_output_dir(output):
    # create a separate output directory for each run
    pid = os.getpid()
    outdir = os.path.join(output,f"{pid}")
    print(f"Outputs are saved to {outdir}")
    os.makedirs(outdir)
    return pid, outdir

if __name__=="__main__":

    args = parser.parse_args()
    print_arguments(parser,args)

    data_dir = args.images
    target_size = args.target_size
    batch_size = args.batch_size
    train_frac = args.train_frac
    random_state = args.random_state
    num_classes = args.num_classes
    channels = args.channels
    output = args.output_dir
    epochs = args.epochs

    # set the color_mode from the number of channels

    #color_dict = {1:'grayscale',3:'rgb'}
    #color_mode = color_dict[channels]

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=1-train_frac,
      subset="both",
      color_mode='rgb',
      seed=random_state,
      image_size=target_size,
      batch_size=batch_size)

    class_names = train_ds.class_names

    print("Training dataset class names are :",class_names)

    pid, outdir = make_output_dir(output)

    print(f"Saving filenames used for training and validation to disk...")

    filenames = train_ds.file_paths
    results=pd.DataFrame({"Filename":filenames})
    results.to_csv(os.path.join(outdir,f"{pid}_train_filenames.csv"),index=False)

    filenames = val_ds.file_paths
    results=pd.DataFrame({"Filename":filenames})
    results.to_csv(os.path.join(outdir,f"{pid}_validation_filenames.csv"),index=False)
 

    # set the num_classes automatically from the number of directories found 
    # by the data generator
    #num_classes = len(class_names)

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # define callbacks
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', \
            mode='min', verbose=1, save_best_only=True)

    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    history_path = os.path.join(outdir,'history.json')
    history_callback = SaveHistoryCallback(history_path)

    classification_threshold = 0.5

    METRICS = [
          tf.keras.metrics.Precision(thresholds=classification_threshold,
                                     name='precision'),
          tf.keras.metrics.Recall(thresholds=classification_threshold,
                                  name="recall"),
          tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='auc_pr'),
    ]

    existing_modelpath = 'best_model.h5'

    if os.path.exists(existing_modelpath):
        print("[INFO] Loading existing model from disk ..")
        model = load_model(existing_modelpath)
    else:

        opt = Adam(learning_rate=1e-3)
        model = AlexNet.build(width=target_size[0], height=target_size[1], depth=channels, classes=1, reg=0.0002)


        print("[INFO] compiling model...")
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=METRICS)

    start = time.time()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, \
            callbacks=[mc,
            #es
            history_callback])

    print("Total time taken for training: %d seconds" % (time.time()-start))
