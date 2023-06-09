from time import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from alexnet_utils.params import parser, print_arguments
import json
import pandas as pd

def create_lenet(target_size, channels):

    input_shape = (target_size[0], target_size[1], channels)

    model = Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(10,10), activation='relu',input_shape=input_shape))
    model.add(layers.AveragePooling2D(pool_size=(4,4)))

    model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='relu'))
    model.add(layers.Dense(units=84, activation='relu'))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    return model


class SaveHistoryCallback(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.history = {'loss': [], 'auc_pr': [], 'val_loss': [], 'val_auc_pr': []}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['auc_pr'].append(logs.get('auc_pr'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_auc_pr'].append(logs.get('val_auc_pr'))
        
        with open(self.file_path, 'w') as f:
            json.dump(self.history, f)

def make_output_dir(output):
    # create a separate output directory for each run
    pid = os.getpid()
    outdir = os.path.join(output,f"{pid}")
    print(f"Outputs are saved to {outdir}")
    os.makedirs(outdir)
    return pid, outdir

def random_choice(x, size, seed, axis=0, unique=True):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.experimental.stateless_shuffle(indices,seed=seed)[:size]
    sample = tf.gather(x, sample_index, axis=axis)

    return sample, sample_index

def random_int_rot_img(inputs,seed):
    angles = tf.constant([1, 2, 3, 4])
    # Make a new seed.
    new_seed = tf.random.experimental.stateless_split((seed,seed), num=1)[0, :]
    angle = random_choice(angles,1,seed=new_seed)[0][0]
    inputs = tf.image.rot90(inputs, k=angle)
    return inputs

    # define custom augmentations
def augment_custom(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    #images = tf.image.rgb_to_grayscale(images)
    images = random_int_rot_img(images,seed=123)
    return (images, labels)

def prepare(ds, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    # Use data augmentation only on the training set.
    if augment :
        ds = ds.map(augment_custom, num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

if __name__=="__main__":
    args = parser.parse_args()
    data_dir = args.images
    train_frac = args.train_frac
    random_state = args.random_state
    epochs = args.epochs
    batch_size = args.batch_size
    target_size = args.target_size
    output = args.output_dir
    channels = args.channels
    print_arguments(parser,args)


    pid, outdir = make_output_dir(output)

    # define callbacks
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', \
            mode='min', verbose=1, save_best_only=True)
    history_path = os.path.join(outdir,'history.json')
    hc = SaveHistoryCallback(history_path)


    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=1-train_frac,
      subset="both",
      color_mode='rgb',
      seed=random_state,
      image_size=target_size,
      batch_size=None)

    # save filenames used for training and validation to disk
    print(f"Saving filenames used for training and validation to disk...")

    filenames = train_ds.file_paths
    results=pd.DataFrame({"Filename":filenames})
    results.to_csv(os.path.join(outdir,f"{pid}_train_filenames.csv"),index=False)

    filenames = val_ds.file_paths
    results=pd.DataFrame({"Filename":filenames})
    results.to_csv(os.path.join(outdir,f"{pid}_validation_filenames.csv"),index=False)

    normalization_layer = layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = prepare(train_ds, shuffle=True, augment=True)
    val_ds = prepare(val_ds)
    # save validation data for future evaluation
    path = os.path.join(outdir,"val_data")
    tf.data.Dataset.save(dataset, path)

    existing_modelpath = 'best_model.h5'

    if os.path.exists(existing_modelpath):
        print("[INFO] Loading existing model from disk ..")
        model = load_model(existing_modelpath)
    else:
        classification_threshold = 0.5

        METRICS = [
                  tf.keras.metrics.Precision(thresholds=classification_threshold,
                                             name='precision'),
                  tf.keras.metrics.Recall(thresholds=classification_threshold,
                                          name="recall"),
                  tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='auc_pr'),
            ]
        model = create_lenet(target_size, channels)
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=METRICS)

    start = time()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,shuffle=True, callbacks=[mc,hc, tensorboard])

    print("Total time taken for training: %d seconds" % (time()-start))
