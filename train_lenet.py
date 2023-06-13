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
import numpy as np



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

def get_output_dir(output):
    # create a separate output directory for each run
    pid = os.getpid()
    outdir = os.path.join(output,f"{pid}")
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

def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [target_size[0], target_size[0]])
  image = (image / 255.0)
  return image, label


def prepare(ds, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    # Use data augmentation only on the training set.
    if augment :
        ds = ds.map(augment_custom, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
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


    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=1-train_frac,
      subset="both",
      color_mode='rgb',
      seed=random_state,
      image_size=target_size,
      batch_size=None)

    pid, outdir = get_output_dir(output)
    print(f"Outputs are saved to {outdir}")
    os.makedirs(outdir)


    # define callbacks
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', \
            mode='min', verbose=1, save_best_only=True)
    history_path = os.path.join(outdir,'history.json')
    hc = SaveHistoryCallback(history_path)

    # save filenames used for training and validation to disk
    print(f"Saving filenames used for training and validation to disk...")

    train_filenames = train_ds.file_paths
    val_filenames = val_ds.file_paths

    pd.DataFrame({"Filename":train_filenames}).to_csv(os.path.join(outdir,f"{pid}_train_filenames.csv"),index=False)
    pd.DataFrame({"Filename":val_filenames}).to_csv(os.path.join(outdir,f"{pid}_validation_filenames.csv"),index=False)

    #normalization_layer = layers.Rescaling(1./255)

    #train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    #val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # define custom augmentations
    def augment_custom(images, labels, seed=random_state):
        images, labels = resize_and_rescale(images, labels)
        # Make a new seed.
        new_seed = tf.random.experimental.stateless_split((seed,seed), num=1)[0, :]
        images = tf.image.stateless_random_flip_left_right(images, seed=new_seed)
        images = tf.image.stateless_random_flip_up_down(images, seed=new_seed)
        images = tf.image.stateless_random_brightness(images, max_delta=0.2, seed=new_seed)
        images = tf.image.stateless_random_contrast(images, lower=0.2, upper=0.5, seed=new_seed)
        images = random_int_rot_img(images,seed=seed)
        return (images, labels)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = prepare(train_ds, shuffle=True, augment=True)
    # do not shuffle or augment the validation dataset
    val_ds = prepare(val_ds)

    # save validation data for future evaluation
    def change_inputs(images, labels, paths):
      return images, labels,  tf.constant(paths)

    # save filenames also along with images and labels in the saved dataset
    val_ds_todisk = val_ds.map(lambda images, labels: change_inputs(images, labels, paths=val_filenames))

    path = os.path.join(outdir,"val_data")
    tf.data.Dataset.save(val_ds_todisk, path)

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
        def make_model(metrics=METRICS, output_bias=None):
            IMG_SIZE = target_size[0]

            input_shape = (IMG_SIZE, IMG_SIZE, channels)

            model = Sequential()
            if output_bias is not None:
                output_bias = tf.keras.initializers.Constant(output_bias)
                model.add(layers.Conv2D(filters=6, kernel_size=(5,5), activation='relu',input_shape=input_shape, bias_initializer=output_bias))
            else:
                model.add(layers.Conv2D(filters=6, kernel_size=(5,5), activation='relu',input_shape=input_shape))
            model.add(layers.AveragePooling2D(pool_size=(2,2)))

            model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
            model.add(layers.AveragePooling2D())

            model.add(layers.Flatten())
            model.add(layers.Dense(units=120, activation='relu'))
            model.add(layers.Dense(units=84, activation='relu'))
            model.add(layers.Dense(units=1, activation='sigmoid'))

            model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=METRICS)
            return model


        pos = train_ds.map(lambda _, label: tf.reduce_sum(label)).reduce(0, lambda count, val: count + val)
        neg = train_ds.map(lambda _, label: tf.reduce_sum(1 - label)).reduce(0, lambda count, val: count + val)
        pos = pos.numpy()
        neg = neg.numpy()

        initial_bias = np.log([pos/neg])
        print("[INFO] Calculated initial weight bias:", initial_bias)

        #model = make_model(metrics=METRICS, output_bias=initial_bias)
        model = make_model(metrics=METRICS)

    start = time()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,shuffle=True, callbacks=[mc,hc, tensorboard])

    print("Total time taken for training: %d seconds" % (time()-start))
