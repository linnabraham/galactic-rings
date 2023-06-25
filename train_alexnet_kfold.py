import os
from time import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from sklearn.model_selection import KFold
from alexnet_utils.alexnet import AlexNet


class SaveHistoryCallback(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.history = {'loss': [], 'val_loss': [], 'auc_pr':[], 'val_auc_pr':[]}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['auc_pr'].append(logs.get('auc_pr'))
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
    sample_index = tf.random.shuffle(indices,seed=seed)[:size]
    sample = tf.gather(x, sample_index, axis=axis)

    return sample, sample_index

def random_int_rot_img(inputs,seed):
    angles = tf.constant([1, 2, 3, 4])
    # Make a new seed.
    #new_seed = tf.random.experimental.stateless_split((seed,seed), num=1)[0, :]
    angle = random_choice(angles,1,seed=seed)[0][0]
    inputs = tf.image.rot90(inputs, k=angle)

    return inputs

def rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)

    return image, label

# define custom augmentations
def augment_custom(images, labels, augmentation_types, seed):
    
    images, labels = rescale(images, labels)
    # Make a new seed.
    #new_seed = tf.random.experimental.stateless_split((seed,seed), num=1)[0, :]
    new_seed = seed
    if 'rotation' in augmentation_types:
        images = random_int_rot_img(images,seed=seed)
    if 'flip' in augmentation_types:
        images = tf.image.random_flip_left_right(images, seed=new_seed)
        images = tf.image.random_flip_up_down(images, seed=new_seed)
    if 'brightness' in augmentation_types:
        images = tf.image.random_brightness(images, max_delta=0.2, seed=new_seed)
    if 'contrast' in augmentation_types:
        images = tf.image.random_contrast(images, lower=0.2, upper=0.5, seed=new_seed)

    return (images, labels)

if __name__=="__main__":

    from alexnet_utils.params import parser, print_arguments
    parser.add_argument('-kfolds', default=3 , type=int, help="No. of splits to use in k-fold splitting")
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
    model_path = args.model_path
    augmentation_types = args.augmentation_types


    # create an output directory to hold saved model, training graphs etc.
    pid, outdir = get_output_dir(output)
    print(f"Outputs are saved to {outdir}")
    os.makedirs(outdir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      color_mode='rgb',
      seed=random_state,
      image_size=target_size,
      batch_size=None)

    X = np.array([tensor.numpy() for tensor in list(train_ds.map(lambda x, y:x))])
    Y = np.array([tensor.numpy() for tensor in list(train_ds.map(lambda x, y:y))])
    print(X.shape)

    # Initialize the k-fold object
    num_folds = args.kfolds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        X_val = X[test_index]
        Y_val = Y[test_index]
        print(X_val.shape)
        print("Validation set created")

        val_ds  = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        print("Length of val_ds;", len(val_ds))
        del X_val, Y_val

        X_train = X[train_index]
        Y_train = Y[train_index]
        print("Training set created")

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        print("Length of train_ds;", len(train_ds))
        del X_train, Y_train

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = (
                train_ds
                .shuffle(1000)
                .map(lambda x, y: augment_custom(x, y, augmentation_types, seed=random_state), num_parallel_calls=AUTOTUNE)
                #.cache()
                .batch(batch_size)
                .prefetch(buffer_size=AUTOTUNE)
                )

        val_ds = (
                val_ds
                .map(rescale, num_parallel_calls=AUTOTUNE)
                #.cache()
                .batch(batch_size)
                .prefetch(buffer_size=AUTOTUNE)
                )
        # define callbacks
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        mc = ModelCheckpoint(f"{i}_{model_path}", monitor='val_auc_pr', \
                mode='max', verbose=1, save_best_only=True)
        history_path = os.path.join(outdir,f"{i}_history.json")
        hc = SaveHistoryCallback(history_path)

        classification_threshold = 0.5

        METRICS = [
              tf.keras.metrics.Precision(thresholds=classification_threshold,
                                         name='precision'),
              tf.keras.metrics.Recall(thresholds=classification_threshold,
                                      name="recall"),
              tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='auc_pr'),
        ]

        model = AlexNet.build(width=target_size[0], height=target_size[1], depth=channels, classes=1, reg=0.0002)


        print("[INFO] compiling model...")
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)

        start = time()

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, shuffle=True, callbacks=[mc, hc, tensorboard])

        print("Total time taken for training: %d seconds" % (time()-start))





