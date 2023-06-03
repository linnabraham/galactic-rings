from time import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from alexnet_utils.params import parser, print_arguments

model = Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(3,3), activation='relu',input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())
model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

classification_threshold = 0.5

METRICS = [
          tf.keras.metrics.Precision(thresholds=classification_threshold,
                                     name='precision'),
          tf.keras.metrics.Recall(thresholds=classification_threshold,
                                  name="recall"),
          tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='auc_pr'),
    ]

def make_output_dir(output):
    # create a separate output directory for each run
    pid = os.getpid()
    outdir = os.path.join(output,f"{pid}")
    print(f"Outputs are saved to {outdir}")
    os.makedirs(outdir)
    return pid, outdir

if __name__=="__main__":
    args = parser.parse_args()
    data_dir = args.images
    train_frac = args.train_frac
    random_state = args.random_state
    epochs = args.epochs
    batch_size = args.batch_size
    args.target_size = (32,32)
    target_size = args.target_size
    output = args.output_dir
    print_arguments(parser,args)


    pid, outdir = make_output_dir(output)

    # define callbacks
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', \
            mode='min', verbose=1, save_best_only=True)

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=1-train_frac,
      subset="both",
      color_mode='grayscale',
      seed=random_state,
      image_size=target_size,
      batch_size=batch_size)

    normalization_layer = layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    existing_modelpath = 'best_model.h5'

    if os.path.exists(existing_modelpath):
        print("[INFO] Loading existing model from disk ..")
        model = load_model(existing_modelpath)
    else:
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=METRICS)

    start = time()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,shuffle=True, callbacks=[tensorboard])

    print("Total time taken for training: %d seconds" % (time()-start))
