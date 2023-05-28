import pathlib
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers
import sys
import json

batch_size = 16
img_height = 240
img_width = 240


data_dir = './data/images_train/E5/'
data_dir = pathlib.Path(data_dir)
print(data_dir)

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="training",
  color_mode='grayscale',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="validation",
  color_mode='grayscale',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

normalization_layer = layers.Rescaling(1./255)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


class_names = train_ds.class_names
print(class_names)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
class AlexNet:
	@staticmethod
	def build(width=240, height=240, depth=1, classes=2, reg=0.0002):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# Block #1: first CONV => RELU => POOL layer set
		model.add(Conv2D(96, (5,5), strides=(2, 2),
		input_shape=inputShape, padding="same",
		kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.25))

		# Block #2: second CONV => RELU => POOL layer set
		model.add(Conv2D(256, (5, 5), padding="same",
		kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.25))

		# Block #3: CONV => RELU => CONV => RELU => CONV => RELU
		model.add(Conv2D(384, (3, 3), padding="same",
		kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(384, (3, 3), padding="same",
		kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(256, (3, 3), padding="same",
		kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
		model.add(Dropout(0.25))

		# Block #4: first set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(4096, kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# Block #5: second set of FC => RELU layers
		model.add(Dense(4096, kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(classes, kernel_regularizer=l2(reg)))
		model.add(Activation("sigmoid"))

		# return the constructed network architecture
		return model


opt = Adam(learning_rate=1e-3)
model = AlexNet.build(width=img_width, height=img_height, depth=1, classes=1, reg=0.0002)

print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy',\
                                        mode='max', verbose=1, save_best_only=True)
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

# Define the checkpoint path
history_path = 'history.json'
history_callback = SaveHistoryCallback(history_path)

epochs=50
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[mc,history_callback]
)
