import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np

if __name__=="__main__":

    from alexnet_utils.params import parser
    parser.add_argument('-pred_dir',  help="Directory containing images for prediction in a single sub-folder")
    args = parser.parse_args()

    modelpath = args.model_path
    data_dir = args.pred_dir
    target_size = args.target_size
    img_height, img_width = target_size
    batch_size = args.batch_size

    # Load the pre-trained model
    model = load_model(modelpath)
    # Return pretty much every information about your model
    config = model.get_config() 
    # Return a tuple of width, height and channels as the expected input shape
    print("Expected input shape for the model")
    print(config["layers"][0]["config"]["batch_input_shape"]) 

    test_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      #color_mode='grayscale',
      shuffle=False,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
 
    normalization_layer = layers.Rescaling(1./255)

    def change_inputs(images, labels, paths):
      x = normalization_layer(images)
      return x, labels,  tf.constant(paths)

    normalized_ds = test_ds.map(lambda images, labels: change_inputs(images, labels, paths=test_ds.file_paths))
    AUTOTUNE = tf.data.AUTOTUNE
    normalized_ds = normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)

    ground_truth = []

    for image_batch, labels_batch, paths in normalized_ds:
        flat = labels_batch.numpy().flatten()
        ground_truth.extend(flat)

    images, labels, paths = next(iter(normalized_ds.take(1)))
    filenames = [ path.numpy().decode('utf-8') for path in paths]

    predictions = model.predict(normalized_ds)

    import csv

    # Example label dictionary for decoding
    label_dict = {0: "NonRings", 1: "Rings"}

    # Decode predicted labels from indices to text labels
    decoded_labels = [label_dict[np.argmax(pred)] for pred in predictions]

    # Combine filename, second column of numpy array, and predicted labels
    rows = [[filename, pred[1], label] for filename, pred, label in zip(filenames, predictions, decoded_labels)]

    # Save rows to a CSV file
    csv_filename = "pred_output.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Prediction", "Label"])  # Write header row
        writer.writerows(rows)
