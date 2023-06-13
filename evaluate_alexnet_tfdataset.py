import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np

if __name__=="__main__":

    from alexnet_utils.params import parser
    parser.add_argument('-model_path',  help="Path of pre-trained model")
    parser.add_argument('-test_dir',  help="Path of tf.data.Dataset object")
    parser.add_argument('-write', action="store_true", help="Switch to enable writing results to disk")
    args = parser.parse_args()

    model_path = args.model_path
    test_dir = args.test_dir
    target_size = args.target_size
    img_height, img_width = target_size
    batch_size = args.batch_size
    img_height, img_width = target_size


    # Load the pre-trained model
    model = load_model(model_path)
    # Return pretty much every information about your model
    config = model.get_config() 

    # Return a tuple of width, height and channels as the expected input shape
    print("Expected input shape for the model")
    print(config["layers"][0]["config"]["batch_input_shape"]) 

    test_ds = tf.data.Dataset.load(test_dir)

    images, labels, paths = next(iter(test_ds.take(1)))
    filenames = [ path.numpy().decode('utf-8') for path in paths]

    def change_inputs(images, labels, paths):
        return images, labels

    test_ds = test_ds.map(change_inputs)

    predictions = model.predict(test_ds)
    
    # create the ground truth
    labels = test_ds.map(lambda _, label: label).unbatch()
    ground_truth = list(labels.as_numpy_iterator())

    threshold = 0.5
    print("Using a classification threshold", threshold)

    #predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels = [1 if pred >= threshold else 0 for pred in predictions] 

    # Compute the confusion matrix
    from sklearn.metrics import confusion_matrix

    confusion_mtx = confusion_matrix(ground_truth, predicted_labels)
    print("Confusion Matrix:")
    print(confusion_mtx)

    # Compute other accuracy metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

    accuracy = accuracy_score(ground_truth, predicted_labels)
    precision = precision_score(ground_truth, predicted_labels)
    recall = recall_score(ground_truth, predicted_labels)
    f1 = f1_score(ground_truth, predicted_labels)
    roc_auc = roc_auc_score(ground_truth, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC AUC Score:", roc_auc)

   # Automatically compute a classification threshold using the ROC in order to maximize the evaluation metrics
    false_pos_rate, true_pos_rate, proba = roc_curve(ground_truth, predictions)
    optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
    print("Optimal probability cutoff", optimal_proba_cutoff)

    roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in predictions]

    #print("Accuracy Score Before and After Thresholding:  {}".format(accuracy_score(y_test, predictions), accuracy_score(y_test, roc_predictions)))
    print("Precision Score After Thresholding: {}".format( precision_score(ground_truth, roc_predictions)))
    print("Recall Score After Thresholding: {}".format(recall_score(ground_truth, roc_predictions)))
    print("F1 Score After Thresholding: {}".format( f1_score(ground_truth, roc_predictions)))
    if args.write:

        import csv

        # Example label dictionary for decoding
        label_dict = {0: "NonRings", 1: "Rings"}

        # Decode predicted labels from indices to text labels
        predicted_labels = [1 if pred >= threshold else 0 for pred in predictions] 
        decoded_labels = [label_dict[label] for label in predicted_labels]

        # Combine filename, second column of numpy array, and predicted labels
        rows = [[filename, pred, label] for filename, pred, label in zip(filenames, predictions, decoded_labels)]

        # Save rows to a CSV file
        csv_filename = "eval_output.csv"

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "Prediction", "Label"])  # Write header row
            writer.writerows(rows)

 
