import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np

def load_dataset(dataset):
    test_ds = tf.data.Dataset.load(dataset)

    images, labels, paths = next(iter(test_ds.take(1)))
    filenames = [ path.numpy().decode('utf-8') for path in paths]

    test_ds = test_ds.map(lambda images, labels, paths: (images, labels))
    return filenames, test_ds

def precision_at_recall(y_true, y_scores, recall_threshold):
    # Sort the predictions by score in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    # Calculate the cumulative sum of true positives
    cumsum = np.cumsum(sorted_labels)
    
    # Calculate the recall and precision for each threshold
    recall = cumsum / np.sum(y_true)
    precision = cumsum / np.arange(1, len(y_true) + 1)
    
    # Find the index where the recall crosses the given threshold
    index = np.argmax(recall >= recall_threshold)
    
    # Return the precision at the specified recall threshold
    return precision[index]

def recall_at_precision(y_true, y_scores, precision_threshold):
    # Sort the predictions by score in descending order
    sorted_indices = np.argsort(y_scores.flatten())[::-1]
    sorted_labels = y_true[sorted_indices]
    
    # Calculate the cumulative sum of true positives
    cumsum = np.cumsum(sorted_labels)
    
    # Calculate the precision and recall for each threshold
    recall = cumsum / np.sum(y_true)
    precision = cumsum / np.arange(1, len(y_true) + 1)
    
    # Find the index where the precision reaches 1.0
    index = np.argmax(precision >= precision_threshold)
    
    # Return the recall at the specified precision threshold
    return recall[index]

if __name__=="__main__":

    from alexnet_utils.params import parser
    parser.add_argument('-test_dir',  help="Directory containing validation or test images sorted into respective classes")
    parser.add_argument('-saved_ds',  default=False, help="Boolean flag that is true if test_dir points to a tf.data.Dataset object")
    parser.add_argument('-threshold', type=float,  default=0.5, help="Decimal threshold to use for creating CM, etc.")
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

    if not args.saved_ds:
        test_ds = tf.keras.utils.image_dataset_from_directory(
          test_dir,
          #color_mode='grayscale',
          shuffle=False,
          image_size=(img_height, img_width),
          batch_size=None)

        filenames = test_ds.file_paths
        normalization_layer = layers.Rescaling(1./255)
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
        labels = test_ds.map(lambda _, label: label)
        AUTOTUNE = tf.data.AUTOTUNE
        test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    else:
        filenames, test_ds = load_dataset(args.test_dir)
        labels = test_ds.unbatch().map(lambda _, label: label)

    ground_truth = list(labels.as_numpy_iterator())

    predictions = model.predict(test_ds)

    threshold = args.threshold
    print("Using a classification threshold", threshold)

    #predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels = [1 if pred >= threshold else 0 for pred in predictions] 

    # Compute the confusion matrix
    from sklearn.metrics import confusion_matrix

    confusion_mtx = confusion_matrix(ground_truth, predicted_labels)
    print("Confusion Matrix:")
    print(confusion_mtx)

    # Compute other accuracy metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    balanced_accuracy_score, brier_score_loss, average_precision_score, fbeta_score, matthews_corrcoef, auc, precision_recall_curve

    accuracy = accuracy_score(ground_truth, predicted_labels)
    precision = precision_score(ground_truth, predicted_labels)
    recall = recall_score(ground_truth, predicted_labels)
    f1 = f1_score(ground_truth, predicted_labels)
    try:
        roc_auc = roc_auc_score(ground_truth, predictions)
    except:
        print("Setting roc_auc to be -1 as it is not defined")
        roc_auc = -1
    precisions, recalls, thresholds = precision_recall_curve(ground_truth, predictions)
    pr_auc = auc(recalls, precisions)
    brier_score = brier_score_loss(ground_truth, predictions)
    avg_precision = average_precision_score(ground_truth, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC AUC Score:", roc_auc)
    print("PR AUC Score:", pr_auc)
    print("Brier score", brier_score)
    print(f"Average precision score", avg_precision)

   # Automatically compute a classification threshold using the ROC in order to maximize the evaluation metrics
    false_pos_rate, true_pos_rate, proba = roc_curve(ground_truth, predictions)
    optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
    print("Optimal probability cutoff", optimal_proba_cutoff)

    roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in predictions]

    recall_threshold = 0.75
    precision = precision_at_recall(np.array(ground_truth), predictions, recall_threshold)
    print(f"Precision at recall {recall_threshold}: {precision}")
    precision_threshold = 0.8
    recall = recall_at_precision(np.array(ground_truth), predictions, precision_threshold)
    print(f"Recall at precision {precision_threshold}: {recall}")

    confusion_mtx = confusion_matrix(ground_truth, roc_predictions)
    print("New Confusion Matrix:")
    print(confusion_mtx)
    print("Accuracy Score After Thresholding:  {}".format(accuracy_score(ground_truth, roc_predictions)))
    tn, fp, fn, tp = confusion_mtx.ravel()
    fpr = fp / (fp + tn)
    specificity = tn / (fp + tn)
    precision_score = precision_score(ground_truth, roc_predictions)
    recall_score = recall_score(ground_truth, roc_predictions)
    bal_acc = balanced_accuracy_score(ground_truth, roc_predictions)
    matthews_corrcoef = matthews_corrcoef(ground_truth, roc_predictions)
    beta = 2
    fbeta = fbeta_score(ground_truth, roc_predictions, beta=beta)
    print("False Positive Rate (FPR):", fpr)
    print("TNR or Specificity:", specificity)
    print("Precision Score After Thresholding: {}".format( precision_score))
    print("Recall Score After Thresholding: {}".format(recall_score))
    print("G-Mean:", np.sqrt(recall_score * specificity))
    print(f"F-beta score beta={beta}", fbeta)
    print("F1 Score After Thresholding: {}".format( f1_score(ground_truth, roc_predictions)))
    print("Matthew Correlation Coefficient:", matthews_corrcoef)
    print("Balanced Accuracy:", bal_acc)
    if args.write:

        import csv

        # Example label dictionary for decoding
        label_dict = {0: "NonRings", 1: "Rings"}

        # Decode predicted labels from indices to text labels
        decoded_labels = [label_dict[label] for label in predicted_labels]

        # Combine filename, second column of numpy array, and predicted labels
        rows = [[filename, pred[0], gt, label] for filename, pred, gt, label in zip(filenames, predictions, ground_truth, decoded_labels)]

        # Save rows to a CSV file
        csv_filename = "eval_output.csv"

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "Prediction", "Ground_Truth", "Label"])  # Write header row
            writer.writerows(rows)

 
