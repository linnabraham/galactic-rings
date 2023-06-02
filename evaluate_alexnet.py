import os
from keras.models import load_model, model_from_json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np

def read_images(image_paths):
    images = []
    for path in image_paths:

        image = Image.open(path)
        images.append(image)
    return images


if __name__=="__main__":

    from alexnet_utils.params import parser
    parser.add_argument('-model_path',  help="path containing pre-trained model")
    parser.add_argument('-inputs',  help="path of image or csv file for prediction")
    args = parser.parse_args()

    # Load the pre-trained model
    model = load_model(args.model_path)
    config = model.get_config() # Returns pretty much every information about your model
    print("Expected input shape for the model")
    print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels

    dataset_dir = 'data/train_data/base/test/'

    image_paths = []
    labels = []

    # Iterate over the subfolders
    for class_name in os.listdir(dataset_dir):
        print(class_name)
        class_dir = os.path.join(dataset_dir, class_name)
        print(class_dir)
        if os.path.isdir(class_dir):
            # Get the class label from the folder name
            label = class_name  
            
            # Iterate over the image files in the subfolder
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpeg'):
                    image_path = os.path.join(class_dir, filename)
                    image_paths.append(image_path)
                    labels.append(label)


    # Encode labels as 1s and 0s
    encoded_labels = [1 if label == 'Rings' else 0 for label in labels]
    ground_truth_labels = encoded_labels
 
    images = read_images(image_paths)
    images_array = np.stack(images)
    print(images_array.shape)

    rescaled_imgs = images_array / 255.0
    predictions = model.predict(rescaled_imgs)
    print(predictions)

    # save prediction probabilities to disk
    filenames = images_array.reshape(-1,1)
    stack = np.hstack([filenames, predictions])
    header = "filenames, " + ", ".join(["NonRings","Rings"])
    savepath =f"output/probabilities_{os.getpid()}.csv"
    np.savetxt(savepath,stack,delimiter=',',fmt='%s',header = header)

    # Convert the predicted probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Compute the confusion matrix
    from sklearn.metrics import confusion_matrix

    confusion_mtx = confusion_matrix(ground_truth_labels, predicted_labels)
    print("Confusion Matrix:")
    print(confusion_mtx)

    # Compute other accuracy metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels)
    recall = recall_score(ground_truth_labels, predicted_labels)
    f1 = f1_score(ground_truth_labels, predicted_labels)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
