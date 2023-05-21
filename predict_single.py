from keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
import csv

def read_csv_file(csv_path):
    image_paths = []
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            image_paths.extend(row)
    return image_paths

def get_image_array(img_path, target_size):
    my_image = image.load_img(img_path,target_size=target_size)
    img_array = image.img_to_array(my_image)
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale the image manually
    rescaled_img = img_array / 255.0
    return rescaled_img

def predict_image(model, input_path, args):
    # Check if the input path is a CSV file
    if input_path.endswith('.csv'):
        # Read the CSV file and extract the image paths
        image_paths = read_csv_file(input_path)

        # Define the output file path
        output_file = 'predictions.csv'

        # Write the predictions to the CSV file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image Path','Prediction'])  # Write the header

            # Process each image path and make predictions
            for image_path in image_paths:
                prediction = predict_single_image(model, image_path, args)
                writer.writerow([image_path,prediction])

    else:
        # Assume the input path is a single image file
        prediction = predict_single_image(model, input_path, args)
        
        print(f"Predictions:\n{prediction}")

def predict_single_image(model, image_path, args):
    img_array = get_image_array(img_path=image_path, target_size=args.targetsize)
    predictions = model.predict(img_array)
    return predictions

if __name__=="__main__":

    from alexnet_utils.params import parser
    parser.add_argument('-modelpath',  help="path containing pre-trained model")
    parser.add_argument('-inputs',  help="path of image or csv file for prediction")
    args = parser.parse_args()

    # Load the pre-trained model
    model = load_model(args.modelpath)
    config = model.get_config() # Returns pretty much every information about your model
    print("Expected input shape for the model")
    print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels


    predict_image(model, args.inputs, args)
