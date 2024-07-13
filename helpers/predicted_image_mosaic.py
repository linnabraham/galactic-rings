import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import sys,os
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from PIL import Image
"""
Script for predicting on a random subset of images and displaying the images as an 
image grid with the predicted probabilities and filenames printed on the image
"""

def custom_imggrid(imglist, nrows, ncols):
    imgarr = [ Image.open(img).convert('RGB') for img in imglist]

    fig = plt.figure(figsize=(14, 12))
    grid = ImageGrid(fig, 111, nrows_ncols = (nrows, ncols), axes_pad=0.1)
    count = 0
    for ax, im in zip(grid, imgarr):
        ax.imshow(im)
        #ax.text(10,40,  np.round(sorted_preds[count], 4),fontsize=8, color='red')
        ax.text(10,40,  sorted_preds[count], fontsize=8, color='red')
        #ax.text(10,240,  os.path.basename(imglist[count]),fontsize=6, color='red')
        count +=1
        ax.set_axis_off()
    #plt.savefig("predicted_imgs_mosaic_lof_20230925pt2.png", bbox_inches="tight")
    plt.show()


if __name__=="__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_script_dir, ".."))
    sys.path.append(parent_dir)

    from alexnet_utils.params import parser
    parser.add_argument('-pred_dir',  help="Directory containing images for prediction in a single sub-folder")
    parser.add_argument('-threshold', type=float,  default=0.5, help="Decimal threshold to use for creating CM, etc.")
    parser.add_argument('-model_path', default="best_model.h5", help="Filepath to save model during training and to load model from when testing")
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
 
    filepaths = test_ds.file_paths

    normalization_layer = layers.Rescaling(1./255)

    def change_inputs(images, labels):
      x = normalization_layer(images)
      return x, labels

    normalized_ds = test_ds.map(lambda images, labels: change_inputs(images, labels))
    AUTOTUNE = tf.data.AUTOTUNE

    ground_truth = []

    for image_batch, labels_batch in normalized_ds:
        flat = labels_batch.numpy().flatten()
        ground_truth.extend(flat)


    predictions = model.predict(normalized_ds)

    preds = [p[0] for p in predictions]
    enum = list(enumerate(preds))
    sorted_list = sorted(enum, key=lambda x: x[1])
    sorted_preds = [ item[1] for item in sorted_list]
    
    original_indices = [ item[0] for item in sorted_list ]

    sorted_files = [ filepaths[idx] for idx in original_indices]
    custom_imggrid(sorted_files, 5, 10)



