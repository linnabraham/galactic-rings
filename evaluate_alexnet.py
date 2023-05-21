from alexnet_utils.params import parser
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, model_from_json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from alexnet_utils.plotcm import plot_confusion_matrix


def predict(modelId, test_dir,TARGET_SIZE):



    print("[INFO] loading model...")
    modelpath = os.path.sep.join([outdir,"{}.h5".format(modelId)])
    print(f"Using model from {modelpath}")
    model = load_model(modelpath)
    config = model.get_config() # Returns pretty much every information about your model
    print("Expected input shape for the model")
    print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels


    test_generator = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    test_dir,
    target_size=TARGET_SIZE,
    shuffle = False,
    #class_mode = None,
    class_mode ='categorical',
    color_mode=COLOR_MODE,
    batch_size=1)

    print("[INFO] test generator information \n")

    for data_batch, labels_batch in test_generator:
            print('data batch shape:', data_batch.shape)
            print('labels batch shape:', labels_batch.shape)
            break

    steps = int(test_generator.samples/test_generator.batch_size)
    print("step size",steps)

    print("test_generator.class_indices", test_generator.class_indices)

    # sort the class indices list as sometimes the order may change
    classes = sorted(test_generator.class_indices.keys())
    print("target names:",classes)

    print("[INFO] predicting on test data (no crops)...")

    probabilities = model.predict_generator(test_generator, steps= steps, verbose =1)

    print("predicted probabilities: \n",probabilities[:10])
    print("shape of predicted proabilities:",probabilities.shape)

    # obtain the true labels
    y_true = test_generator.classes

    # obtain actual predicition from probabilities
    y_pred =np.argmax(probabilities,axis=1)
    #print("predictions ",self.y_pred[:10])
    #print("True labels and predicted labels\n",np.c_[self.y_true[:10],self.y_pred[:10]])
    return test_generator, probabilities, classes



def predict_analyze(modelId, test_generator, probabilities):

    # save all predictions to file with filenames
    savepath =f"{outdir}/{modelId}_probabilities_{os.getpid()}.csv"
    print(f"[INFO] Saving prediction probabilities to {savepath}")

    filenames = np.array(test_generator.filenames)
    print(f"shape of filenames is {filenames.shape}")

    filenames = filenames.reshape(-1,1)
    stack = np.hstack([filenames, probabilities])
    #print("hstack of filenames,probabilities")
    #print(stack)


    classes = sorted(test_generator.class_indices.keys())
    y_true = test_generator.classes
    y_pred = np.argmax(probabilities,axis=1)
    header = "filenames, " + ", ".join(classes)

    np.savetxt(savepath,stack,delimiter=',',fmt='%s',header = header)

    # save all prediction losses to disk
    savepath =f"{outdir}/{modelId}_predictionlosses_{os.getpid()}.csv"
    print(f"[INFO] Saving filenames, true labels, prediction and losses to {savepath}")
    gt = y_true.reshape(-1,1)
    probs = probabilities
    #predicted_prob =    np.max(self.probabilities,axis=1).reshape(-1,1)
    # get the proability prediction for the true class
    pp = np.take_along_axis(probs,gt,1)
    loss = 1-pp
    stack = np.hstack([filenames, gt, loss ])
    header = "filenames, target, loss"
    np.savetxt(savepath,stack,delimiter=',',fmt='%s',header = header)


    # save wrongly predicted filenames and probabilities to disk
    savepath = f"{outdir}/{modelId}_wrong_predictions_{os.getpid()}.csv"
    print(f"[INFO] Saving wrong predictions to {savepath}")

    # obtain the wrongly predicted cases
    wrong_predicts = probabilities[y_true != y_pred]
    print("shape of wrong prediction",wrong_predicts.shape)
    wrong_predict_stack = np.hstack([filenames[y_true != y_pred],wrong_predicts])
    print("hstack of filenames,wrong_predicts")
    print(wrong_predict_stack)



    np.savetxt(savepath,wrong_predict_stack,delimiter=',',fmt='%s',header = header)

    # obtain the confusion matrix and display it
    cm = confusion_matrix(y_true, y_pred)
    #print('Confusion matrix:\n', cm)
    plt.figure(figsize=(12,8))
    plot_confusion_matrix(cm, classes)
    plt.savefig(os.path.join(outdir,'{}_confusion_matrix.png'.format(modelId)),bbox_inches='tight')

    #TODO: fix this portion
    #report = classification_report(self.y_true, self.y_pred, target_names=self.classes)
    # print the classification report
    #print(report)


if __name__ == "__main__":

    # define the constants

    NUM_CLASSES = 2
    CHANNELS = 3
    color_dict = {1:'grayscale',3:'rgb'}
    # define the path to the output directory used for storing plots,
    # classification reports, etc.
    OUTPUT_PATH = "output"
    COLOR_MODE = color_dict[CHANNELS]


    parser.add_argument('-modelid', help="id of saved model file")
    args = parser.parse_args()
    print(f"[INFO] Using the following default values:\n",args)


    # obtain tweakable parameters from command line
    TARGET_SIZE =args.targetsize
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batchsize)
    base_dir            = args.traindata
    test_dir            = os.path.join(base_dir, 'test')
    validation_dir      = os.path.join(base_dir, 'validation')
    STEP = int(args.step)
    RANDOM_STATE = args.randomstate
    OUTPUT_PATH = 'output'


    # obtain the process id to uniquely name outputs like saved model, training graphs etc.
    # pid = os.getpid()


    pid = args.modelid
    # create a separate output directory for each run
    outdir = os.path.join(OUTPUT_PATH,f"{pid}")
    # os.mkdir(outdir)

    test_generator, probabilities, classes = predict(pid, test_dir,TARGET_SIZE)
    predict_analyze(pid,test_generator, probabilities)
