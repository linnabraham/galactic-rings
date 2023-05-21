import os
from alexnet_utils.params import parser
from sklearn.model_selection import train_test_split
import shutil
import subprocess



# Function to take a directory containing two subdirectories with images and
# create the training data by splitting into three fractions and optionally augmenting the train samples

# create the directories to hold the data first
def make_dirs(BASE_DIR,label):
    train_dir = os.path.join(BASE_DIR,"train/",label)
    val_dir   = os.path.join(BASE_DIR,"validation/",label)
    test_dir  = os.path.join(BASE_DIR,"test/",label)

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)

    print("train dir is ",train_dir)
    print("valdation dir is ",val_dir)
    print("test dir is ",test_dir)

    return train_dir, val_dir, test_dir

# function to iteratively augment images in the given training directory using a given augmentation script
# and save it into a given destination folder
#
# this function first renames the given training directory to something like RingsTemp
# then takes each image from RingsTemp, augments it and saves into the now empty Rings folder

def augment_on_train(train_dir, train_samples, AUGSCRIPT, STEP, aug=True):
        if aug:
            tmpdir = train_dir+"Temp/"
            shutil.move(train_dir,tmpdir)
            os.makedirs(train_dir)
            print(f"generating augmented samples and saving to {train_dir}")
            for sample in train_samples:
                item = os.path.join(tmpdir,sample)
                # print("image is :",item)
                cmd = f"python {AUGSCRIPT} {item} {train_dir} {STEP}"
                # print("Executing the following command \n",cmd)
            # subprocess.run([sys.executable, augscript, tmpdir, train_dir, str(config.STEP)],shell=True,check=True )
                subprocess.run([cmd],shell=True,check=True)
            shutil.rmtree(tmpdir)



# Given the original images path and a class label, this function reads all images in the folder named with the class label
# inside the images path, splits it into three sets and creates a separate directory for the class label inside the train
# validation and test directories and copies the images there.

def train_val_test_split(images_path, label, train_frac, val_frac, train_dir, val_dir, test_dir, random_state):
   # list all images for each label
    allImages = os.listdir(os.path.join(images_path,label))
    total = len(allImages)
    print("No. of files in subdir is",total)

    # Use the train_test_split function twice for creating three splits
    train_samples, rest_samples = train_test_split(allImages,train_size=train_frac,shuffle=True,random_state=random_state)


    val_samples, test_samples = train_test_split(rest_samples,train_size=(val_frac/(1-train_frac)),shuffle=True,random_state=random_state)

    return train_samples, val_samples, test_samples

def make_rings(AUGSCRIPT, STEP, aug, traindata_path, images_path,  train_frac, val_frac, random_state, label='Rings'):
    print("Making Rings....")
    train_dir, val_dir, test_dir = make_dirs(traindata_path,label)
    
    train_samples, val_samples, test_samples = train_val_test_split(images_path, label, train_frac, val_frac, train_dir, val_dir, test_dir, random_state)

    print(len(train_samples), len(val_samples), len(test_samples))

    print("Copying images to train folder ...")
    for img in train_samples:
        shutil.copy(os.path.join(images_path,label,img),train_dir)

    print("Copying images to validation folder ...")
    for img in val_samples:
        shutil.copy(os.path.join(images_path,label,img),val_dir)

    print("Copying images to test folder ... ")
    for img in test_samples:
        shutil.copy(os.path.join(images_path,label,img),test_dir)

    # do augmentation only on the images in the train folder using a given augmentation script
    augment_on_train(train_dir, train_samples, AUGSCRIPT, STEP, aug)



    return len(train_samples), len(val_samples), len(test_samples)

def make_nonrings(ntest, traindata_path, images_path,  train_frac, val_frac, random_state, label='NonRings'):
    print("Making NonRings....")
    train_dir, val_dir, test_dir = make_dirs(traindata_path,label)

    allImages = os.listdir(os.path.join(images_path,label))
    total = len(allImages)
    print("No. of files in subdir is",total)

    nonring_testfrac = ntest/total
    rest_samples, test_samples = train_test_split(allImages,train_size=1-nonring_testfrac,shuffle=True,random_state=random_state)
    train_samples, val_samples = train_test_split(rest_samples,train_size=train_frac,shuffle=True,random_state=random_state)
    print(len(train_samples),len(val_samples),len(test_samples))
    #print(len(train_samples)/len(rest_samples), len(val_samples)/len(rest_samples))


    print("Copying images to train folder ...")
    for img in train_samples:
        shutil.copy(os.path.join(images_path,label,img),train_dir)

    print("Copying images to validation folder ...")
    for img in val_samples:
        shutil.copy(os.path.join(images_path,label,img),val_dir)

    print("Copying images to test folder ... ")
    for img in test_samples:
        shutil.copy(os.path.join(images_path,label,img),test_dir)

    return train_samples, val_samples, test_samples

if __name__=="__main__":
    parser.add_argument('-noaugment', action="store_false", help="switch to augment images for training")
    parser.add_argument('-augscript', default='../augment_sample.py', help="path of augmentation script")
    parser.add_argument('-step', default=90, help="step size for rotating images")

    args = parser.parse_args()
    print("Using the following parameters", args)

    images_path = args.images
    traindata_path = args.traindata
    train_frac = args.trainfrac
    val_frac = args.valfrac
    random_state = args.randomstate
    aug = args.noaugment # noaugment has default value of True
    AUGSCRIPT = args.augscript
    STEP = int(args.step)

    ntrain, nval, ntest = make_rings(AUGSCRIPT, STEP, aug, traindata_path, images_path,  train_frac, val_frac, random_state, label='Rings')

    make_nonrings(ntest, traindata_path, images_path,  train_frac, val_frac, random_state, label='NonRings' )



