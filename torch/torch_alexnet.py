"""
Script for training an AlexNet model with PyTorch
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader 
from torchvision import datasets, transforms
import numpy as np
import time
import os
import math
import wandb

from argscommon import parser, setup_logger
parser.add_argument('-images_dir', '--images_dir', required=True, help="Directory containing images for training")
parser.add_argument('-test_images', '--test_images', help="Directory containing images for testing")
parser.add_argument('-retrain', '--retrain', action="store_true", help="Switch for loading weights from previous training")
parser.add_argument('-augment', '--augment', action="store_true", help="Switch for doing augmnetation on training data")
parser.add_argument('-saved_model', '--saved_model', default="trained_model.pth", help="Path of saved model")
parser.add_argument('-batch_size', '--batch_size', type=int, default=32)
parser.add_argument('-epochs', '--epochs', type=int, default=10)
parser.add_argument('-lr', '--lr', type=float, default=0.001)


project_name = "Ring_Train_Torch"

def modify_alexnet(model):
    """
    This function modifies the base architecture of AlexNet to conform as far as
    posssible with the architecture that used in tensorflow
    """

    model.features[0].out_channels = 96
    model.features[0].kernel_size = (5,5)
    #model.features[0].padding = 'same'
    model.features[0].stride = (2,2)

    model.features[3].in_channels = 96
    model.features[3].out_channels = 256
    model.features[3].kernel_size = (5,5)
    model.features[3].padding = 'same'
    #model.features[3].stride = (2,2)

    model.features[6].in_channels = 256
    model.features[6].out_channels = 384
    model.features[6].kernel_size = (3,3)
    model.features[6].padding = 'same'

    model.features[8].in_channels = 384
    model.features[8].out_channels = 384
    model.features[8].padding = 'same'

    model.features[10].in_channels = 384
    model.features[10].out_channels = 256
    model.features[10].padding = 'same'

    #TODO: find out why the following code doesn't work
    # model.classifier[6].out_features = 2
    model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
    model = nn.Sequential(model, nn.Sigmoid())

    return model

def random_choice(x, size, seed, axis=0, unique=True):
    torch.manual_seed(seed)
    indices = torch.randperm(x.size(axis))[:size]
    sample = torch.index_select(x, axis, indices)

    return sample, indices

def random_int_rot_img(inputs, seed):
    angles = [0, 90, 180, 270]
    angle = random_choice(torch.tensor(angles), 1, seed=seed)[0][0].item()
    inputs = TF.rotate(inputs, angle)

    return inputs

def rescale(image, label):
    # Convert PIL image to numpy array
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image)

    return image, label

def apply_augmentations(image, label=None):
        augmented_image, _ = augment_custom(image, None, augmentation_types=['rotation','flip'], seed=42)
        return augmented_image, label

class CustomDataset(Dataset):
    def __init__(self, data_dir, augmentation_types=None, seed=None):
        self.data = datasets.ImageFolder(root=data_dir)
        self.augmentation_types = augmentation_types
        self.seed = seed
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]) 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.transforms(image) 
        if self.augmentation_types:
            image, _ = self.apply_augmentations(image, label)
        return image, label

    def apply_augmentations(self, image, label):
        if 'rotation' in self.augmentation_types:
            image = random_int_rot_img(image, seed=self.seed)
        if 'flip' in self.augmentation_types:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
        if 'brightness' in self.augmentation_types:
            image = TF.adjust_brightness(image, torch.rand(1).item() * 0.4 - 0.2)
        if 'contrast' in self.augmentation_types:
            image = TF.adjust_contrast(image, torch.rand(1).item() * 0.3 + 0.2, torch.rand(1).item() * 0.3 + 0.7)

        return image, label    

def validate_model(model, val_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(val_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            labels = labels.unsqueeze(1)
            labels = labels.float()
            val_loss += loss_func(outputs, labels)*labels.size(0)

            # Compute accuracy and accumulate
            pred_score, _ = torch.max(outputs.data, 1)
            threshold = 0.5
            predicted = (pred_score > threshold).to(torch.int64)
            labels = labels.squeeze().to(torch.int64)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if i==batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.data)
    return val_loss / len(val_dl.dataset), correct / len(val_dl.dataset)

def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    table = wandb.Table(columns=["image", "pred", "target", "score"])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(np.moveaxis(img.numpy()*255,0,2)), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)

class SaveBestModel:
    def __init__(self, monitor='val_loss', mode='min'):
        self.monitor = monitor
        self.mode = mode
        if mode == 'min':
            self.best_value = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best_value = float('-inf')
            self.monitor_op = lambda x, y: x > y
        
    def __call__(self, val_metric, model, filepath):
        if self.monitor_op(val_metric, self.best_value):
            print(f"Validation {self.monitor}: {val_metric} improved from {self.best_value} to {val_metric}. Saving model...")
            self.best_value = val_metric
            torch.save(model.state_dict(), filepath)
        else:
            print(f"Validation {self.monitor}: {val_metric} did not improve from {self.best_value}.")

def create_output_dir():
    wandb_dir = wandb.run.name
    output_dir = os.path.join("output", wandb_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def validate_args(args):
    """
    Load model with weights from file only if retrain parameter is passed and the weights file exists on disk.
    Raise appropriate errors if one of the conditions are not met.
    If saved_model path is specified but retrain is not passed do not proceed with fresh training in order not to overwrite existing file but raise a ValueError
    """
    logger.info(f"Saved model path:{args.saved_model}")
    if os.path.exists(args.saved_model):
        if args.retrain:
            logger.warn("Loading weights from previous training")
            model.load_state_dict(torch.load(args.saved_model))
        else:
            raise ValueError("Found existing file on disk. The --retrain flag is required to resume training from an existing model.")
    else:
        if args.retrain:
            raise FileNotFoundError("The specified saved model path does not exist.")
        else:
            logger.warn("Starting a fresh training")

def train_val_dataloaders(train_data):

    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size

    logger.info(f"Train size: {train_size}, Val size:{val_size}")

    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader

if __name__=="__main__":

    args = parser.parse_args()

    args_dict = vars(args)
    logger = setup_logger(args.log_level)

    wandb.init(
        project= project_name,
        config= args_dict
            )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    alexnet = models.alexnet()
    logger.debug(f"Original AlexNet architecture:\n {alexnet}")
    model = modify_alexnet(alexnet)

    logger.debug(f"Model Architecture: \n {model}")

    output_dir = create_output_dir()
    logger.info(f"Outputs are saved to {output_dir}")

    validate_args(args)

    model.to(device)

    if args.augment:
        train_data = CustomDataset(data_dir=args.images_dir, augmentation_types=["rotation", "flip"], seed=42)
    else:
        train_data = CustomDataset(data_dir=args.images_dir)
    
    train_loader, val_loader = train_val_dataloaders(train_data)

    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    logger.info(f"Steps per epoch:{n_steps_per_epoch}")

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = args.epochs

    save_best_model_callback = SaveBestModel(monitor='val_loss', mode='min')

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}...")
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # add following two lines to prevent errors 
            # https://stackoverflow.com/questions/57798033/valueerror-target-size-torch-size16-must-be-the-same-as-input-size-torch
            labels = labels.unsqueeze(1)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            metrics = {"train/train_loss": loss,
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch
                       }

            if step + 1 < n_steps_per_epoch:
                # Log train metrics to wandb 
                wandb.log(metrics)

        val_loss, accuracy = validate_model(model, val_loader, criterion, log_images=(epoch==(args.epochs-1)))

        model_save_path = os.path.join(output_dir,args.saved_model)
        logger.info(f"Saving best model to {model_save_path}")
        save_best_model_callback(val_loss, model, model_save_path)

        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy}

        wandb.log({**metrics, **val_metrics})

        epoch_loss = running_loss / len(train_data)
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} seconds")
