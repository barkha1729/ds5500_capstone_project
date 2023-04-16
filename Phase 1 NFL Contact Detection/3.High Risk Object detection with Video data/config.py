import torch
BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 224 # resize the image for training and transforms
NUM_EPOCHS = 5 # number of epochs to train for
NUM_WORKERS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '/kaggle/working/images'
# validation images and XML files directory
VALID_DIR = '/kaggle/working/images'
TRAINING_FILE_PATH="train_mini.csv"
VALID_FILE_PATH="valid.csv"

# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'person'
]
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = 'outputs'
