import torch
BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 200 # resize the image for training and transforms
NUM_EPOCHS = 1 # number of epochs to train for
NUM_WORKERS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '/home/balaji/manthan/nfl/data/images'
# validation images and XML files directory
VALID_DIR = '/home/balaji/manthan/nfl/data/images'
TRAINING_FILE_PATH="/home/balaji/manthan/nfl/data/train_labels.csv"
VALID_FILE_PATH="/home/balaji/manthan/nfl/data/train_labels.csv"

# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'person'
]
HELMET_FP="/home/balaji/manthan/nfl/data/train_baseline_helmets.csv"
HELMET_FP_valid="/home/balaji/manthan/nfl/data/train_baseline_helmets.csv"
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = 'outputs'