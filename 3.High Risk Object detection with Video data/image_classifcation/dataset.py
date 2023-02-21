import cv2
import numpy as np
import os
import sys
import pandas as pd
import glob as glob
from torchvision.transforms import transforms
from xml.etree import ElementTree as et
from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE, TRAINING_FILE_PATH, VALID_FILE_PATH, HELMET_FP,HELMET_FP_valid
)
import torch

from torch.utils.data import Dataset, DataLoader
from utilities import collate_fn, get_train_transform, get_valid_transform
# the dataset class
class NFLDataset(Dataset):
    def __init__(self, image_dir_path,training_file, width, height, helmet_path, transforms=None):
        self.transforms = transforms
        self.image_dir_path = image_dir_path
        self.height = height
        self.width = width
    
        # self.classes = classes
        self.helmet_path=helmet_path
        
        # get all the image paths in sorted order
        self.helmet_df=pd.read_csv(self.helmet_path)
        self.df=pd.read_csv(training_file)
        self.all_images=[]


        # self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
       
    def transform_grayscale(self,img):
        transform = transforms.Compose([
                transforms.ToTensor(),
                    transforms.Grayscale()
                    
                                        ])

        img_gray = transform(img)

        return img_gray

     

    def process_each_view(self,image_path,roi):
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image_resized = cv2.resize(image, (self.width, self.height))
        # image_resized /= 255.0
        print(roi)
        image_resized = image[roi[1][0]:(roi[1][0]+roi[3]),roi[0][0]:(roi[0][0]+roi[2])]
        return image_resized
                
    def __getitem__(self, idx):
        # capture the image name and the full image path
        row = self.df.iloc[idx]
        frame=int(row["step"]*59.94*0.1+5*59.94)
        player1=row["nfl_player_id_1"]
        player2=row["nfl_player_id_2"]
        images=[]
        for view in ["Sideline","Endzone"]:
            relevant_row1=self.helmet_df.loc[(self.helmet_df.game_play==row["game_play"]) & (self.helmet_df.frame==int(frame)) & (self.helmet_df.nfl_player_id==int(player1)) & (self.helmet_df.view==view)]
            relevant_row2=self.helmet_df.loc[(self.helmet_df.game_play==row["game_play"]) & (self.helmet_df.frame==frame) & (self.helmet_df.nfl_player_id==player2) & (self.helmet_df.view==view)]
            x=min(relevant_row1["left"].values,relevant_row2["left"].values)-30
            if(x<0):
                x=np.array([0])
            
            roi=(x,min(relevant_row1["top"].values,relevant_row2["top"].values)+30,self.height,self.width)
            path=self.image_dir_path+"/"+row["game_play"]+"_"+view+".mp4_frame_"+str(frame)+".jpg"
            image=self.process_each_view(path,roi)
            img=self.transform_grayscale(image)
            images.append(img)
        # read the image
        
        input_return=torch.concatenate(images,dim=0)
        return input_return,torch.tensor(row["contact"])
    def __len__(self):
        return len(self.df)
# prepare the final datasets and data loaders
def create_train_dataset():
    train_dataset = NFLDataset(TRAIN_DIR,TRAINING_FILE_PATH, RESIZE_TO, RESIZE_TO,HELMET_FP, get_train_transform())
    return train_dataset
def create_valid_dataset():
    valid_dataset = NFLDataset(VALID_DIR,VALID_FILE_PATH, RESIZE_TO, RESIZE_TO, HELMET_FP_valid, get_valid_transform())
    return valid_dataset
def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader

if __name__ == '__main__':
 
    train_dataset=create_train_dataset()
    valid_dataset=create_valid_dataset()
    train_loader=create_train_loader(train_dataset)
    valid_loader=create_valid_loader(valid_dataset)
    for batch_idx, (data, targets) in enumerate(train_loader):
        break
    
