import cv2
import os
import pandas as pd
import argparse
import numpy as np
sample_rate=60

# This Method 
class VideoReader():
    def __init__(self,df_path,sample_rate=60):
        self.df=pd.read_csv(df_path)
        self.sample_rate=sample_rate
    def draw_bbox(self,org_image, ann):
    
        image = np.copy(org_image)
        
        left,width,top,height = ann    
        print(left,width,top,height) 
        image = cv2.rectangle(image, (left, top), (left+width,top+height), (255, 0, 0), 1)

        return image
    def video_reader(self,video_path): 
        base_path = os.path.basename(os.path.normpath(video_path))
    
        cap = cv2.VideoCapture(video_path)
        success, img = cap.read()
        fno = 0
        base_path=base_path.split("_")
        #getting the params
        game_key=base_path[0]
        play_id=base_path[1]
        zone=base_path[2][:-4]
        while success:
            fno+=1
            df=self.get_bounding_box(game_key,play_id,zone,fno)
            # print(df)
            if(len(df)!=0):

                # print(fno,df)
                for idx,row in df.iterrows():
                    img = self.draw_bbox(img,row[["left","width","top","height"]])
                # img = cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 3)
            if(fno%10==0 ):
                cv2.imshow('img ', img)
                cv2.waitKey(-1)
                # print(fno)
            # read next frame
            success, img = cap.read()

    def get_bounding_box(self,game_key,play_id,zone,frame):
        return self.df.loc[(self.df.game_key == int(game_key)) & (self.df.play_id == int(play_id)) & (self.df.view == zone) &(self.df.frame==frame)]


parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    vr=VideoReader("../../data/train_baseline_helmets.csv")
    vr.video_reader(args.video_path)






    
