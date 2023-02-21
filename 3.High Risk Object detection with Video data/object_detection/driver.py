from utilities import video_to_frames
import glob, os
import pandas as pd
import argparse
class Driver():
    
    def __init__(self,train_file_path="/home/balaji/manthan/nfl/data/train_baseline_helmets.csv"):
        self.df=pd.read_csv(train_file_path)

    def retrieve_frames_from_videos(self,video_folder,save_path):


        files = glob.glob(save_path+"/*")
        for f in files:
            os.remove(f)
        os.chdir(video_folder)
        for file in glob.glob("*.mp4"):
            video_to_frames(file,save_path)
    def get_relevant_csv(self,game_key_list,csv_file_path="train.csv"):
         self.df[self.df['column_name'].isin(game_key_list)].to_csv(csv_file_path,)

            
if __name__=="__main__":
    
    d=Driver()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gk_list', type=str)
    args = parser.parse_args()

    if(args.gk_list):
        d.get_relevant_csv(args.gk_list,"train.csv")

    

