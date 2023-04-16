from utilities import video_to_frames
import glob, os

class Driver():
    
    def __init__(self):
        pass

    def retrieve_frames_from_videos(self,video_folder,save_path):


        files = glob.glob(save_path+"/*")
        for f in files:
            os.remove(f)
        os.chdir(video_folder)
        for file in glob.glob("*.mp4"):
            video_to_frames(file,save_path)
            

d=Driver()
d.retrieve_frames_from_videos("/home/manthan/capstone/data/train","/home/manthan/capstone/data/images")

