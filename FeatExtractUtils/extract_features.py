from PIL import Image
import cv2
import numpy as np
import torch
#import torchvision
from tqdm import tqdm
import h5py
import os
import torch.nn as nn

class THWC_to_CTHW(torch.nn.Module):
    """ Class which converts channels to the right order. Specifically for the deep neural networks 
    """
    def forward(self, data):
        # Do some transformations
        return data.permute(3, 0, 1, 2)
class PreProcessorVidSum(object):
    def __init__(self,feature_extractor,target_downsample:int=2,shot_aware:bool= True):
        """ Class accepts a function as a feature extractor 
        Input: 
        feature_extractor: nn.module or function : Receives image input to output features
        target_downsample: the rate at which the signal needs to be downsampled to 
        shot_aware: Not in use 
        """
        self.target_downsample = target_downsample
        self.feature_extractor = feature_extractor # TODO add support for GPU
        self.shot_aware = shot_aware
    def run(self,video_path:str,shot_boundaries = []):
        ''' Input:
        video_path: the video path 
          This is using the shot boundaries from the h5 datasets to frames to pick the selected, so it returns all the frames features and the selected ones
          returns: list of features 
        '''
        shot_boundaries = np.array(shot_boundaries).astype(int)
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_rate)
        print(total_frames)
        downsample_target = frame_rate//self.target_downsample if self.target_downsample!=0 else 1
        picked_frames = np.arange(0,total_frames,downsample_target )
        selected_frames = np.union1d(shot_boundaries,picked_frames)
        print(len(selected_frames))
        if selected_frames[-1]>total_frames-1: selected_frames[-1]=total_frames-1
        print(selected_frames[-1],selected_frames[-2])
        all_frames = []
        for sub_frame in tqdm(selected_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,sub_frame)
            ret,frame = cap.read()
            if not ret:
                print(f"Error reading frame at index {sub_frame}")
                continue
            all_frames.append(self.feature_extractor.run(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).numpy())
        cap.release()
        return all_frames, selected_frames

class FeatureExtractor():
    def __init__(self,model,transforms):
        """ Wrapper function for neural network based feature extractor"""
        self.model = model
        self.transforms = transforms # Transforms should act like one function, otherwise, one should do this outside and pass identity through this transform

    def run(self,input):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # The model has to be in eval mode and on the GPU/CPU set outside.
        with torch.no_grad():
            return self.model(self.transforms(input).unsqueeze(0).to(device)).squeeze().to('cpu')
        




def run_feature_extractor(video_paths:str,model:torch.nn.Module,downsample_rate:int,preprocess,save_name:str):
    "Wrapper function"
    feature_extractor = FeatureExtractor(model,preprocess)
    preprocesser_sum = PreProcessorVidSum(feature_extractor,target_downsample=downsample_rate)

    with h5py.File(f'{save_name}.hdf5','w') as h5out: 
        for video in os.listdir(video_paths):
            video_key = video.split('.')[0]
            print(video_key)
            video_path = os.path.join(video_paths,video)
            features,_ = preprocesser_sum.run(video_path)
            h5out.create_dataset(f'{video_key}',data = features)

    print(f'Features saved at : {save_name}')
