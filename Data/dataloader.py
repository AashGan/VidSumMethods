import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
import os
import torch.nn.functional as F
#TODO Add the transforms in a different place 

datasets_dict = {'googlenet':'Data\\original','open_clip':'Data\\open_clip','resnet':'Data\\resnet50','3dresnet':'Data\\3dresnet','videoMAE':'Data\\videoMAE'}
splits_dir = 'Splits'
class VideoData(Dataset):
    def __init__(self,mode,splits_json,split_index = 0,transforms = None,**kwargs):
        super().__init__()
        self.mode = mode
        self.dataset_dict = {}
        self.feature_reps = kwargs.get('feature_extractor','googlenet')
        splits_json = os.path.join(splits_dir,splits_json)

        with open(splits_json) as f:
            data = json.loads(f.read())
            self.all_datapoints = data[split_index][mode +'_keys']
        self._create_data_dict(datasets_dict[self.feature_reps])
        if transforms:
            assert type(transforms) == list, "Ensure the transformations are given as a list"
            self.transforms = transforms
        else:
            self.transforms = False

    def __len__(self):
        return len(self.all_datapoints)
    
    def _create_data_dict(self,main_data_path):
        print(main_data_path) # Main datapath should be a list the data points 
        list_of_datasets = os.listdir(main_data_path)
        for dataset in list_of_datasets:
            print(os.path.join(main_data_path,dataset))
            hdf = h5py.File(os.path.join(main_data_path,dataset),'r')
            key = dataset.split('.')[0].split('_')[1]  # This should return the dataset
            print(key)
            self.dataset_dict[key] = hdf
        
    def __getitem__(self,index):
        dataset,video_index  = self.all_datapoints[index].split('/')
        
        features = self.dataset_dict[dataset][video_index]['features'][...]
        shot_bounds = self.dataset_dict[dataset][video_index]['downsampled_shot_boundaries'][...] if 'downsampled_shot_boundaries' in self.dataset_dict[dataset][video_index].keys() else None
        gtscore = self.dataset_dict[dataset][video_index]['gtscore'][...]
        features = torch.from_numpy(features)
        gtscore = torch.from_numpy(gtscore)
        
        if self.transforms and self.mode=='train':
            for transform in self.transforms:
                if shot_bounds is not None:
                    features,gtscore = transform.shuffle(features,gtscore,shot_bounds = shot_bounds)
                else:
                    features,gtscore = transform.shuffle(features,gtscore)
        
        if self.mode =='test':
            
            return features,video_index
        
        return features, gtscore

            