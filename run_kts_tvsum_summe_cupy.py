import numpy as np
from numpy import linalg as LA
import h5py
import os 
import json
import cupy as cnp
from Utils import kts_cupy
feature_list = ['googlenet','resnet','densenet','vit','fisher']
c_param = [0.8,1,2.5,5,7.5]
def run_experiments(dataset,result_path='Results'):
    if not os.path.exists(f'{result_path}/kts/{dataset}'):
        os.makedirs(f'{result_path}/kts/{dataset}')
    result_path = f'{result_path}/kts/{dataset}'
    for feat in feature_list:
        features = h5py.File(f'DatasetFeatures/tvsum_summe/{dataset}_features_{feat}.hdf5')
        for c in c_param:
            results_dict = {}
            for key in list(features.keys()):
                feature = np.array([feat/LA.norm(feat) for feat in features[key]])
                if feat == 'fisher':
                    cps,_,_ = kts_cupy(len(features[key]),feature,vmax = c,frame_skip=5)
                else:
                    cps,_,_ = kts_cupy(len(features[key]),feature,vmax = c,frame_skip=1)
                results_dict[key] = cps.tolist()
            json.dump(results_dict,open(f'{result_path}/Shot_Boundaries_{dataset}_{feat}_{c}.json','w'),indent = 4)

if __name__ == "__main__":
    #cnp.cuda.runtime.setDevice(1)
    run_experiments('summe')
    run_experiments('tvsum')
