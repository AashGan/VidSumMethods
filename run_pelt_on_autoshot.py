import numpy as np
import numpy.linalg as LA
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters
import h5py
import json
import os 
feature_list = ['resnet','googlenet','densenet','vit']
beta_vals = [1,2,5,10,25,50]
algo_c = rpt.KernelCPD(kernel="linear", min_size=2)
def run_pelt(features,change_point_function,beta=100):
    length = len(features)
    result = change_point_function.fit_predict(np.asarray(features),beta)
    result = np.insert(result,0,0)
    change_points = np.array([[result[i],result[i+1]-1] for i in range(len(result)-1)])
    change_points[-1] = [result[-2],length-1]
    return change_points


def run_experiments(result_path='Results'):
    dataset = 'autoshot'
    if not os.path.exists(f'{result_path}/pelt/{dataset}'):
        os.makedirs(f'{result_path}/pelt/{dataset}')
    result_path = f'{result_path}/pelt/{dataset}'
    for feat in feature_list:   
        features = h5py.File(f'DatasetFeatures/autoshot/{dataset}_features_{feat}.hdf5') 
        for beta in beta_vals:
            results_dict = {}
            for key in list(features.keys()):
                feature = np.array([feat/LA.norm(feat) for feat in features[key]])
                try:
                    change_points = run_pelt(feature,algo_c,beta)
                except BadSegmentationParameters:
                    print(f'Video: {key} cannot for dataset summe be segmented with this beta: {beta}' )
                    change_points = np.array([0,0])
                results_dict[key] = change_points.tolist()
            json.dump(results_dict,open(f'{result_path}/autoshot_shot_boundaries_{feat}_beta_{beta}.json','w'),indent=4)

    

if __name__ == "__main__":
    run_experiments()


