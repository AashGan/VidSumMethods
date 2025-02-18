import cv2
import numpy as np
from sklearn.decomposition import PCA
import time
from skimage.feature import fisher_vector, ORB, learn_gmm
import h5py
import json
import os
import skimage

def popatov_feat_extract(video_path,dataset = 'tvsum'):
    '''A description of the video feature extraction used by Popatov et al on Category Specific video summarization. Which is described as SIFT feature extraction, PCA and Fisher model'''
    cap = cv2.VideoCapture(video_path)

    all_descriptors = []
    frame_count = 0
    sift = cv2.SIFT_create() 
    pca = PCA(n_components=64)
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 5 == 0:  
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if dataset == 'summe':
                frame = cv2.resize(frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            _,descriptors = sift.detectAndCompute(frame,None)
            if descriptors is not None and descriptors.shape[0]>64:
                all_descriptors.append(pca.fit_transform(descriptors))
        frame_count += 1
    end = time.time()
    cap.release()
    print(all_descriptors[0].shape)
    print(f'Time to read video detect sift and normalize: {end-start}')
    k = 128
    start = time.time()

    gmm = learn_gmm(all_descriptors, n_modes=k)
    end = time.time()

    print(f'Time to learn GMM  : {end-start}')
    
    def normalize(fisher_vector):
        fisher_vector = (fisher_vector-np.mean(fisher_vector,axis=0))/(np.std(fisher_vector,axis=0))  #Mentioned 
        v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
        return v / np.sqrt(np.dot(v, v))
    start = time.time()

    fisher_vectors_array = np.array([normalize(fisher_vector(descriptor,gmm)) for descriptor in all_descriptors])
    end = time.time()   
    print(f'Time to create fishers : {end-start}')


    return fisher_vectors_array


def run():
    save_name = 'DatasetFeatures/tvsum_summe/summe_features_fisher'
    video_dir_path = "Videos"
    with h5py.File(f'{save_name}.hdf5','w') as h5out:
        for i in range(25):
            video_path = os.path.join(video_dir_path,f'summe/video_{i+1}.mp4')
            features = popatov_feat_extract(video_path,'summe')
            h5out.create_dataset(f'video_{i+1}',data=features)
    save_name = 'DatasetFeatures/tvsum_summe/tvsum_features_fisher'
    video_dir_path = "Videos"
    with h5py.File(f'{save_name}.hdf5','w') as h5out:
        for i in range(50):
            video_path = os.path.join(video_dir_path,f'tvsum/video_{i+1}.mp4')
            features = popatov_feat_extract(video_path,'tvsum')
            h5out.create_dataset(f'video_{i+1}',data=features)

if __name__ == "__main__":
    run()