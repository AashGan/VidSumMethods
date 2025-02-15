from extract_features import run_feature_extractor_subset
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights,googlenet,GoogLeNet_Weights,densenet121, DenseNet121_Weights
import gc
import torchvision.transforms as transforms
import h5py
import os
import pickle
def run():
    dataset_path = 'C:\\Datasets\\video_shot_boundaries\\all_videos'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoshot_keys = pickle.load(open('gt_scenes_dict_baseline_v2.pickle','rb'))
    autoshot_keys = [f'{key}.mp4' for key in list(autoshot_keys.keys())]
    
    model = densenet121(weights =DenseNet121_Weights.IMAGENET1K_V1)
    preprocess = DenseNet121_Weights.IMAGENET1K_V1.transforms()
    submodel = nn.Sequential(*list(model.children())[:-1],nn.AdaptiveAvgPool2d(1)).to('cuda').eval()
    save_name = 'autoshot_feature_densenet'
    run_feature_extractor_subset(dataset_path,autoshot_keys,submodel,0,preprocess,save_name)

    del submodel
    gc.collect()
    model = googlenet(weights = GoogLeNet_Weights.IMAGENET1K_V1)

    preprocess =  GoogLeNet_Weights.IMAGENET1K_V1.transforms()
    submodel = nn.Sequential(*list(model.children())[:-2]).to(device).eval()
    save_name = 'autoshot_feature_googlenet'
    run_feature_extractor_subset(dataset_path,autoshot_keys,submodel,0,preprocess,save_name)
    del submodel
    gc.collect()
    torch.cuda.empty_cache()
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
    submodel = nn.Sequential(*list(model.children())[:-1])
    submodel.eval().to(device)
    save_name = 'autoshot_feature_resnet'
    run_feature_extractor_subset(dataset_path,autoshot_keys,submodel,0,preprocess,save_name)
    del submodel
    gc.collect()
    torch.cuda.empty_cache()   
    
    vitb16 =  torchvision.models.vit_l_16(weights = torchvision.models.ViT_L_16_Weights).eval().to('cuda')
    preprocess = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()
    def vit_feat_extract(img):
        feats = vitb16._process_input(img)
        batch_class_token = vitb16.class_token.expand(img.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        feats = vitb16.encoder(feats)
        feats = feats[:,1:].mean(1)
        return feats
    submodel = vit_feat_extract
    save_name = 'autoshot_feature_vit'
    run_feature_extractor_subset(dataset_path,autoshot_keys,submodel,0,preprocess,save_name)

if __name__ =="__main__":
    run()