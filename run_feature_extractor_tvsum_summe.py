from FeatExtractUtils.extract_features import run_feature_extractor
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights,googlenet,GoogLeNet_Weights,densenet121, DenseNet121_Weights
import gc
import torchvision.transforms as transforms
import h5py
import os
import os
def run():
    dataset_path_tvsum = 'Videos/tvsum'
    dataset_path_summe = 'Videos/summe'
    save_path = 'DatasetFeature/tvsum_summe'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = googlenet(weights = GoogLeNet_Weights.IMAGENET1K_V1)
    
    preprocess =  GoogLeNet_Weights.IMAGENET1K_V1.transforms()
    submodel = nn.Sequential(*list(model.children())[:-2]).to(device).eval()
    save_name = 'tvsum_features_googlenet'

    run_feature_extractor(dataset_path_tvsum,submodel,0,preprocess,os.path.join(save_path,save_name))
    save_name = 'summe_features_googlenet'
    run_feature_extractor(dataset_path_summe,submodel,0,preprocess,os.path.join(save_path,save_name))
    del submodel
    gc.collect()
    torch.cuda.empty_cache()
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
    submodel = nn.Sequential(*list(model.children())[:-1])
    submodel.eval().to(device)
    save_name = 'tvsum_features_resnet'
    run_feature_extractor(dataset_path_tvsum,submodel,0,preprocess,os.path.join(save_path,save_name))
    save_name = 'summe_features_resnet'
    run_feature_extractor(dataset_path_summe,submodel,0,preprocess,os.path.join(save_path,save_name))
    del submodel
    gc.collect()
    torch.cuda.empty_cache()   
    model = densenet121(weights =DenseNet121_Weights.IMAGENET1K_V1)
    preprocess = DenseNet121_Weights.IMAGENET1K_V1.transforms()
    submodel = nn.Sequential(*list(model.children())[:-1],nn.AdaptiveAvgPool2d(1)).to(device).eval()
    save_name = 'tvsum_features_densenet'
    run_feature_extractor(dataset_path_tvsum,submodel,0,preprocess,os.path.join(save_path,save_name))
    save_name = 'summe_features_densenet'
    run_feature_extractor(dataset_path_summe,submodel,0,preprocess,os.path.join(save_path,save_name))
    del submodel
    gc.collect()
    torch.cuda.empty_cache()
    vitb16 =  torchvision.models.vit_l_16(weights = torchvision.models.ViT_L_16_Weights).eval().to(device)
    preprocess = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()

    def vit_feat_extract(img):
        feats = vitb16._process_input(img)
        batch_class_token = vitb16.class_token.expand(img.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        feats = vitb16.encoder(feats)
        feats = feats[:,1:].mean(1)
        return feats
    submodel = vit_feat_extract
    save_name = 'tvsum_features_vit'
    run_feature_extractor(dataset_path_tvsum,submodel,0,preprocess,os.path.join(save_path,save_name))
    save_name = 'summe_features_vit'
    run_feature_extractor(dataset_path_summe,submodel,0,preprocess,os.path.join(save_path,save_name))



if __name__ =="__main__":
    run()