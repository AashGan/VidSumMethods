# Reproducibility of the KTS algorithm 

This repository contains the code for the paper titled "Reproducibility study for shot boundary detection within the context of video summarization"
## Hardware requirements

To successfully run all the experiments for this paper. A minimum of 16 GB of RAM and 16GB of GPU VRAM is needed. Otherwise, there are risks of crashing and not finishing.
We validated these experiments on the following GPU hardware:
RTX Quadro 6000 24GB

## Setup
We first provide the dataset links alongside where the dataset should be extracted. All paths specified are within the root of the repository.

### Datasets and Videos
#### TVSum and SumMe annotated dataset
Extract the content downloaded from this [link](https://drive.google.com/drive/folders/1Y2pje5lRhwPolTmY4q3wsORgno92uPUi?usp=sharing) to the path "Data/original".
#### Videos from TVSum and SumMe
Extract the video content from this [link](https://drive.google.com/file/d/1z8u1VoXEUvPIWWZpX-pd8TJGrbydxOJM/view?usp=sharing) to the path "Videos/summe" and "Videos/tvsum" for each dataset.
#### Annotations for AutoShot dataset
The annotations for the SHOT dataset is provided for in this repository in the file 'gt_scenes_dict_baseline_v2.pickle'. The original source can be found in the original [codebase](https://github.com/wentaozhu/AutoShot) provided by the authors
#### Videos for the AutoShot dataset
The videos of the test split of the autoshot dataset can be found in this [link](https://drive.google.com/file/d/1LmcYisX6hiX2MCIapC2ClEe3FztSYPFH/view?usp=sharing). Extract to the path "Videos/autoshot". The complete dataset can be found in the original [codebase](https://github.com/wentaozhu/AutoShot) provided by the authors

#### Features for both datasets
The features extracted from our system can be found [here](https://drive.google.com/drive/folders/1hF5Ob9tIpzr47ZPj8FGnzHmmWinU3y4d?usp=sharing). Ensure that the file structure is maintained as follows after extraction:
``` 
DataFeatures
        |autoshot
        |tvsum_summe
```
#### Paper reported Experimental Results
The original results of the paper can be found in the repository
## Experimental Setup for Reproduction
We describe how our experimental set up can be reproduced in three parts. We strongly recommend a separate virutal environment where-ever specified. 

### Feature Extraction 
The feature extraction is done in two separate environments and python versions due to conflicts. 

#### Fisher Features
This feature extractor used Python 3.10, with a virtual environment setup with requirements_Fisher.txt. 
To run the Fisher Feature extractor for the TVSum and SumMe dataset, use the following command in your virtual environment

````bash
python run_fisher_feature_extractor.py
````

#### Deep Learning Feature extractors
This feature extractor used Python 3.7.9, with a virtual environment setup with requirements_DL.txt. 
To run the Deep Learning Feature extractor for the TVSum and SumMe dataset, use the following command in your virtual environment

````bash
python run_feature_extractor_tvsum_summe.py
````
To run the Deep Learning Feature extractor for the Autoshot dataset, use the following command in your virtual environment

````bash
python extract_features_autoshot.py

````
### Shot boundary detection
Prior to running each code, setup the virtual environment using the requirments_expts.txt file for Python 3.11.2. 

It is an important point to note:
``cupy`` dependency must be adjusted according to the version of CUDA on your system. Adjust the requirements_expts accordingly 
``pip install cupy-cuda11x `` for cuda versions 11.2 to 11.8
``pip install cupy-cuda12x ` for cuda versions 12 onwards.

To run the shot boundary detectors, first take the features we provide in the link or use the feature extraction code listed above
Note: CUPY by default uses the first GPU that it sees. To set a specific GPU for each experiments Uncomment ```cnp.cuda.runtime.setDevice()``` and set it to the appropriate GPU that you intend to use based on how it is arranged in your system for EG ```cnp.cuda.runtime.setDevice(1)``` will use the second GPU on your system
#### KTS
##### TVSum and SumMe
Run the following to obtain the TVSum and SumMe shot boundaries:
```
python run_kts_tvsum_summe_cupy.py
```
##### Autoshot
Run the following to obtain the autoshot shot boundaries:
```
python run_kts_autoshot.py
```
#### PELT 
##### Autoshot
Run the following to obtain the autoshot shot boundaries:
```
python run_pelt_on_autoshot.py
```
### Results
Please run the notebook ExperimentalResults.ipynb to obtain all the obtained results from the experiments.

## Miscellanous Code

We've also provided the code for different aspects of the study

### RunTime differences

We demonstrate the run time differences between our implementation of the KTS algoritm compared to that of one available public implementation of the KTS algorithm. This can be seen in PerformanceComparisons.ipynb 

### Summary generation

We provide a notebook to generate summaries in notebook SummaryGeneration.ipynb 

### Validate Fisher Features

We also provide a notebook to validate the process followed for the fisher based feature extraction procedure. 


# Other important details

Some of the videos in the autoshot test split were missing from the video as pointed in this [link](https://github.com/wentaozhu/AutoShot/issues/6) as the original authors did not provide it 

