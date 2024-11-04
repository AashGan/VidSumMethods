# Shots in the dark

This repository contains the code for the paper "An investigation into the inconsistency of shot boundaries and evaluation protocols within video summarization".

# Setup


## Datasets and Videos
### Dataset
Extract the content downloaded from this [link](https://drive.google.com/file/d/1GmEHfITTp_bDJ3l_mC9KnOsxD-onCrpn/view?usp=sharing) to the path "Data\googlenet"

## Videos
Extract the video content from this [link](https://drive.google.com/file/d/1z8u1VoXEUvPIWWZpX-pd8TJGrbydxOJM/view?usp=sharing) to the path "Videos" 


# Experimental Setup
For each of the experiment, two separate versions of python and two different virutal enviroments were used. 

## Shot Boundary Consistency(Fisher) 

This uses Python 3.10, for all the dependencies to be installed correctly set up a virtual environment with a version 3.10 Interpreter. Set this up using the following command

```bash
pip install -r requirements_2.txt
```
This was done as the Fisher feature extractor used was provided by the SkImage Library.

To run these experiments, follow the instructions given in the following notebooks:

1. Running Fischer Extractor SumMe
2. Runing Fischer Extractor TVsum


## Shot Boundary Consistency(CNNs) 
This uses Python 3.7.9, for all the dependencies to be installed correctly set up a virtual environment with a version 3.7.9 Interpreter. Set this up using the following command
```bash
pip install -r requirements_1.txt
```
This allows the following notebooks to be run: 

1. Shot Boundary Consistency SumMe
2. Shot Boundary Consistency TVsum

## Post-Processing Evaluation Experiment

This uses Python 3.7.9, for all the dependencies to be installed correctly set up a virtual environment with a version 3.7.9 Interpreter. Set this up using the following command. If the setup step has been done for the shot boundary(CNN) experiment, then this step can be skipped as the environment is shared between both. 

```bash
pip install -r requirements_1.txt
```

To run these experiments, follow the instructions given in the following notebooks:

1. Evaluation Divergence Video Summarization
2. RandomSumMeScoring


# Exact paper reported results

The results obtained by re-running the notebooks may vary slightly from the original paper due to differences in the CUDA versions and pytorch version differences. To obtain the results reported from the paper. Please run the notebooks below with the weights/extracted features from our experiments

Replication-Reported-Results.ipynb ( Requirements 1 environment)
Replication-Reported-Results-Fisher.ipynb (Requirements 2 environment)

The existing weights, extracted features and predicted shot boundaries can be found [here](https://drive.google.com/file/d/19INY4tJTCjlE9P1oacx05-tm80JgWzIA/view?usp=drive_link) (shot boundaries) and [here](https://drive.google.com/file/d/14Uliuz_jsEMxhce699X2xY2yAQgRx9wZ/view?usp=sharing) (weights).

Unzip the weights into "weights" directory and unzip the shot boundaries into the "Reported" directly. Please make these directories if they do not exist already 

