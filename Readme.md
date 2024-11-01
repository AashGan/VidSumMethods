# Shots in the dark

This repository contains the code for the paper "Shots in the dark" a replication and reproduction study into shot boundary detection and evaluation of video summarization.

# Setup

For this experiment, two different python versions were used
## Datasets and Videos
### Dataset
Extract the content downloaded from this [link](https://drive.google.com/file/d/1GmEHfITTp_bDJ3l_mC9KnOsxD-onCrpn/view?usp=sharing) to the path "Data\googlenet"

## Videos
Extract the video content from this [link](https://drive.google.com/file/d/1z8u1VoXEUvPIWWZpX-pd8TJGrbydxOJM/view?usp=sharing) to the path "Videos" 

This uses Python 3.10, for all the dependencies to be installed correctly set up a virtual environment with a version 3.10 Interpreter. Set this up using the following command

```bash
pip install -r requirements_2.txt
```
## 
To run these experiments, follow the instructions given in the following notebooks:

1. Running Fischer Extractor SumMe
2. Runing Fischer Extractor TVsum

The other 
1. Shot Boundary Consistency SumMe
2. Shot Boundary Consistency TVsum


## Post-Processing Evaluation Experiment

This uses Python 3.7.9, for all the dependencies to be installed correctly set up a virtual environment with a version 3.7.9 Interpreter. Set this up using the following command

```bash
p
pip install -r requirements_1.txt
```

To run these experiments, follow the instructions given in the following notebooks:
1. Evaluation Divergence Video Summarization
2. RandomSumMeScoring


# Exact paper reported results

The results obtained by re-running the notebooks may vary slightly from the original paper due to differences in the CUDA versions and pytorch version differences. To obtain the results reported from the paper. Please run the notebooks below with the weights/extracted features from our experiments
