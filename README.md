# Nucleic Transformer: Classifying DNA sequences with Self-attention and Convolutions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5641875.svg)](https://doi.org/10.5281/zenodo.5641875)


Source code to reproduce results in the paper "Nucleic Transformer: Classifying DNA sequences with Self-attention and Convolutions".

<p align="center">
  <img src="https://github.com/Shujun-He/Nucleic-Transformer/blob/master/graphics/overview.PNG"/>
</p>


## How to use the models

I also made a web app to use the models. Check it out at https://github.com/Shujun-He/Nucleic-Transformer-WebApp


## Requirements
I included a file (environment.yml) to recreate the exact environment I used. Since I also use this environment for computer vision tasks, it includes some other packages as well. This should take around 10 minutes. After installing anaconda:


```
conda env create -f environment.yml
```

Then to activate the environment

```
conda activate torch
```

Additionally, you will need Nvidai Apex: https://github.com/NVIDIA/apex

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install .
```

Also you need to install the Ranger optimizer

```bash
git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
cd Ranger-Deep-Learning-Optimizer
pip install -e . 
```

## Repo file structure 

The src folder includes all the code needed to reproduce results in the paper and the OpenVaccine competition. Additional instructions are in each folder

```src/Ecoli_Promoter_classification``` includes all the code and file needed to reproduce results for E.coli promoter classification

```src/Eukaryotic_Promoters_Classification``` includes all the code and file needed to reproduce results for eukaryotic promoter classification



```src/Non_Coding_Variant_Effects``` includeds all the code needed to reproduce results for the deepsea dataset

```src/Viral_identification``` includeds all the code needed to reproduce results for the viraminer dataset

```src/Enchancer_classification``` includeds all the code needed to reproduce results for the enhancer dataset

```src/OpenVaccine``` includes all the code needed to run a ten-fold model for the openvaccine dataset



## Datasets

### Promoter classification

This dataset is quite small so I include the file in the src folder

### Viraminer dataset

Download from https://github.com/NeuroCSUT/ViraMiner. I used the same train/val/test split as the viraminer paper.


