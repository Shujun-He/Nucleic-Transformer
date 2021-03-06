# NUCLEIC TRANSFORMER: DEEP LEARNING On NUCLEIC ACIDS WITH SELF-ATTENTION AND CONVOLUTIONS

Source code to reproduce results in the paper "NUCLEIC TRANSFORMER: DEEP LEARNING On NUCLEIC ACIDS WITH SELF-ATTENTION AND CONVOLUTIONS". Preprint available on bioarxiv: https://www.biorxiv.org/content/10.1101/2021.01.28.428629v1

<p align="center">
  <img src="https://github.com/Shujun-He/Nucleic-Transformer/blob/master/graphics/overview.PNG"/>
</p>

I also made a web app to use the models. Check it out at https://github.com/Shujun-He/Nucleic-Transformer-WebApp


## Requirements
I included a file (environment.yml) to recreate the exact environment I used. Since I also use this environment for computer vision tasks, it includes some other packages as well.

```
conda env create -f environment.yml
```

## Repo file structure 

The src folder includes all the code needed to reproduce results in the paper and the OpenVaccine competition. Additional instructions are in each folder

src/promoter_classification includes all the code and file needed to reproduce results for E.coli promoter classification

src/viral_identification includeds all the code needed to reproduce results for the viraminer dataset

src/openvaccine includes all the code needed to run a ten-fold model for the openvaccine dataset



## Datasets

### Promoter classification

This dataset is quite small so I include the file in the src folder

### Viraminer dataset

Download from https://github.com/NeuroCSUT/ViraMiner. I used the same train/val/test split as the viraminer paper.

### OpenVaccine dataset

For original dataset, see https://www.kaggle.com/c/stanford-covid-vaccine/data

In addition to the secondary structure features given by Das Lab, I also generated additional secondary structure features at 2 temperatures with 6 biophysical packages (12x), for these features, see https://www.kaggle.com/shujun717/openvaccine-12x-dataset
