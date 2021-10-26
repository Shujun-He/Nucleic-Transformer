# classifying effects of non-coding variants

1. download datasets from http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz
2. create folder ```deepsea_train``` and unzip contents into folder and run ```preprocess.py``` to extract train val set to a numpy file (for faster data loading)
3. ```bash run.sh``` to run training
4. ```bash test.sh``` to make inference on the test set
5. ```compute_val_aucs.py``` and ```compute_median_aucs.py``` to calculate test aucs and median aucs in TF/DNS/HM
