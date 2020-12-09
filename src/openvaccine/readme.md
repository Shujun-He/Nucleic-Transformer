# Code to reproduce results for the openvaccine dataset

Here I include the hypeparameters that give the best single model.

1. pretrain with all available sequences: ```./pretrain.sh```

2. train on targets: ```./run.sh```

3. to make predictions and generate a csv file for submission: ```./predict.sh``` then you can make a submission at https://www.kaggle.com/c/stanford-covid-vaccine/submissions
