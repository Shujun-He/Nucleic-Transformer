# Source code to train nucleic transformer to reproduce results in the paper for the viraminer dataset

Dataset can be downloaded at https://github.com/NeuroCSUT/ViraMiner/tree/master/data/DNA_data

Download fullset_test.csv, fullset_train.csv, and fullset_validation.csv and put them on directory above the folder where you plan to run training (their paths should be ../fullset_test.csv etc)

To run training:  ```./run.sh```

You might need to lower the batch size depending on what GPU you have. If you run into memory error with cuda, lower --batch_size in run.sh

To check results on the test set: ```./evaluate_test.sh```

Test results will be saved in a pickle file named test_results.p, and the AUC score will be printed out to test_score.txt
