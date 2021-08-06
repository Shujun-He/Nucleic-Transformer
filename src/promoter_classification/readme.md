# Source code to train nucleic transformer to reproduce results in the paper

Dataset is included here in v9d3.csv

To run: 
```./run.sh```

To get cross validation results: 
```./evaluate.sh```

Results will be in cv.txt

To extract top promoter motifs based on attention weights:
```extract_motif.py```

An eps file named promoter_motifs.eps will be generated
