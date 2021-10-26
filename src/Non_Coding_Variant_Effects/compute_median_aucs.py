import pandas as pd

df=pd.read_csv('test_aucs.csv')
deepsea=pd.read_excel('../41592_2015_BFnmeth3547_MOESM646_ESM.xlsx')
deepsea_aucs=deepsea.iloc[1:,4]
deepsea_aucs[599]=1

with open("test_results.txt",'w+') as f:
    f.write('###NT###\n')
    f.write(f"DNase_median_acu: {df.AUC.iloc[:125].median()}\n")
    f.write(f"TF_median_acu: {df.AUC.iloc[125:815].median()}\n")
    f.write(f"Histone_median_acu: {df.AUC.iloc[815:919].median()}\n")
    f.write('###Deep Sea###\n')
    f.write(f"DNase_median_acu: {deepsea_aucs[:125].median()}\n")
    f.write(f"TF_median_acu: {deepsea_aucs[125:815].median()}\n")
    f.write(f"Histone_median_acu: {deepsea_aucs[815:919].median()}\n")
