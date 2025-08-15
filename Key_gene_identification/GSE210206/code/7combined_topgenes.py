import pandas as pd
import glob

file_names = [
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE210206_result/DESeq2_result/top_300_nodes_0.15.csv',
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE210206_result/DESeq2_result/top_300_nodes_0.4.csv',
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE210206_result/DESeq2_result/top_300_nodes_0.7.csv',
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE210206_result/DESeq2_result/top_300_nodes_0.9.csv'
]

dataframes = [pd.read_csv(file) for file in file_names]

merged_dataframe = pd.concat(dataframes,axis=1)

merged_dataframe.to_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE210206_result/DESeq2_result/top300genes_python.csv', index=False)



