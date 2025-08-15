import pandas as pd
import glob

file_names = [
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE194368_result/DESeq2_result/ph_cocaine_norm_statistics_0.15.csv',
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE194368_result/DESeq2_result/ph_cocaine_norm_statistics_0.4.csv',
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE194368_result/DESeq2_result/ph_cocaine_norm_statistics_0.7.csv',
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE194368_result/DESeq2_result/ph_cocaine_norm_statistics_0.9.csv'
]

dataframes = [pd.read_csv(file, usecols=range(2, 6)) for file in file_names]

merged_dataframe = pd.concat(dataframes,axis=1)

df = pd.read_csv(file_names[0])

first_column_name = df.columns[1]
first_column_data = df[first_column_name]
merged_dataframe.insert(0, first_column_name, first_column_data)
merged_dataframe.to_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE194368_result/DESeq2_result/combined_ph_cocaine_norm_statistics.csv', index=False)



