import pandas as pd
import glob

file_names = [
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top300/top_300_nodes_0.15.csv',
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top300/top_300_nodes_0.4.csv',
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top300/top_300_nodes_0.7.csv',
    'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top300/top_300_nodes_0.9.csv'
]


dataframes = [pd.read_csv(file) for file in file_names]

#dataframes = [pd.read_csv(file) for file in file_names]

#merged_dataframe = pd.concat(dataframes, ignore_index=True,axis=1)
merged_dataframe = pd.concat(dataframes,axis=1)


# df = pd.read_csv(file_names[0])

# first_column_name = df.columns[0]
# first_column_data = df[first_column_name]
# merged_dataframe.insert(0, first_column_name, first_column_data)
merged_dataframe.to_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top300/top300genes_python.csv', index=False)



