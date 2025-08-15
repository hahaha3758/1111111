import pandas as pd

file_names = ['E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE210682_result/DESeq2_result/ph_cocaine_norm_statistics_0.4.csv']

df = pd.read_csv(file_names[0])

top_25_nodes = df.sort_values(by='lap_sum_norm_0.4', ascending=False).head(300)

top_25_nodes = top_25_nodes[['node']].rename(columns={'node': 'node_0.4'})

top_25_nodes.to_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE210682_result/DESeq2_result/top_300_nodes_0.4.csv', index=False)
