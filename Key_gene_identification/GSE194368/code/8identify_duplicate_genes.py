import pandas as pd


input_file = "E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE194368_result/DESeq2_result/top300genes_python.csv"  
output_file = "E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE194368_result/DESeq2_result/duplicate_gene_top300version.csv"  

data = pd.read_csv(input_file)


required_columns = ["node_0.15", "node_0.4", "node_0.7", "node_0.9"]
if not all(column in data.columns for column in required_columns):
    raise ValueError("The input file is missing the specified column")


common_genes = set(data[required_columns[0]])
for column in required_columns[1:]:
    common_genes &= set(data[column])


common_genes_df = pd.DataFrame(list(common_genes), columns=["Gene"])
common_genes_df.to_csv(output_file, index=False)

print(f"over  results have been saved to {output_file}")