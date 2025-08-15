import lcc
import networkx as nx
import pandas as pd

df2 = pd.read_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top100/string_interactions_short_11.5_0.9.tsv',sep='\t')

labels = []
s = pd.read_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top100/DEGs_DESeq2.csv')
s = list(s['geneName'])  

for i in range(len(df2)):
    if df2.iloc[i]['#node1'] in s and df2.iloc[i].node2 in s:
        labels.append(1)
    else:
        labels.append(0)
df2.insert(loc=0,column='label',value=labels) 
df = df2[df2['label'] == 1] 
df.to_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top100/deg_ppi_0.9_clean.tsv',sep='\t')
G = nx.from_pandas_edgelist(df, '#node1', 'node2', 'combined_score')
G = G.to_undirected() 
largest_component = max(nx.connected_components(G), key=len) 
LCC = G.subgraph(largest_component)  
G=LCC  
node1 = []
node2 = []
node3 = []
for edge in list(G.edges): 
    node1.append(edge[0])
    node2.append(edge[1])

df = pd.DataFrame({
    'node1':node1,
    'node2':node2
})

df.to_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top100/deg_ppi_large.csv')

for node in list(G): 
    node3.append(node)

df = pd.DataFrame({ 
    'Gene':node3
})
df.to_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result/DEseq2_result/top100/deg_ppi_nodes_large.csv')
print('over')
