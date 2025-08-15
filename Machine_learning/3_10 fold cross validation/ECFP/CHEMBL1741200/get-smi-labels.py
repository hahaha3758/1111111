import csv 
import pandas as pd
import glob
import os
from collections import Counter, defaultdict
import numpy as np
import rdkit
from rdkit import Chem
import sys


def duplicates(lst):
    cnt= Counter(lst)
    return [key for key in cnt.keys() if cnt[key]> 0]


def indices(lst, items= None):
    items, ind= set(lst) if items is None else items, defaultdict(list)
    for i, v in enumerate(lst):
        if v in items: ind[v].append(i)
    return ind

# list1 = ['a','b','a','c','d','d','a']
# labels = [1,2,3,4,5,6]
# print(duplicates(list1))
# list2 = indices(list1,duplicates(list1))
# idx_keep = [x[0] for x in list2.values()]
# print(list2)
# print(idx_keep)
# new_labels = [labels[x]   for x in idx_keep]
# print(new_labels)
# exit()
dataset = sys.argv[1]
#path='C:/Users/administered/Desktop/new_data/'
path='//public/home/chenlong666/desktop/my_desk2/data'
assays = ['IC50','KI']
df_files = []
for assay in assays:
    filename = f'{assay}.csv'
    if os.path.exists(path+'%s/'%dataset+filename):
        df = pd.read_csv(path+'%s/'%dataset+filename,delimiter=';')
        df=df.loc[(df['Standard Relation']=="'='") &( df['Data Validity Comment']!='Outside typical range')&(df['Data Validity Comment']!='Non standard unit for type')]
        df.dropna(subset=['Standard Value','Smiles'],inplace=True)
        df_files.append(df)

df = pd.concat(df_files, axis=0)

df.loc[df['Standard Type']=='IC50','Binding Affinity']= 1.3633*np.log10(df.loc[df['Standard Type']=='IC50','Standard Value'].values*10**(-9)/2)
df.loc[df['Standard Type']=='Ki','Binding Affinity']= 1.3633*np.log10(df.loc[df['Standard Type']=='Ki','Standard Value'].values*10**(-9))

mol_merge = df['Molecule ChEMBL ID'].values
BAs_merge = df['Binding Affinity'].values
smiles_merge = df['Smiles'].values

idx_duplicates = indices(mol_merge, duplicates(mol_merge))
IDS_chemb =[]
labels_chemb = []
smiles_chemb = []

for x,y in idx_duplicates.items():
    if len(y)==1:
        IDS_chemb.append(x) 
        labels_chemb.append(BAs_merge[y].mean())
        smiles_chemb.append(smiles_merge[y[0]])
    elif len(y)>1:
        IDS_chemb.append(x) 
        labels_chemb.append(BAs_merge[y].mean())
        smiles_chemb.append(smiles_merge[y[0]])

if not os.path.exists(path+'%s/'%dataset+'smi-labels'):
    os.mkdir(path+'%s/'%dataset+'smi-labels')

fw_sm=open(path+'%s/'%dataset+'%s.smi'%(dataset),'w')
fw_lb=open(path+'%s/'%dataset+'label_%s.csv'%(dataset),'w')
SMILES_cano = []
for ismi,smi in enumerate(smiles_chemb):
    try: 
        smi_cano = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except:
        smi_cano = smi
    SMILES_cano.append(smi_cano)
    # print(smi_cano,file=fw_sm)
# fw_sm.close()

print(len(SMILES_cano),len(set(SMILES_cano)))

can_smi = indices(SMILES_cano, duplicates(SMILES_cano))
idx_keep = [idx[0] for idx in can_smi.values()]
new_labels = [labels_chemb[idx]  for idx in idx_keep]

for smi, idx in can_smi.items():
    if len(idx) > 1:
        print(smi,idx)
print(len(duplicates(SMILES_cano)))

for sm in duplicates(SMILES_cano):
    print(sm,file=fw_sm)
fw_sm.close()

for lb in new_labels:
    print(lb,file=fw_lb)
fw_lb.close()

