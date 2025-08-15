# Meta-analysis and Multiscale Topology in Interactomic Network for Antiaddiction Drug Repopusing

This script is applicable to the paper "Meta analysis and Multiscale Topology in Interactive Network for Anti addiction Drug Repopulation". The content of the script includes all the code and raw data involved in the paper, mainly divided into two parts: “key_gene recognition” and “Machine_learning”. The table (“BA_prediction_results.xlsx”) contains the final predicted BA values for each drug in DrugBank.

## Requirements

### OS Requirements
- CentOS Linux 7 (Core)

### Python Dependencies
- setuptools (>=18.0)
- python (>=3.7)
- pytorch (>=1.2)
- rdkit (2020.03)
- biopandas (0.2.7)
- numpy (1.17.4)
- scikit-learn (0.23.2)
- scipy (1.5.2)
- pandas (0.25.3)
- cython (0.29.17)

### R Dependencies
- R（>=4.4.1）
- GEOquery (2.72.0)
- DESeq2 (1.44.0)
- org.Hs.eg.db (3.19.1)
- ggplot2 (3.5.1)
- dplyr (1.1.4)
- data.table (1.16.2)
- AnnotationDbi (1.66.0)
- clusterProfiler (4.12.6)
- readr (2.1.5)

## Download the repository

Download the repository from Github:

```bash
git clone https://github.com/hahaha3758/Drug_repopusing.git
```

## Data sources

- Transcriptome data sourced from the National Center for Biotechnology Information ([NCBI](https://www.ncbi.nlm.nih.gov/)), GEO access is GSE174409, GSE182321, GSE194368, GSE210206, GSE210682, GSE260711,GSE167922.
- The data for drawing the PPI network comes from STRING ([STRING](https://string-db.org/), version=11.5).
- The inhibitor data is sourced from CHEMBL ([CHEMBL](https://www.ebi.ac.uk/chembl/)).

The above data has been published.

## Key gene identification

Each dataset folder in this section contains the corresponding code, data, and results. The number at the beginning of the code file name indicates its running order, for example, code starting with "1_" should be run first.

## Machine learning

Each folder in this section contains the required code, data, and results. The first digit of the folder name represents the order of its operations, for example, folder contents starting with "1_" should be prioritized for processing.

## Install the pretrained model for molecular fingerprint generation

### AE fingerprint environment

The autoencoder (AE) feature generation follows the work "Learning Continuous and Data-Driven Molecular Descriptors by Translating Equivalent Chemical Representations" by Robin Winter, Floriane Montanari, Frank Noe and Djork-Arne Clevert.

```bash
git clone https://github.com/jrwnter/cddd.git
cd cddd
conda env create -f environment.yml
source activate cddd
```

### BET fingerprint environment

The bidirectional encoder transformer (BET) feature generation follows the work "Extracting Predictive Representations from Hundreds of Millions of Molecules" by Dong Chen, Jiaxin Zheng, Guo-Wei Wei, and Feng Pan.

```bash
git clone https://github.com/WeilabMSU/PretrainModels.git
cd PretrainModels/bt_pro
mkdir bt_pro
mkdir bt_pro/fairseq
mkdir bt_pro/fairseq/data
python setup.py build_ext --inplace
mv ./bt_pro/fairseq/data/* ./fairseq/data/
```

### ECFP fingerprint environment

The extended-connectivity fingerprint (ECFP) feature generation follows the work "Extended-connectivity fingerprints" by Rogers, David and Hahn, Mathew.

```python
from rdkit import Chem as ch
from rdkit import rdBase, Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Draw

def Morgan(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return list(fp)
```

## Generate molecular fingerprints

After creating the environment, use the following code to generate AE fingerprints, BET fingerprints, and ECFP fingerprints separately.

```bash
python generate_AE_feature.py
python generate_BET_feature.py
python generate_ECFP_feature.py
```

## Build machine learning model with the molecular fingerprints

We built a machine learning model to predict binding affinity.

```bash
python BA_value_fei_SVM_con.py
```

## Reference

1. Chen, Dong, et al. "Extracting predictive representations from hundreds of millions of molecules." The journal of physical chemistry letters 12.44 (2021): 10793-10801.
2. Winter, Robin, et al. "Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations." Chemical science 10.6 (2019): 1692-1701.
3. Rogers, David, and Mathew Hahn. "Extended-connectivity fingerprints." Journal of chemical information and modeling 50.5 (2010): 742-754.
