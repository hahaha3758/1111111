import os

import pandas as pd
import numpy as np
import math
from rdkit import Chem as ch
from rdkit import rdBase, Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Draw
import multiprocessing
import time
import sys

os.system('export MKL_THREADING_LAYER=GNU')
t_start = time.time()

def Morgan(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return list(fp)


if __name__ == "__main__":
    name = sys.argv[1]  # Get target from command line

    lines = open(f'/public/home/chenlong666/Chunhuanzhang/top300_chembl/{name}/{name}.smi', 'r').readlines()
    mols = [line.strip() for line in lines]
    pool = multiprocessing.Pool(48)
    re = pool.starmap(Morgan, zip(mols))
    pool.close()
    pool.join()
    np.save(f'/public/home/chenlong666/Chunhuanzhang/ECFP/{name}_ECFP.npy', np.array(re))

#os.system(cmd)
t_end = time.time()
print('total time:',(t_end-t_start)/3600,'h',flush=True)