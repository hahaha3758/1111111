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
    for i in range (1,11):
        lines1 = open(f'/public/home/chenlong666/Chunhuanzhang/chansheng10ci/ECFP/{name}/{name}_regression_train_{i}.smi', 'r').readlines()
        mols1 = [line.strip() for line in lines1]
        pool = multiprocessing.Pool(48)
        re1 = pool.starmap(Morgan, zip(mols1))
        pool.close()
        pool.join()
        np.save(f'/public/home/chenlong666/Chunhuanzhang/chansheng10ci/ECFP/{name}_train_ECFP_{i}.npy', np.array(re1))


        lines2 = open(f'/public/home/chenlong666/Chunhuanzhang/chansheng10ci/ECFP/{name}/{name}_regression_valid_{i}.smi',
                     'r').readlines()
        mols2 = [line.strip() for line in lines2]
        pool = multiprocessing.Pool(48)
        re2 = pool.starmap(Morgan, zip(mols2))
        pool.close()
        pool.join()
        np.save(f'/public/home/chenlong666/Chunhuanzhang/chansheng10ci/ECFP/{name}_valid_ECFP_{i}.npy', np.array(re2))


        lines3 = open(f'/public/home/chenlong666/Chunhuanzhang/chansheng10ci/ECFP/{name}/{name}_regression_test_{i}.smi',
                      'r').readlines()
        mols3 = [line.strip() for line in lines3]
        pool = multiprocessing.Pool(48)
        re3 = pool.starmap(Morgan, zip(mols3))
        pool.close()
        pool.join()
        np.save(f'/public/home/chenlong666/Chunhuanzhang/chansheng10ci/ECFP/{name}_test_ECFP_{i}.npy', np.array(re3))


#os.system(cmd)
t_end = time.time()
print('total time:',(t_end-t_start)/3600,'h',flush=True)