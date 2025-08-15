import os
import time
import sys

os.system('export MKL_THREADING_LAYER=GNU')
t_start = time.time()

name = sys.argv[1]
cmd='cddd --input /public/home/chenlong666/Chunhuanzhang/AE/{}/{}.smi --out /public/home/chenlong666/Chunhuanzhang/AE/{}_AE.csv --smiles_header smiles --no-preprocess'.format(name, name, name)
os.system(cmd)
t_end = time.time()
print('total time:',(t_end-t_start)/3600,'h',flush=True)