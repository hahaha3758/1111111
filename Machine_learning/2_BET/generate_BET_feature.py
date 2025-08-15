import os
import time
import sys

os.system('export MKL_THREADING_LAYER=GNU')
t_start = time.time()

name = sys.argv[1]

cmd='python "/public/home/chenlong666/desktop/PretrainModels/bt_pro/generate_bt_fps.py" --model_name_or_path "/public/home/chenlong666/desktop/PretrainModels/examples/models/"  --checkpoint_file "/public/home/chenlong666/desktop/PretrainModels/checkpoint_best.pt" --data_name_or_path  "/public/home/chenlong666/desktop/PretrainModels/examples/models/" --dict_file "/public/home/chenlong666/desktop/PretrainModels/examples/models/dict.txt" --target_file "/public/home/chenlong666/Chunhuanzhang/top300_chembl/{}/{}.smi" --save_feature_path "/public/home/chenlong666/Chunhuanzhang/BET/{}_BET.npy"'.format(name, name, name)
os.system(cmd)
t_end = time.time()
print('total time:',(t_end-t_start)/3600,'h',flush=True)