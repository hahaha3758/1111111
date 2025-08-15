import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import scipy, argparse, sys, os
from random import randrange, seed
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
import time
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

t_start = time.time()

def normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)


model = sys.argv[1] 
path_smi = '/public/home/chenlong666/Chunhuanzhang/chansheng10ci/AE/{}/'.format(model)
path_label = '/public/home/chenlong666/Chunhuanzhang/chansheng10ci/AE/{}/'.format(model)
file = open(path_smi + '{}.smi'.format(model,model),'r')
data = [line for line in file.readlines()]
data = np.array(data)
print('size of data:', np.shape(data), flush=True)



y_val_ori = open(path_label + 'label_{}.csv'.format(model,model),'r')
y_val_list = []
for i in y_val_ori.readlines():
    i = eval(i.strip())
    y_val_list.append(i)
y_val = np.array(y_val_list)
print('size of y_val:', np.shape(y_val), flush=True)

train_size = int(float(np.shape(data)[0]) * 0.8) 
print('size of train size:', train_size,flush=True)

if train_size < 1000:
    max_depth, min_samples_split, min_samples_leaf = 7, 3, 1
    subsample = 0.7
    C = 10
elif train_size >= 1000 and train_size < 5000:
    max_depth, min_samples_split, min_samples_leaf = 8, 4, 2
    subsample = 0.5
    C = 5
elif train_size >= 5000:
    max_depth, min_samples_split, min_samples_leaf = 9, 7, 3
    subsample = 0.3
    C = 1


parser = argparse.ArgumentParser(description='GBDT inputs')
parser.add_argument('--n_estimators', default=10000, type=int,
                    help='Maximum tree depth')
parser.add_argument('--dataset', default=model, type=str,
                    help='Dataset selected')
parser.add_argument('--max_depth', default=max_depth, type=int,
                    help='Maximum tree depth')
parser.add_argument('--learning_rate', default=0.01, type=float,
                    help='Learning rate for gbrt')
parser.add_argument('--criterion', default='friedman_mse', type=str,
                    help='Loss function for gbrt')
parser.add_argument('--subsample', default=subsample, type=float,
                    help='Subsample for fitting individual learners')
parser.add_argument('--max_features', default='sqrt', type=str,
                    help='Number of features to be considered')
parser.add_argument('--min_samples_split', default=min_samples_split, type=int,
                    help='Minimum sample num of each leaf node.')
parser.add_argument('--loss', default='ls', type=str,
                    help='Loss function to be optimized.')
parser.add_argument('--n_iter_no_change', default=None, type=int,
                    help='Early stopping will be used to terminate training')
parser.add_argument('--random_seed', default=0, type=int,
                    help='random seed')

args = parser.parse_known_args()[0]
print('GBDT parameter', flush=True)
print(args, flush=True)

for ii in range(1):
    results = np.zeros(y_val.shape)

    kf = KFold(n_splits=10, shuffle=True, random_state=ii)
    fold = 0
    for train_idx, test_idx in kf.split(data):
        fold += 1
        print('fold=', fold, flush=True)
        X_trainset, X_testset = data[train_idx], data[test_idx]
        y_train, y_test = y_val[train_idx], y_val[test_idx]
        f_train_smi = open(path_smi + '%s_regression_train_%d.smi' % (model, fold), 'w')
        f_valid_smi = open(path_smi + '%s_regression_valid_%d.smi' % (model, fold), 'w')
        f_test_smi  = open(path_smi + '%s_regression_test_%d.smi'  % (model, fold), 'w')
        f_train_label = open(path_smi + '%s_regression_train_%d.label' % (model, fold), 'w')
        f_valid_label = open(path_smi + '%s_regression_valid_%d.label' % (model, fold), 'w')
        f_test_label  = open(path_smi + '%s_regression_test_%d.label'  % (model, fold), 'w')

        test_num = np.shape(X_testset)[0]
        train_num = train_size
        valid_num = np.shape(X_trainset)[0] - train_num
        print('train_num:', train_num)
        print('valid_num:',valid_num)
        print('test_num:',test_num)

        for i in range(train_num):
            f_train_smi.write(X_trainset[i].strip() + '\n')
        for j in range(train_num, train_num + valid_num):
            f_valid_smi.write(X_trainset[j].strip() + '\n')
        for i in range(test_num):
            f_test_smi.write(X_testset[i].strip() + '\n')
        for k in range(train_num):
            f_train_label.write(str(y_train[k]) + '\n')
        for k in range(train_num, train_num + valid_num):
            f_valid_label.write(str(y_train[k]) + '\n')
        for p in range(test_num):
            f_test_label.write(str(y_test[p]) + '\n')
        f_train_smi.close()
        f_valid_smi.close()
        f_test_smi.close()
        f_train_label.close()
        f_valid_label.close()
        f_test_label.close()
        print('step1 finished!')


        
        cmd = 'cddd --input /public/home/chenlong666/Chunhuanzhang/chansheng10ci/AE/{}/{}_regression_train_{}.smi\
                --out /public/home/chenlong666/Chunhuanzhang/chansheng10ci/AE/{}_train_AE_{}.csv \
                --smiles_header smiles --no-preprocess'.format(model, model, fold, model, fold)
        os.system(cmd)

        cmd = 'cddd --input /public/home/chenlong666/Chunhuanzhang/chansheng10ci/AE/{}/{}_regression_valid_{}.smi\
                --out //public/home/chenlong666/Chunhuanzhang/chansheng10ci/AE/{}_valid_AE_{}.csv \
                --smiles_header smiles --no-preprocess'.format(model, model, fold, model, fold)
        os.system(cmd)

        cmd = 'cddd --input /public/home/chenlong666/Chunhuanzhang/chansheng10ci/AE/{}/{}_regression_test_{}.smi\
                --out /public/home/chenlong666/Chunhuanzhang/chansheng10ci/AE/{}_test_AE_{}.csv \
                --smiles_header smiles --no-preprocess'.format(model, model, fold, model, fold)
        os.system(cmd)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

t_end = time.time()
print('total time:', (t_end - t_start) / 3600, 'h', flush=True)