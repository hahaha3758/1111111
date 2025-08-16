from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy.stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pickle, scipy, argparse, sys, os
from random import randrange, seed
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
import time, csv
t_start = time.time()
from sklearn.model_selection import StratifiedShuffleSplit




def normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)



def csv_to_npy(filename):
    nlist=[]
    with open(filename, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            
            if 'cddd_1' in row: 
                continue
            row_list=[]
            for i in row[2:]:
                row_list.append(eval(i))
            nlist.append(row_list)
    f.close()
    nlist=np.array(nlist)
    return nlist




model = sys.argv[1]
name = sys.argv[2]



path_smi = '/public/home/chenlong666/Chunhuanzhang/path_smi/{}/'.format(model)

path_label = '/public/home/chenlong666/Chunhuanzhang/path_smi/{}/'.format(model)
path_BET_npy = '/public/home/chenlong666/Chunhuanzhang/path_BET_npy/'
path_AE_csv = '/public/home/chenlong666/Chunhuanzhang/path_AE_csv/'
path_ECFP_npy = '/public/home/chenlong666/Chunhuanzhang/path_ECFP_npy/'

file = open(path_smi + '{}.smi'.format(model, model), 'r')
data = [line for line in file.readlines()]
data = np.array(data)
print('size of data:', np.shape(data), flush=True)



y_val_ori = open(path_label + 'label_{}.csv'.format(model, model), 'r')
y_val_list = []
for i in y_val_ori.readlines():
    i = eval(i.strip())
    y_val_list.append(i)
y_val = np.array(y_val_list)


train_size = int(float(np.shape(data)[0]) * 0.8)  
print('size of train size:', train_size, flush=True)

if train_size < 1000:
    
    
    C = 10
elif train_size >= 1000 and train_size < 5000:
    
    
    C = 5
elif train_size >= 5000:
    
    
    C = 1



paperSVM = argparse.ArgumentParser(description='SVM inputs')

paperSVM.add_argument('--dataset', default=model, type=str,
                      help='Dataset selected')
paperSVM.add_argument('--C', default=C, type=float,
                      help='Penalty parameter C of the error term')
paperSVM.add_argument('--kernel', default='rbf', type=str,
                      help='Kernel type to be used in the algorithm')
paperSVM.add_argument('--gamma', default='scale', type=str,
                      help='Kernel coefficient for rbf, poly and sigmoid')


argsSVM = paperSVM.parse_known_args()[0]
print('SVM parameter', flush=True)
print(argsSVM, flush=True)



X_train_BET = np.load(path_BET_npy + '{}_BET.npy'.format(model),allow_pickle=True)
X_train_AE= csv_to_npy(path_AE_csv + '{}_AE.csv'.format(model))
X_train_ECFP = np.load(path_ECFP_npy + '{}_ECFP.npy'.format(model),allow_pickle=True)
X_train = np.hstack((X_train_BET, X_train_AE, X_train_ECFP))

X_train = normalize(X_train)




y_train = np.array([float(i.strip()) for i in
                         open(path_smi + 'label_{}.csv'.format(model),
                              'r').readlines()])

X_test_BET = np.load(path_BET_npy + '{}_BET.npy'.format(name),allow_pickle=True)
X_test_AE = csv_to_npy(path_AE_csv + '{}_AE.csv'.format(name))
X_test_ECFP = np.load(path_ECFP_npy + '{}_ECFP.npy'.format(name),allow_pickle=True)
X_test = np.hstack((X_test_BET, X_test_AE, X_test_ECFP))
X_test = normalize(X_test)




y_pred_mean = []
for i in range(1):
    print('>>>>>>>>>>>>>training.............',flush=True)

    
    SVR_model = SVR(kernel = argsSVM.kernel, \
                    C=argsSVM.C, \
                    gamma=argsSVM.gamma,\
                    )
    SVR_model.fit(X_train, y_train)  
    y_pred = SVR_model.predict(X_test)  
    y_pred_mean.append(y_pred)  

print('the mininum value:',np.min(np.array(np.mean(y_pred_mean,axis=0))))
path_all_BA = '/public/home/chenlong666/Chunhuanzhang/prediction_result/'


np.savetxt(path_all_BA + '%s_%s_all_BA.txt'%(model,name),np.array(np.mean(y_pred_mean,axis=0)))

t_end = time.time()
print('total time:',(t_end-t_start)/3600,'h',flush=True)