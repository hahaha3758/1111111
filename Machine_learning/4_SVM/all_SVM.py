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
import pickle, scipy, h5py, argparse, sys, os
from random import randrange, seed
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
import time, csv

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

t_start = time.time()



def normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)



def csv_to_npy(filename):
    nlist = []
    with open(filename, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            
            if 'cddd_1' in row:  
                continue
            row_list = []
            for i in row[2:]:
                row_list.append(eval(i))
            nlist.append(row_list)
    f.close()
    nlist = np.array(nlist)
    return nlist



model = sys.argv[1]  
path_smi = '/public/home/chenlong666/Chunhuanzhang/path_smi/{}/'.format(model)
path_label = '/public/home/chenlong666/Chunhuanzhang/path_smi/{}/'.format(model)
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

paperSVM = argparse.ArgumentParser(description='SVM inputs')

paperSVM.add_argument('--dataset', default=model, type=str,
                      help='Dataset selected')
paperSVM.add_argument('--C', default=C, type=float,
                      help='Penalty parameter C of the error term')
paperSVM.add_argument('--kernel', default='rbf', type=str,
                      help='Kernel type to be used in the algorithm')
paperSVM.add_argument('--gamma', default='scale', type=str,
                      help='Kernel coefficient for rbf, poly and sigmoid')
paperSVM.add_argument('--random_seed', default=0, type=int,
                      help='random seed')
argsSVM = paperSVM.parse_known_args()[0]
print('SVM parameter', flush=True)
print(argsSVM, flush=True)


for ii in range(1):

    
    SVM_results_AE = np.zeros(y_val.shape)
    SVM_results_BET = np.zeros(y_val.shape)
    SVM_results_ECFP = np.zeros(y_val.shape)
    SVM_results_AE_BET = np.zeros(y_val.shape)
    SVM_results_AE_ECFP = np.zeros(y_val.shape)
    SVM_results_BET_ECFP = np.zeros(y_val.shape)
    SVM_results_AE_BET_ECFP = np.zeros(y_val.shape)

    
    kf = KFold(n_splits=10, shuffle=True, random_state=ii)
    fold = 0
    for train_idx, test_idx in kf.split(data):
        fold += 1
        print('fold=', fold, flush=True)
        
        path_BET_npy = '/public/home/chenlong666/Chunhuanzhang/huizong_SVM/all_BET_npy/'
        
        path_AE_npy = '/public/home/chenlong666/Chunhuanzhang/huizong_SVM/all_AE_CSV/'
        path_ECFP_npy = '/public/home/chenlong666/Chunhuanzhang/huizong_SVM/all_ECFP_npy/'  

        
        X_train_1_BET = np.load(path_BET_npy + '{}_train_BET_{}.npy'.format(model, fold),
                                allow_pickle=True)
        X_train_2_BET = np.load(path_BET_npy + '{}_valid_BET_{}.npy'.format(model, fold),
                                allow_pickle=True)
        X_train_BET = np.concatenate((X_train_1_BET, X_train_2_BET))
        
        
        
        X_test_BET = np.load(path_BET_npy + '{}_test_BET_{}.npy'.format(model, fold),
                             allow_pickle=True)
        
        X_train_BET = normalize(X_train_BET)
        X_test_BET = normalize(X_test_BET)

        y_train_1_BET = [float(i.strip()) for i in
                         open(path_smi + '{}_regression_train_{}.label'.format(model, fold),
                              'r').readlines()]
        y_train_2_BET = [float(i.strip()) for i in
                         open(path_smi + '{}_regression_valid_{}.label'.format(model, fold),
                              'r').readlines()]
        y_train_BET = np.concatenate((y_train_1_BET, y_train_2_BET))
        
        
        
        
        
        y_test_BET = np.array([float(i.strip()) for i in
                               open(path_smi + '{}_regression_test_{}.label'.format(model, fold),
                                    'r').readlines()])

        
        X_train_1_AE = csv_to_npy(path_AE_npy + '{}_train_AE_{}.csv'.format(model, fold), )
        X_train_2_AE = csv_to_npy(path_AE_npy + '{}_valid_AE_{}.csv'.format(model, fold), )
        X_train_AE = np.concatenate((X_train_1_AE, X_train_2_AE))
        
        
        
        X_test_AE = csv_to_npy(path_AE_npy + '{}_test_AE_{}.csv'.format(model, fold), )
        
        X_train_AE = normalize(X_train_AE)
        X_test_AE = normalize(X_test_AE)

        y_train_1_AE = [float(i.strip()) for i in
                        open(path_smi + '{}_regression_train_{}.label'.format(model, fold),
                             'r').readlines()]
        y_train_2_AE = [float(i.strip()) for i in
                        open(path_smi + '{}_regression_valid_{}.label'.format(model, fold),
                             'r').readlines()]
        y_train_AE = np.concatenate((y_train_1_AE, y_train_2_AE))
        
        
        
        
        
        y_test_AE = np.array([float(i.strip()) for i in
                              open(path_smi + '{}_regression_test_{}.label'.format(model, fold),
                                   'r').readlines()])
        
        
        

        
        X_train_1_ECFP = np.load(path_ECFP_npy + '{}_train_ECFP_{}.npy'.format(model, fold),
                                 allow_pickle=True)
        X_train_2_ECFP = np.load(path_ECFP_npy + '{}_valid_ECFP_{}.npy'.format(model, fold),
                                 allow_pickle=True)
        X_train_ECFP = np.concatenate((X_train_1_ECFP, X_train_2_ECFP))
        
        
        
        X_test_ECFP = np.load(path_ECFP_npy + '{}_test_ECFP_{}.npy'.format(model, fold),
                              allow_pickle=True)
        
        X_train_ECFP = normalize(X_train_ECFP)
        X_test_ECFP = normalize(X_test_ECFP)

        y_train_1_ECFP = [float(i.strip()) for i in
                          open(path_smi + '{}_regression_train_{}.label'.format(model, fold),
                               'r').readlines()]
        y_train_2_ECFP = [float(i.strip()) for i in
                          open(path_smi + '{}_regression_valid_{}.label'.format(model, fold),
                               'r').readlines()]
        y_train_ECFP = np.concatenate((y_train_1_ECFP, y_train_2_ECFP))
        
        
        
        
        
        y_test_ECFP = np.array([float(i.strip()) for i in
                                open(path_smi + '{}_regression_test_{}.label'.format(model, fold),
                                     'r').readlines()])
        
        
        

        print('>>>>>>>>>>>>>training.............', flush=True)
        
        print(
            '
            flush=True)

        SVR_model = SVR(kernel=argsSVM.kernel, \
                        C=argsSVM.C, \
                        gamma=argsSVM.gamma)

        
        SVR_model.fit(X_train_BET, y_train_BET)  
        SVM_results_BET[test_idx] = SVR_model.predict(
            X_test_BET)  
        SVM_y_pred_BET = SVR_model.predict(X_test_BET)  
        
        print(f'SVM_BET的BA值：{np.min(SVM_y_pred_BET)}', flush=True)  
        SVM_RMSD_BET = np.sqrt(mean_squared_error(y_test_BET, SVM_y_pred_BET))  
        SVM_pearsonr_BET = scipy.stats.pearsonr(y_test_BET, SVM_y_pred_BET)  
        SVM_r2_BET = r2_score(y_test_BET, SVM_y_pred_BET)  
        print('SVM_BET:RMSD: %f, P: %f, R^2: %f' % (
            SVM_RMSD_BET, SVM_pearsonr_BET[0], SVM_r2_BET), flush=True)

        
        SVR_model.fit(X_train_AE, y_train_AE)
        SVM_results_AE[test_idx] = SVR_model.predict(X_test_AE)
        SVM_y_pred_AE = SVR_model.predict(X_test_AE)
        print(f'SVM_AE的BA值 ：{np.min(SVM_y_pred_AE)}', flush=True)
        SVM_RMSD_AE = np.sqrt(mean_squared_error(y_test_AE, SVM_y_pred_AE))
        SVM_pearsonr_AE = scipy.stats.pearsonr(y_test_AE, SVM_y_pred_AE)
        SVM_r2_AE = r2_score(y_test_AE, SVM_y_pred_AE)
        print('SVM_AE :RMSD: %f, P: %f, R^2: %f' % (
            SVM_RMSD_AE, SVM_pearsonr_AE[0] , SVM_r2_AE), flush=True)

        
        SVR_model.fit(X_train_ECFP, y_train_ECFP)
        SVM_results_ECFP[test_idx] = SVR_model.predict(X_test_ECFP)
        SVM_y_pred_ECFP = SVR_model.predict(X_test_ECFP)
        print(f'SVM_ECFP的BA值 ：{np.min(SVM_y_pred_ECFP)}', flush=True)
        SVM_RMSD_ECFP = np.sqrt(mean_squared_error(y_test_ECFP, SVM_y_pred_ECFP))
        SVM_pearsonr_ECFP = scipy.stats.pearsonr(y_test_ECFP, SVM_y_pred_ECFP)
        SVM_r2_ECFP = r2_score(y_test_ECFP, SVM_y_pred_ECFP)
        print('SVM_ECFP :RMSD: %f, P: %f, R^2: %f' % (
            SVM_RMSD_ECFP, SVM_pearsonr_ECFP[0], SVM_r2_ECFP), flush=True)

        
        SVM_results_AE_BET[test_idx] = (SVM_results_BET[test_idx] + SVM_results_AE[test_idx]) / 2
        SVM_y_pred_AE_BET = (SVM_y_pred_BET + SVM_y_pred_AE) / 2
        y_test_AE_BET = (y_test_AE + y_test_BET) / 2
        print(f'SVM_AE_BET的BA值    ：{np.min(SVM_y_pred_AE_BET)}', flush=True)
        SVM_RMSD_AE_BET = np.sqrt(mean_squared_error(y_test_AE_BET, SVM_y_pred_AE_BET))
        SVM_pearsonr_AE_BET = scipy.stats.pearsonr(y_test_AE_BET, SVM_y_pred_AE_BET)
        SVM_r2_AE_BET = r2_score(y_test_AE_BET, SVM_y_pred_AE_BET)
        print('SVM_AE_BET    :RMSD: %f, P: %f, R^2: %f' % (
        SVM_RMSD_AE_BET, SVM_pearsonr_AE_BET[0], SVM_r2_AE_BET),
              flush=True)

        
        SVM_results_AE_ECFP[test_idx] = (SVM_results_ECFP[test_idx] + SVM_results_AE[test_idx]) / 2
        SVM_y_pred_AE_ECFP = (SVM_y_pred_ECFP + SVM_y_pred_AE) / 2
        y_test_AE_ECFP = (y_test_ECFP + y_test_AE) / 2
        print(f'SVM_AE_ECFP的BA值    ：{np.min(SVM_y_pred_AE_ECFP)}', flush=True)
        SVM_RMSD_AE_ECFP = np.sqrt(mean_squared_error(y_test_AE_ECFP, SVM_y_pred_AE_ECFP))
        SVM_pearsonr_AE_ECFP = scipy.stats.pearsonr(y_test_AE_ECFP, SVM_y_pred_AE_ECFP)
        SVM_r2_AE_ECFP = r2_score(y_test_AE_ECFP, SVM_y_pred_AE_ECFP)
        print('SVM_AE_ECFP    :RMSD: %f, P: %f, R^2: %f' % (
        SVM_RMSD_AE_ECFP, SVM_pearsonr_AE_ECFP[0] , SVM_r2_AE_ECFP),
              flush=True)

        
        SVM_results_BET_ECFP[test_idx] = (SVM_results_ECFP[test_idx] + SVM_results_BET[test_idx]) / 2
        SVM_y_pred_BET_ECFP = (SVM_y_pred_ECFP + SVM_y_pred_BET) / 2
        y_test_BET_ECFP = (y_test_ECFP + y_test_BET) / 2
        print(f'SVM_BET_ECFP的BA值    ：{np.min(SVM_y_pred_BET_ECFP)}', flush=True)
        SVM_RMSD_BET_ECFP = np.sqrt(mean_squared_error(y_test_BET_ECFP, SVM_y_pred_BET_ECFP))
        SVM_pearsonr_BET_ECFP = scipy.stats.pearsonr(y_test_BET_ECFP, SVM_y_pred_BET_ECFP)
        SVM_r2_BET_ECFP = r2_score(y_test_BET_ECFP, SVM_y_pred_BET_ECFP)
        print('SVM_BET_ECFP    :RMSD: %f, P: %f, R^2: %f' % (
        SVM_RMSD_BET_ECFP, SVM_pearsonr_BET_ECFP[0], SVM_r2_BET_ECFP),
              flush=True)

        
        SVM_results_AE_BET_ECFP[test_idx] = (SVM_results_ECFP[test_idx] + SVM_results_BET[test_idx] + SVM_results_AE[
            test_idx]) / 3
        SVM_y_pred_AE_BET_ECFP = (SVM_y_pred_ECFP + SVM_y_pred_BET + SVM_y_pred_AE) / 3
        y_test_AE_BET_ECFP = (y_test_ECFP + y_test_BET + y_test_AE) / 3
        print(f'SVM_AE_BET_ECFP的BA值    ：{np.min(SVM_y_pred_AE_BET_ECFP)}', flush=True)
        SVM_RMSD_AE_BET_ECFP = np.sqrt(mean_squared_error(y_test_AE_BET_ECFP, SVM_y_pred_AE_BET_ECFP))
        SVM_pearsonr_AE_BET_ECFP = scipy.stats.pearsonr(y_test_AE_BET_ECFP, SVM_y_pred_AE_BET_ECFP)
        SVM_r2_AE_BET_ECFP = r2_score(y_test_AE_BET_ECFP, SVM_y_pred_AE_BET_ECFP)
        print('SVM_AE_BET_ECFP    :RMSD: %f, P: %f, R^2: %f' % (
        SVM_RMSD_AE_BET_ECFP, SVM_pearsonr_AE_BET_ECFP[0] , SVM_r2_AE_BET_ECFP),
              flush=True)

    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@最终结果@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
          flush=True)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@最终结果@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
          flush=True)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SVM%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',
          flush=True)

    
    SVM_RMSD_BET = np.sqrt(mean_squared_error(y_val, SVM_results_BET))
    SVM_pearsonr_BET = scipy.stats.pearsonr(y_val, SVM_results_BET)
    SVM_determination_BET = r2_score(y_val, SVM_results_BET)
    print('SVM_BET:%d-th Final RMSD: %f, Final P: %f, Final R^2: %f' % (
        ii, SVM_RMSD_BET, SVM_pearsonr_BET[0], SVM_determination_BET), flush=True)

    
    SVM_RMSD_AE = np.sqrt(mean_squared_error(y_val, SVM_results_AE))
    SVM_pearsonr_AE = scipy.stats.pearsonr(y_val, SVM_results_AE)
    SVM_determination_AE = r2_score(y_val, SVM_results_AE)
    print('SVM_AE :%d-th Final RMSD: %f, Final P: %f, Final R^2: %f' % (
        ii, SVM_RMSD_AE, SVM_pearsonr_AE[0] , SVM_determination_AE), flush=True)

    
    SVM_RMSD_ECFP = np.sqrt(mean_squared_error(y_val, SVM_results_ECFP))
    SVM_pearsonr_ECFP = scipy.stats.pearsonr(y_val, SVM_results_ECFP)
    SVM_determination_ECFP = r2_score(y_val, SVM_results_ECFP)
    print('SVM_ECFP :%d-th Final RMSD: %f, Final P: %f, Final R^2: %f' % (
        ii, SVM_RMSD_ECFP, SVM_pearsonr_ECFP[0] , SVM_determination_ECFP), flush=True)

    
    SVM_RMSD_AE_BET = np.sqrt(mean_squared_error(y_val, SVM_results_AE_BET))
    SVM_pearsonr_AE_BET = scipy.stats.pearsonr(y_val, SVM_results_AE_BET)
    SVM_determination_AE_BET = r2_score(y_val, SVM_results_AE_BET)
    print('SVM_AE_BET    :%d-th Final RMSD: %f, Final P: %f, Final R^2: %f' % (
        ii, SVM_RMSD_AE_BET, SVM_pearsonr_AE_BET[0], SVM_determination_AE_BET), flush=True)

    
    SVM_RMSD_AE_ECFP = np.sqrt(mean_squared_error(y_val, SVM_results_AE_ECFP))
    SVM_pearsonr_AE_ECFP = scipy.stats.pearsonr(y_val, SVM_results_AE_ECFP)
    SVM_determination_AE_ECFP = r2_score(y_val, SVM_results_AE_ECFP)
    print('SVM_AE_ECFP    :%d-th Final RMSD: %f, Final P: %f, Final R^2: %f' % (
        ii, SVM_RMSD_AE_ECFP, SVM_pearsonr_AE_ECFP[0] , SVM_determination_AE_ECFP),
          flush=True)

    
    SVM_RMSD_BET_ECFP = np.sqrt(mean_squared_error(y_val, SVM_results_BET_ECFP))
    SVM_pearsonr_BET_ECFP = scipy.stats.pearsonr(y_val, SVM_results_BET_ECFP)
    SVM_determination_BET_ECFP = r2_score(y_val, SVM_results_BET_ECFP)
    print('SVM_BET_ECFP    :%d-th Final RMSD: %f, Final P: %f, Final R^2: %f' % (
        ii, SVM_RMSD_BET_ECFP, SVM_pearsonr_BET_ECFP[0], SVM_determination_BET_ECFP),
          flush=True)

    
    SVM_RMSD_AE_BET_ECFP = np.sqrt(mean_squared_error(y_val, SVM_results_AE_BET_ECFP))
    SVM_pearsonr_AE_BET_ECFP = scipy.stats.pearsonr(y_val, SVM_results_AE_BET_ECFP)
    SVM_determination_AE_BET_ECFP = r2_score(y_val, SVM_results_AE_BET_ECFP)
    print('SVM_AE_BET_ECFP    :%d-th Final RMSD: %f, Final P: %f, Final R^2: %f' % (
        ii, SVM_RMSD_AE_BET_ECFP, SVM_pearsonr_AE_BET_ECFP[0] ,
        SVM_determination_AE_BET_ECFP),
          flush=True)

    
    
    
    print(f'{ii}-th SVM_AE的BA值 ：{np.min(SVM_results_AE):f}', flush=True)
    print(f'{ii}-th SVM_BET的BA值：{np.min(SVM_results_BET):f}', flush=True)
    print(f'{ii}-th SVM_ECFP的BA值 ：{np.min(SVM_results_ECFP):f}', flush=True)
    print(f'{ii}-th SVM_AE_BET的BA值：{np.min(SVM_results_AE_BET):f}', flush=True)
    print(f'{ii}-th SVM_AE_ECFP的BA值    ：{np.min(SVM_results_AE_ECFP):f}', flush=True)
    print(f'{ii}-th SVM_BET_ECFP的BA值    ：{np.min(SVM_results_BET_ECFP):f}', flush=True)
    print(f'{ii}-th SVM_AE_BET_ECFP的BA值    ：{np.min(SVM_results_AE_BET_ECFP):f}', flush=True)

t_end = time.time()
print('total time:', (t_end - t_start) / 3600, 'h', flush=True)