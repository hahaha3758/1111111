import gudhi
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from functools import reduce
dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

from time import process_time

start = process_time()




def call_eig(A):
    eigens = np.linalg.eigvalsh(A)
    return np.real(eigens)


class network_complex:
    def __init__(self,path,path_deg) -> None:
        '''
        path: string network 的文件路径
        '''
        
        
        self.path = path

        
        df = pd.read_csv(self.path,sep='\t')
        count = 0

        degs = pd.read_csv(path_deg)
        self.nodes = list(degs['geneName']) 

        self.node_id = {}
        for i in range(len(self.nodes)): 
            self.node_id[self.nodes[i]] = i
        
        self.edges = {}
        for i in range(len(df)): 
            node1 = df['
            node2 = df['node2'].iloc[i]
            score = float(df['combined_score'].iloc[i])
            self.edges[f'{node1}_{node2}'] = score
        pass

    
    def rips(self,start = 0,end = 0.3, stride = 0.03):
        matrixA = np.ones((len(self.nodes),len(self.nodes))) * 1000 
        
        intervals = np.arange(start,end+stride,stride) 

        features = np.zeros((len(intervals)-1,3)) 

        def get_inter_id(num):
            for i in range(len(intervals)-1) :
                if intervals[i] <= num < intervals[i+1]:
                    return i
            return len(intervals) -1 -1
        
        for i in range(len(self.nodes)):
            for j in range(i,len(self.nodes)):
                node1 = self.nodes[i]
                node2 = self.nodes[j]

                score = -1000  
                try:
                    score = self.edges[f'{node1}_{node2}']
                except:
                    pass
                try:
                    score = self.edges[f'{node2}_{node1}']
                except:
                    pass
                if score != -1000:
                    matrixA[i,j] = 1-score
                    matrixA[j,i] = 1-score
        
        rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=1) 
        Ph = rips_complex.create_simplex_tree(max_dimension = 2).persistence() 
        temp_bar = np.zeros(len(Ph),dtype=dt) 

        ct = 0
        for simplex in Ph:
            dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
            if d-b > 0.00002 and d != math.inf: 
                temp_bar[ct]['dim'] = dim 
                temp_bar[ct]['birth'] = b
                temp_bar[ct]['death'] = d
                ct+=1
        bars = temp_bar[:ct] 

        for bar in bars: 
            if bar['dim'] == 0:
                features[get_inter_id(bar['death']),0] += 1
            if bar['dim'] == 1:
                features[get_inter_id(bar['birth']),1] += 1
                features[get_inter_id(bar['death']),2] += 1
        
        return features.flatten()  
    
    def lap(self,start = 0,end = 0.3, stride = 0.03,remove_id=None):
        matrixA = np.zeros((len(self.nodes),len(self.nodes)))
        intervals = np.arange(start,end+stride,stride)
        num_inter = (end-start)/stride

        eigen_values = []
        for radius in intervals: 
            for i in range(len(self.nodes)-1):
                for j in range(i+1,len(self.nodes)):
                    if i != remove_id and j != remove_id:
                        node1 = self.nodes[i]
                        node2 = self.nodes[j]

                        score = -1000
                        try:
                            score = self.edges[f'{node1}_{node2}']
                        except:
                            pass
                        try:
                            score = self.edges[f'{node2}_{node1}']
                        except:
                            pass

                        if score != -1000 and (1-score)<=radius:
                            matrixA[i,j] = -1
                            matrixA[j,i] = -1
            
            for i in range(len(self.nodes)):
                matrixA[i,i] = - np.sum(matrixA, axis = 1)[i]

            eigens = call_eig(matrixA) 
            nonzero_eigens = eigens[eigens>1e-6] 

            if nonzero_eigens.shape[0] > 0:
                mini = nonzero_eigens.min()
            else:
                mini = 0.0

            eigen_values.append(mini) 
        return np.array(eigen_values) 
        
    def lap_statistics(self,start = 0,end = 0.3, stride = 0.03,remove_id=None):
        matrixA = np.zeros((len(self.nodes),len(self.nodes)))
        intervals = np.arange(start,end+stride,stride)
        num_inter = (end-start)/stride

        eigen_values = []
        for radius in intervals:
            for i in range(len(self.nodes)-1):
                for j in range(i+1,len(self.nodes)):
                    if i != remove_id and j != remove_id:
                        node1 = self.nodes[i]
                        node2 = self.nodes[j]

                        score = -1000
                        try:
                            score = self.edges[f'{node1}_{node2}']
                        except:
                            pass
                        try:
                            score = self.edges[f'{node2}_{node1}']
                        except:
                            pass

                        if score != -1000 and (1-score)<=radius:
                            matrixA[i,j] = -1
                            matrixA[j,i] = -1
            
            for i in range(len(self.nodes)):
                matrixA[i,i] = - np.sum(matrixA, axis = 1)[i]

            eigens = call_eig(matrixA)
            nonzero_eigens = eigens[eigens>1e-6]

            if nonzero_eigens.shape[0] > 0:
                features_ele1 = [
                    np.sum(nonzero_eigens),
                    np.mean(nonzero_eigens),
                    np.max(nonzero_eigens),
                    np.std(nonzero_eigens),
                    np.min(nonzero_eigens),
                    ]
            else:
                features_ele1 = [
                    0.0, 0.0, 0.0, 0.0, 0.0
                ]


            
            
            
            

            eigen_values.append(features_ele1)

        return np.array(eigen_values).flatten()

    
    def remove_node(self,remove_id,start = 0,end = 0.3, stride = 0.03):
        matrixA = np.ones((len(self.nodes),len(self.nodes))) * 1000
        
        intervals = np.arange(start,end+stride,stride)

        features = np.zeros((len(intervals)-1,3))


        def get_inter_id(num):
            for i in range(len(intervals)-1) :
                if intervals[i] <= num < intervals[i+1]:
                    return i
            return len(intervals) -1 -1
        
        for i in range(len(self.nodes)):
            for j in range(i,len(self.nodes)):
                if i != remove_id and j != remove_id:
                    node1 = self.nodes[i]
                    node2 = self.nodes[j]

                    score = -1000
                    try:
                        score = self.edges[f'{node1}_{node2}']
                    except:
                        pass
                    try:
                        score = self.edges[f'{node2}_{node1}']
                    except:
                        pass
                    if score != -1000:
                        matrixA[i,j] = 1-score
                        matrixA[j,i] = 1-score
                else:
                    if self.nodes[remove_id] == 'EGR1':
                        x = 1
                        pass
        rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=1)
        Ph = rips_complex.create_simplex_tree(max_dimension = 2).persistence()
        temp_bar = np.zeros(len(Ph),dtype=dt)

        ct = 0
        for simplex in Ph:
            dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
            if d-b > 0.00002 and d != math.inf:
                temp_bar[ct]['dim'] = dim
                temp_bar[ct]['birth'] = b
                temp_bar[ct]['death'] = d
                ct+=1
        bars = temp_bar[:ct]

        for bar in bars:
            if bar['dim'] == 0:
                features[get_inter_id(bar['death']),0] += 1
            if bar['dim'] == 1:
                features[get_inter_id(bar['birth']),1] += 1
                features[get_inter_id(bar['death']),2] += 1

        return features.flatten()

    
    def static(self,dim=1):
        matrixA = np.ones((len(self.nodes),len(self.nodes))) * 1000
        for i in range(len(self.nodes)):
            for j in range(i,len(self.nodes)):
                node1 = self.nodes[i]
                node2 = self.nodes[j]

                score = -1000
                try:
                    score = self.edges[f'{node1}_{node2}']
                except:
                    pass
                try:
                    score = self.edges[f'{node2}_{node1}']
                except:
                    pass
                if score != -1000:
                    matrixA[i,j] = 1-score
                    matrixA[j,i] = 1-score
        
        rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=2)
        complex_ = rips_complex.create_simplex_tree(max_dimension = dim)
        
        complex_.compute_persistence() 
        re = complex_.persistent_betti_numbers(2,100) 
        return np.array(re)

    
    def static_remove_node(self,remove_id,dim = 1):
        matrixA = np.ones((len(self.nodes),len(self.nodes))) * 1000
        for i in range(len(self.nodes)):
            for j in range(i,len(self.nodes)):
                if i != remove_id and j != remove_id:
                    node1 = self.nodes[i]
                    node2 = self.nodes[j]

                    score = -1000
                    try:
                        score = self.edges[f'{node1}_{node2}']
                    except:
                        pass
                    try:
                        score = self.edges[f'{node2}_{node1}']
                    except:
                        pass
                    if score != -1000:
                        matrixA[i,j] = 1-score
                        matrixA[j,i] = 1-score
        
        rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=2)
        complex_ = rips_complex.create_simplex_tree(max_dimension = dim)
        
        complex_.compute_persistence() 
        re = complex_.persistent_betti_numbers(2,100)
        return np.array(re)

    
    def get_result(self,start = 0,end = 0.3, stride = 0.03):
        
        original = self.rips(start = start,end = end, stride = stride)
        result_dic = {}
        for i,node in enumerate(self.nodes):
            
            score = np.linalg.norm(self.remove_node(i,start = start,end = end, stride = stride)-original) 
            result_dic[node] = score
            if node == 'EGR1': 

                x = 1
                pass
        return result_dic

    def get_result_static(self,dim=1):
        
        original = self.static(dim=dim)
        result_dic = {}
        for i,node in enumerate(self.nodes):
            
            score = np.linalg.norm(self.static_remove_node(i,dim=dim)-original) 
            result_dic[node] = score
            if node == 'EGR1':
                x = 1
                pass
        return result_dic
        
    def get_result_lap(self,start = 0,end = 0.3, stride = 0.03):
        
        original1 = self.rips(start = start,end = end, stride = stride)
        original2 = self.lap(start = start,end = end, stride = stride,remove_id=None)

        original = np.concatenate((original1,original2),axis=0)

        result_dic = {}
        ls = []
        ls1 = []
        ls2 = []
        for i,node in enumerate(self.nodes):
            

            fea1 = self.remove_node(i,start = start,end = end, stride = stride)
            fea2 = self.lap(start = start,end = end, stride = stride,remove_id=i)

            score1 = np.linalg.norm(fea1-original1)
            score2 = np.linalg.norm(fea2-original2)
            score = np.linalg.norm(np.concatenate((fea1,fea2),axis=0)-original)
            result_dic[node] = score
            ls1.append(score1)
            ls2.append(score2)
            ls.append(score)
            if node == 'EGR1':
                x = 1
                pass
        return result_dic,ls1,ls2,ls,list(preprocessing.minmax_scale(np.array(ls1))+preprocessing.minmax_scale(np.array(ls2)))

    def get_result_lap_statistics(self,start = 0,end = 0.3, stride = 0.03):

        original1 = self.rips(start = start,end = end, stride = stride)
        original2 = self.lap_statistics(start = start,end = end, stride = stride,remove_id=None)

        original = np.concatenate((original1,original2),axis=0)

        result_dic = {}
        ls = []
        ls1 = []
        ls2 = []
        for i,node in enumerate(self.nodes):
            

            fea1 = self.remove_node(i,start = start,end = end, stride = stride)
            fea2 = self.lap_statistics(start = start,end = end, stride = stride,remove_id=i)

            score1 = np.linalg.norm(fea1-original1) 
            score2 = np.linalg.norm(fea2-original2) 
            score = np.linalg.norm(np.concatenate((fea1,fea2),axis=0)-original) 
            result_dic[node] = score
            ls1.append(score1)
            ls2.append(score2)
            ls.append(score)
        return result_dic,ls1,ls2,ls,list(preprocessing.minmax_scale(np.array(ls1))+preprocessing.minmax_scale(np.array(ls2)))


    def get_result_lap_only_lamda(self,start = 0,end = 0.3, stride = 0.03):
        
        
        original2 = self.lap(start = start,end = end, stride = stride,remove_id=None)

        

        result_dic = {}
        for i,node in enumerate(self.nodes):
            

            
            fea2 = self.lap(start = start,end = end, stride = stride,remove_id=i)
            score = np.linalg.norm(fea2-original2)
            result_dic[node] = score
            if node == 'EGR1':
                x = 1
                pass
        return result_dic



df_new = {}
dfs = []

patp = 'E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE260711/'
compl = network_complex(patp + f'/string_interactions_short_11.5_0.4.tsv', patp + '/DEGs_seurat.csv')
df = pd.DataFrame()
df['node'] = compl.nodes
lap = compl.get_result_lap_statistics(start=0,end=1-0.4,stride=(1-0.4)/10)
    

df[f'lap_barcode_0.4'] = lap[1]
df[f'lap_lamda_0.4'] = lap[2]
df[f'lap_sum_0.4'] = lap[3]
df[f'lap_sum_norm_0.4'] = lap[4]
dfs.append(df)
merged_df = reduce(lambda left, right: pd.merge(left, right, on='node'), dfs)
merged_df.to_csv('E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE260711/ph_cocaine_norm_statistics_0.4.csv')

end = process_time()
time = (end - start)/3600
run_time = np.around(time,3)
print("run time：%.3f  h" % run_time)



