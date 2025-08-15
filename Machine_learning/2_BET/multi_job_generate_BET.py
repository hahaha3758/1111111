#!/usr/bin/python
#!/bin/bash
import numpy as np
from numpy import arange
import os
import math
import numpy



####for i in range(len(CHEMBL_dataset)):
CHEMBL_dataset = ['CHEMBL216','CHEMBL1801','CHEMBL1836','CHEMBL1856','CHEMBL1995','CHEMBL2047','CHEMBL2068','CHEMBL2073','CHEMBL2083','CHEMBL2085', 'CHEMBL2107',
			'CHEMBL2147', 'CHEMBL2319', 'CHEMBL2434', 'CHEMBL2525', 'CHEMBL2593', 'CHEMBL2717' , 'CHEMBL2730' , 'CHEMBL2778' ,'CHEMBL2789' , 'CHEMBL2889' , 'CHEMBL2959', 'CHEMBL3018',
			'CHEMBL3060' , 'CHEMBL3100', 'CHEMBL3106', 'CHEMBL3119' , 'CHEMBL3194', 'CHEMBL3286', 'CHEMBL3359' ,'CHEMBL3401', 'CHEMBL3475' , 'CHEMBL3544' , 'CHEMBL3572',
			'CHEMBL3650','CHEMBL3785', 'CHEMBL3869', 'CHEMBL4029', 'CHEMBL4073', 'CHEMBL4076', 'CHEMBL4223', 'CHEMBL4306', 'CHEMBL4394', 'CHEMBL4409', 'CHEMBL4427', 'CHEMBL4462',
			'CHEMBL4506', 'CHEMBL4507', 'CHEMBL4607', 'CHEMBL4616', 'CHEMBL4691', 'CHEMBL4767', 'CHEMBL4789', 'CHEMBL4835', 'CHEMBL5247', 'CHEMBL5314', 'CHEMBL5319', 'CHEMBL5378',
			'CHEMBL5398', 'CHEMBL5480', 'CHEMBL5493', 'CHEMBL5918', 'CHEMBL6003', 'CHEMBL6007', 'CHEMBL1250348', 'CHEMBL1293255', 'CHEMBL1293293', 'CHEMBL1741179', 'CHEMBL1741186', 
			'CHEMBL1741200', 'CHEMBL1741213', 'CHEMBL2424504', 'CHEMBL3392948', 'CHEMBL3714079', 'CHEMBL3989381', 'CHEMBL4105860']
for i in range(len(CHEMBL_dataset)):
    target_ID_input = CHEMBL_dataset[i]
    feature_ID = 'BET'
    f = open('%s_%s.pbs'%(target_ID_input,feature_ID), 'w')
    f.write('#!/bin/bash\n')
    f.write('#PBS -q batch\n')
    f.write('########## Define Resources Needed with PBS Lines ##########\n')
    f.write('#PBS -l walltime=10000:00:00\n')
    f.write('#PBS -l nodes=node05:ppn=1\n')
    f.write('#PBS -l mem=2gb\n')
    f.write('#PBS -o %s_%s.out\n'%(target_ID_input,feature_ID))
    f.write('#PBS -e %s_%s.err\n'%(target_ID_input,feature_ID))
    f.write('cd /public/home/chenlong666/Chunhuanzhang/BET')
    f.write('\n')
    #f.write('module load python/conda/3.9\n')
    f.write('source /public/home/chenlong666/anaconda3/bin/activate pre-gpu\n')
    f.write('python generate_BET_feature.py %s' % (target_ID_input))
    f.close()
    cmd = 'qsub %s_%s.pbs'%(target_ID_input,feature_ID)
    os.system(cmd)
