import numpy as np
import pandas as pd
import dask.dataframe as dd
import time
import math
import os
import sys
from NeurPrc.init import ATTR
from NeurPrc.util import read_finish_signal, make_dataset

def generate(mc,rank,size) :

    file_ = open('log%d'%rank,'wt')
    print('start',file=file_)

    ######################################### NeurPrc_MC
    save_dir = ATTR.save_dir
    columns = mc.column_names
    file_num = math.ceil( mc.file_num/size )
    n_path_high = mc.n_path
    #########################################

    finish_flag = [0,0,0]
    for _ in range(file_num) :

        total_start = time.time()
        for acc,n_path in zip(["high","mid","low"],[n_path_high,int(0.1*n_path_high),int(0.01*n_path_high)]) :

            start = time.time()

            filelist = os.listdir(os.path.join("data",save_dir[acc]))
            filenum = len(filelist)
            if filenum >= file_num : 
                if   acc == "high" : finish_flag[0] = 1
                elif acc == "mid"  : finish_flag[1] = 1
                elif acc == "low"  : finish_flag[2] = 1
                if np.prod(finish_flag) == 1 :
                    print('finish',flush=True,file=file_)
                    sys.exit()
                else:
                    continue

            tmp = []

            for i in range(500) :

                ######################################### NeurPrc_MC
                x_dat,y_dat = mc(n_path)
                #########################################
                dat = np.concatenate([x_dat,y_dat],axis=1)
                tmp.append(dat)

                if read_finish_signal() : 
                    sys.exit()
                    file_.close()

                if i%25 == 24 :
                    print('%3d%%'%((i+1)/5),end=' ',flush=True,file=file_)
            print('',flush=True,file=file_)

            tmp = np.vstack(tmp)

            filename="data/%s/%s.hdf"%(save_dir[acc],"_".join([save_dir[acc],str(filenum)]))
            df = pd.DataFrame(tmp,columns=columns)
            df = dd.from_pandas(df,npartitions=10)
            df.to_hdf(filename,key="df",mode="w")

            print(filename,file=file_)

            last = time.time()
            cost_time = last-start
            print("%6.3f(s) took"%cost_time,file=file_)

        total_last = time.time()
        total_cost_time = total_last-total_start
        print("%6.3f(s) took"%total_cost_time,file=file_)
        print('*'*125,flush=True,file=file_)

    file_.close()
