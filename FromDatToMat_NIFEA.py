import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio

from HelpFunctions import *

PHYSIONET_PATH = "../physionet.org/files/nifeadb/1.0.0/"
ALL_SIMULATED_DATA_MAT = "real_signals_nifeadb_mat"

dir_path = os.path.dirname(os.path.realpath(__file__))
physionet_path = os.path.join(dir_path,PHYSIONET_PATH)
save_mat_dir = os.path.join(dir_path,ALL_SIMULATED_DATA_MAT)


if __name__ == '__main__':
    # Transforming from dat to mat
    channels=[0,1,2,3,4]  # ch 0 is chest
    for filename in os.listdir(physionet_path):
        if not(filename.endswith(".dat")) or not "NR_" in filename:
            continue
        print(filename[:len(filename)-4])
        for i in channels:
            print(i)
            record, fields = wfdb.rdsamp(os.path.join(physionet_path,filename[:len(filename)-4]), channels=[i])
            #print(fields)
            sio.savemat(os.path.join(save_mat_dir,filename[:len(filename)-4]+'_ch'+str(i)+'.mat'),{'data': record})

