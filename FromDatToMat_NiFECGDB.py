import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.io import loadmat
import os
import scipy.io as sio
import mne

from HelpFunctions import *

PHYSIONET_PATH = "../physionet.org/files/nifecgdb/1.0.0/"
ALL_SIMULATED_DATA_MAT = "real_signals_nifecgdb_mat"

dir_path = os.path.dirname(os.path.realpath(__file__))
physionet_path = os.path.join(dir_path,PHYSIONET_PATH)
save_mat_dir = os.path.join(dir_path,ALL_SIMULATED_DATA_MAT)


if __name__ == '__main__':
    # Transforming from dat to mat
    channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    #channels=[19,21,23]
    for filename in os.listdir(physionet_path):
                if not(filename.endswith(".edf")):
                    continue
                #print(filename[:len(filename)-4])
                path_1 = os.path.join(physionet_path, filename)
                #print(filename[:19])
                data = mne.io.read_raw_edf(os.path.join(physionet_path,filename))#, channels=[i]) #  TODO: change to channel 28
                record = data.get_data()
                # you can get the metadata included in the file and a list of all channels:
                info = data.info
                channels = data.ch_names
                for index, item in enumerate(channels):
                    print('Channel: ', item)

                    #print(fields)
                    sio.savemat(os.path.join(save_mat_dir,filename[:len(filename)-4]+str(item)+filename[19:len(filename)-4]+'.mat'),{'data': record[index]})
