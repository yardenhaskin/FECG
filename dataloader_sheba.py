import torch
import os
from torch.utils.data import Dataset
from scipy.io import loadmat

class RealDataset(Dataset):
    def __init__(self, real_dir):
        self.real_dir = real_dir
        self.real_signals = os.listdir(real_dir)

    def __len__(self):
        return len(self.real_signals)

    def __getitem__(self, idx):
        path_signal = os.path.join(self.real_dir, self.real_signals[idx])
        signal = torch.from_numpy(loadmat(path_signal)['data'])
        return signal


class RealOverfitDataset(Dataset):
    def __init__(self, real_AECG_dir, real_TECG_dir):
        self.real_AECG_dir = real_AECG_dir
        self.real_AECG_signals = sorted(os.listdir(real_AECG_dir))
        self.real_TECG_dir = real_TECG_dir
        self.real_TECG_signals = sorted(os.listdir(real_TECG_dir))

    def __len__(self):
        return len(self.real_AECG_signals)# + len(self.real_TECG_signals)

    def __getitem__(self, idx):
        #print(idx)
        path_signal_AECG = os.path.join(self.real_AECG_dir, self.real_AECG_signals[idx])
        # TODO: it was the most updated for real nifeadb
        #signal_AECG = torch.from_numpy(loadmat(path_signal_AECG)['data'])
        signal_AECG = torch.from_numpy(loadmat(path_signal_AECG)['Abdomen'])
        path_signal_TECG = os.path.join(self.real_TECG_dir, self.real_TECG_signals[idx])
        # TODO: it was the most updated for real nifeadb
        #signal_TECG = torch.from_numpy(loadmat(path_signal_TECG)['data'])
        signal_TECG = torch.from_numpy(loadmat(path_signal_TECG)['Thorax'])
        return signal_AECG, signal_TECG


class SimulatedDataset(Dataset):
    def __init__(self, simulated_dir, list):
        self.simulated_dir = simulated_dir
        self.simulated_signals = list

    def __len__(self):
        return len(self.simulated_signals)

    def __getitem__(self, idx):
        path_mix = os.path.join(self.simulated_dir, self.simulated_signals[idx][0])
        path_mecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][1])
        path_fecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][2])
        mix = torch.from_numpy(loadmat(path_mix)['data'])
        mecg = torch.from_numpy(loadmat(path_mecg)['data'])
        fecg = torch.from_numpy(loadmat(path_fecg)['data'])
        return mix, mecg, fecg