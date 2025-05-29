import torch
import torch.nn as nn
import os
import numpy as np
import scipy.stats


def check_correlation(orig_signal, changed_signal):
    orig = orig_signal
    signal_to_compare = changed_signal
    correlation, p_value = scipy.stats.pearsonr(orig, signal_to_compare)
    return (
        correlation.real
    )  # the correlation of the original signal in comperison to the changed signal


def increase_sampling_rate(signal, rate):
    signal_size = len(signal)
    x = [j for j in range(signal_size)]
    y = [signal[i] for i in range(signal_size)]
    xvals = np.linspace(0, signal_size, int(signal_size * rate))
    interpolated_signal = np.interp(xvals, x, y)
    if rate >= 1:
        interpolated_signal = interpolated_signal[:signal_size]
    return interpolated_signal


def activation_func(activation):
    return nn.ModuleDict(
        [
            ["relu", nn.ReLU(inplace=True)],
            ["leaky_relu", nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ["selu", nn.SELU(inplace=True)],
            ["tanh", nn.Tanh()],
            ["none", nn.Identity()],
        ]
    )[activation]


def criterion_hinge_loss(m_feature, f_feature, delta):
    # m_normalized = torch.norm(m_feature)
    # f_normalized = torch.norm(f_feature)
    # print(f_feature.size())
    distance = nn.functional.mse_loss(m_feature, f_feature)
    # print(str(delta-distance))
    return nn.functional.relu(delta - distance)


def simulated_database_list(sim_dir):
    list = []
    signal_given = []
    for filename in os.listdir(sim_dir):
        if "mix" not in filename:
            continue
        name = filename[: (filename.find("mix"))]
        if name not in signal_given:
            signal_given.append(name)
            # for ch in [1,3,5,19,21,23]:
            for i in range(73):
                list.append(
                    [
                        name + "mix" + str(i),
                        name + "mecg" + str(i),
                        name + "fecg1" + str(i),
                    ]
                )
        else:
            continue
    return list
