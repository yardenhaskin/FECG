from ResnetNetwork import *
import torch.optim as optim
import torch.utils.data as data
from CenterLoss import *
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import math
from model import *
import dataloader
from scipy.io import loadmat

SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../AECG/simulated_signals_windows_l1to2_baseline_and_c0to2_11_Oct_22_Ch19_21_23")
LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./Models/mecg2IsShiftedBy1NotNoised")
MODELS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./Models/mecg2IsShiftedBy1NotNoised")

BATCH_SIZE = 32
#epochs = 200
epochs = 10
learning_rate = 1e-3
#LOAD_CHCKPONTS = False
LOAD_CHCKPONTS = True
BEST_MODEL_ACC = 0.3660

def main():
    pl.seed_everything(1234)
    list_simulated = simulated_database_list(SIMULATED_DATASET)

    #list_simulated_OVERFIT_REAL = list_simulated[:]  # TODO: put in comment after validating

    simulated_dataset = dataloader.SimulatedDataset(SIMULATED_DATASET,list_simulated)
    print("simulated_dataset",len(simulated_dataset))

    train_size_sim = int(0.6 * len(simulated_dataset))
    val_size_sim = int(0.2 * len(simulated_dataset))
    test_size_sim = int(0.2 * len(simulated_dataset))

    train_dataset_sim, val_dataset_sim, test_dataset_sim = torch.utils.data.random_split(simulated_dataset, [train_size_sim, val_size_sim,test_size_sim])
    # A good rule of thumb is: num_workers = 4 * num_GPU
#    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)  #num_workers=12)
    #    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)  #num_workers=12)
    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)  #num_workers=12)
    val_data_loader_sim = data.DataLoader(val_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader_sim = data.DataLoader(test_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)

    #  use gpu if available
    #device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    #  use ALL the available GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #resnet_model = ResNet(1)
    resnet_model = ResNet(2)  # 2 input channels - 1. AECG = MECG + FECG     2. TECG thoracic)

    resnet_model = nn.DataParallel(resnet_model)
    resnet_model.to(device)
    print(device)

    # if LOAD_CHCKPONTS:
    #     resnet_model.load_state_dict(torch.load(os.path.join(MODELS, "best_model")))

    optimizer_model = optim.SGD(resnet_model.parameters(), lr=learning_rate, momentum=0.9)
    if not LOAD_CHCKPONTS:
        best_model_accuracy = math.inf
    else:
        best_model_accuracy = BEST_MODEL_ACC
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr=learning_rate,amsgrad= True)
    train_loss_f_list = []
    train_loss_m_list = []
    train_loss_average_list = []
    validation_loss_f_list = []
    validation_loss_m_list = []
    validation_loss_average_list = []
    validation_corr_m_list = []
    validation_corr_f_list = []

    for epoch in range(epochs):
        #Train
        print(f"epoch number:{epoch}")
        resnet_model.train()
        train(resnet_model,
              train_data_loader_sim,
              optimizer_model,
              optimizer_centloss,
              criterion,
              criterion_cent,
              epoch,
              epochs,
              train_loss_f_list,
              train_loss_m_list,
              train_loss_average_list)

        #Evaluation
        resnet_model.eval()
        best_model_accuracy = val(val_data_loader_sim,
                                    resnet_model,
                                    criterion,
                                    epoch,
                                    epochs,
                                    validation_loss_m_list,
                                    validation_loss_f_list,
                                    validation_loss_average_list,
                                    validation_corr_m_list,
                                    validation_corr_f_list,
                                    best_model_accuracy)

    #Saving graphs training
    path_losses = os.path.join(LOSSES, "TL1M")
    np.save(path_losses, np.array(train_loss_m_list))
    path_losses = os.path.join(LOSSES, "TL1F")
    np.save(path_losses, np.array(train_loss_f_list))
    path_losses = os.path.join(LOSSES, "TL1Avg")
    np.save(path_losses, np.array(train_loss_average_list))

    #Saving graphs validation
    path_losses = os.path.join(LOSSES, "VL1M")
    np.save(path_losses, np.array(validation_loss_m_list))
    path_losses = os.path.join(LOSSES, "VL1F")
    np.save(path_losses, np.array(validation_loss_f_list))
    path_losses = os.path.join(LOSSES, "VL1Avg")
    np.save(path_losses, np.array(validation_loss_average_list))

    path_losses = os.path.join(LOSSES, "CorrM")
    np.save(path_losses, np.array(validation_corr_m_list))
    path_losses = os.path.join(LOSSES, "CorrF")
    np.save(path_losses, np.array(validation_corr_f_list))

    #Test
    test_loss_m, test_loss_f, test_loss_avg, test_corr_m, test_corr_f, test_corr_average = test(str(network_save_folder + network_file_name_best),test_data_loader_sim)

    with open("test_loss.txt", 'w') as f:
        f.write("test_loss_m = {:.4f},test_loss_f = {:.4f},test_loss_avg = {:.4f},"
                "test_corr_m = {:.4f},test_corr_f = {:.4f},test_corr_avg = {:.4f}\n".format(test_loss_m, test_loss_f, test_loss_avg,
                                                                                            test_corr_m,test_corr_f,test_corr_average))

    del resnet_model
    del simulated_dataset
    del train_data_loader_sim
    torch.cuda.empty_cache()


if __name__=="__main__":
    main()
    for filename in os.listdir(ECG_OUTPUTS_TEST):  # present the fecg outputs
        if "fecg" in filename:
            # num_of_f += 1
            path = os.path.join(ECG_OUTPUTS_TEST, filename)
            number_file = filename.index("g") + 1
            end_path = filename[number_file:]
            if int(end_path[:-4]) > 15:
                continue
            print(int(end_path[:-4]))
            fecg = os.path.join(ECG_OUTPUTS_TEST, "fecg" + end_path)
            fecg_label = os.path.join(ECG_OUTPUTS_TEST, "label_f" + end_path)
            mecg = os.path.join(ECG_OUTPUTS_TEST, "mecg" + end_path)
            mecg_label = os.path.join(ECG_OUTPUTS_TEST, "label_m" + end_path)
            ecg = os.path.join(ECG_OUTPUTS_TEST, "ecg_all" + end_path)
            real = np.load(path)
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
            ax1.plot(np.load(mecg))
            ax1.set_ylabel("MECG")
            ax2.plot(np.load(mecg_label))
            ax2.set_ylabel("MECG_LABEL")
            ax3.plot(np.load(fecg))
            ax3.set_ylabel("FECG")
            ax4.plot(np.load(fecg_label))
            ax4.set_ylabel("FECG_LABEL")
            ax5.plot(np.load(ecg))
            ax5.set_ylabel("ECG")
            save_path = os.path.join(ECG_OUTPUTS_REAL, f"plot_{filename}.png")
            plt.savefig(save_path)
            plt.show()
            plt.close()

    # for filename in os.listdir(ECG_OUTPUTS_TEST): #present the fecg outputs
    #     if "fecg" in filename:
    #         path = os.path.join(ECG_OUTPUTS_TEST, filename)
    #         number_file = filename.index("g") + 1
    #         end_path = filename[number_file:]
    #         path_label = os.path.join(ECG_OUTPUTS_TEST,"label_f" + end_path)
    #         fig, (ax1, ax2) = plt.subplots(2, 1)
    #         ax1.plot(np.load(path))
    #         ax1.set_ylabel("FECG")
    #         ax2.plot(np.load(path_label))
    #         ax2.set_ylabel("LABEL")
    #         plt.show()
    #         plt.close()
    #     if "mecg" in filename:
    #         path = os.path.join(ECG_OUTPUTS_TEST, filename)
    #         number_file = filename.index("g") + 1
    #         end_path = filename[number_file:]
    #         path_label = os.path.join(ECG_OUTPUTS_TEST, "label_m" + end_path)
    #         fig, (ax1, ax2) = plt.subplots(2, 1)
    #         ax1.plot(np.load(path))
    #         ax1.set_ylabel("MECG")
    #         ax2.plot(np.load(path_label))
    #         ax2.set_ylabel("LABEL")
    #         plt.show()
    #         plt.close()
    print("VAL")
    for filename in os.listdir(ECG_OUTPUTS_VAL): #present the fecg outputs
        if "fecg" in filename:
            # num_of_f += 1
            path = os.path.join(ECG_OUTPUTS_VAL, filename)
            number_file = filename.index("g") + 1
            end_path = filename[number_file:]
            if int(end_path[:-4]) > 5:
                continue
            fecg = os.path.join(ECG_OUTPUTS_VAL, "fecg" + end_path)
            fecg_label = os.path.join(ECG_OUTPUTS_VAL, "label_f" + end_path)
            mecg = os.path.join(ECG_OUTPUTS_VAL, "mecg" + end_path)
            mecg_label = os.path.join(ECG_OUTPUTS_VAL, "label_m" + end_path)
            ecg = os.path.join(ECG_OUTPUTS_VAL, "ecg_all" + end_path)
            real = np.load(path)
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
            plt.title("VALIDATION " + end_path[:-4])
            ax1.plot(np.load(mecg))
            ax1.set_ylabel("MECG")
            ax2.plot(np.load(mecg_label))
            ax2.set_ylabel("MECG_LABEL")
            ax3.plot(np.load(fecg))
            ax3.set_ylabel("FECG")
            ax4.plot(np.load(fecg_label))
            ax4.set_ylabel("FECG_LABEL")
            ax5.plot(np.load(ecg))
            ax5.set_ylabel("ECG")
            plt.show()
            plt.close()


        # if "fecg" in filename:
        #     path = os.path.join(ECG_OUTPUTS_VAL, filename)
        #     number_file = filename.index("g") + 1
        #     end_path = filename[number_file:]
        #     path_label = os.path.join(ECG_OUTPUTS_VAL,"label_f" + end_path)
        #     fig, (ax1, ax2) = plt.subplots(2, 1)
        #     ax1.plot(np.load(path))
        #     ax1.set_ylabel("FECG")
        #     ax2.plot(np.load(path_label))
        #     ax2.set_ylabel("LABEL")
        #     plt.show()
        #     plt.close()
        # if "mecg" in filename:
        #     path = os.path.join(ECG_OUTPUTS_VAL, filename)
        #     number_file = filename.index("g") + 1
        #     end_path = filename[number_file:]
        #     path_label = os.path.join(ECG_OUTPUTS_VAL, "label_m" + end_path)
        #     fig, (ax1, ax2) = plt.subplots(2, 1)
        #     ax1.plot(np.load(path))
        #     ax1.set_ylabel("MECG")
        #     ax2.plot(np.load(path_label))
        #     ax2.set_ylabel("LABEL")
        #     plt.show()
        #     plt.close()
    path_losses = os.path.join(LOSSES, "TL1M.npy")
    train_loss_m_list = np.load(path_losses)
    print(len(train_loss_m_list))
    path_losses = os.path.join(LOSSES, "TL1F.npy")
    train_loss_f_list = np.load(path_losses)
    print(len(train_loss_f_list))
    path_losses = os.path.join(LOSSES, "TL1Avg.npy")
    train_loss_average_list = np.load(path_losses)
    print(len(train_loss_average_list))
    path_losses = os.path.join(LOSSES, "VL1M.npy")
    validation_loss_m_list = np.load(path_losses)
    print(len(validation_loss_m_list))
    path_losses = os.path.join(LOSSES, "VL1F.npy")
    validation_loss_f_list = np.load(path_losses)
    print(len(validation_loss_f_list))
    path_losses = os.path.join(LOSSES, "VL1Avg.npy")
    validation_loss_average_list = np.load(path_losses)
    print(len(validation_loss_average_list))
    path_losses = os.path.join(LOSSES, "CorrF.npy")
    correlation_f_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "CorrM.npy")
    correlation_m_list = np.load(path_losses)
    # plotting validation and training losses and saving them
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(train_loss_m_list, label="training")
    ax1.plot(validation_loss_m_list, label="validation")
    ax1.set_ylabel("L1 M")
    ax1.set_xlabel("Epoch")
    ax2.plot(train_loss_f_list, label="training")
    ax2.plot(validation_loss_f_list, label="validation")
    ax2.set_ylabel("L1 F")
    ax2.set_xlabel("Epoch")
    ax3.plot(train_loss_average_list, label="training")
    ax3.plot(validation_loss_average_list, label="validation")
    ax3.set_ylabel("L1 Avg")
    ax3.set_xlabel("Epoch")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
    plt.close()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(correlation_f_list)
    ax1.set_ylabel("CorrF")
    ax1.set_xlabel("Epoch")
    ax2.plot(correlation_m_list)
    ax2.set_ylabel("CorrM")
    ax2.set_xlabel("Epoch")
    plt.show()
    plt.close()
