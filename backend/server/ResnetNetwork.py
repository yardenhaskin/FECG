# import torch
# import torch.distributed as dist
# from ResnetBasics import *

# class ResNetEncoder(nn.Module):

#     def __init__(self, in_channels=2, activation='leaky_relu'):
#         super().__init__()

#         self.dropout1 = nn.Dropout(0.3)
#         self.dropout2 = nn.Dropout(0.3)
#         self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
#         self.batch = nn.BatchNorm1d(16)
#         self.relu = activation_func(activation)

#         self.block1 = ResNetBasicBlockEncoder(16, 32, downsampling=2)
#         self.block2 = ResNetBasicBlockEncoder(32, 32)

#         self.block3 = ResNetBasicBlockEncoder(32, 64)
#         self.block4 = ResNetBasicBlockEncoder(64, 64)

#         self.block5 = ResNetBasicBlockEncoder(64, 128, downsampling=2)
#         self.block6 = ResNetBasicBlockEncoder(128, 128)

#         self.block7 = ResNetBasicBlockEncoder(128, 256)
#         self.block8 = ResNetBasicBlockEncoder(256, 256)

#         self.block9 = ResNetBasicBlockEncoder(256, 512, downsampling=2)
#         self.block10 = ResNetBasicBlockEncoder(512, 512)

#         self.block11 = ResNetBasicBlockEncoder(512, 1024)
#         self.block12 = ResNetBasicBlockEncoder(1024, 1024)

#         self.block13 = ResNetBasicBlockEncoder(1024, 2048, downsampling=2)
#         self.block14 = ResNetBasicBlockEncoder(2048, 2048)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batch(x)
#         x = self.relu(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         # x = self.dropout1(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         x = self.block7(x)
#         # x = self.dropout2(x)
#         x = self.block8(x)
#         x = self.block9(x)
#         x = self.block10(x)
#         x = self.block11(x)
#         x = self.block12(x)
#         x = self.block13(x)
#         x = self.block14(x)
#         return x


# class ResnetDecoder(nn.Module):
#     def __init__(self, out_channels=1):
#         super().__init__()

#         self.conv_out = nn.ConvTranspose1d(16, out_channels, kernel_size=3, stride=2, padding=2, output_padding=0,
#                                            bias=False)
#         self.batch_norm = nn.BatchNorm1d(16)

#         self.block1 = ResNetBasicBlockDecoder(1024, 1024)
#         self.block2 = ResNetBasicBlockDecoder(1024, 512, upsampling=2)

#         self.block3 = ResNetBasicBlockDecoder(512, 512)
#         self.block4 = ResNetBasicBlockDecoder(512, 256)

#         self.block5 = ResNetBasicBlockDecoder(256, 256)
#         self.block6 = ResNetBasicBlockDecoder(256, 128, upsampling=2)

#         self.block7 = ResNetBasicBlockDecoder(128, 128)
#         self.block8 = ResNetBasicBlockDecoder(128, 64, upsampling=2)

#         self.block9 = ResNetBasicBlockDecoder(64, 64)
#         self.block10 = ResNetBasicBlockDecoder(64, 32, upsampling=2)

#         self.block11 = ResNetBasicBlockDecoder(32, 32)
#         self.block12 = ResNetBasicBlockDecoder(32, 16)

#     def forward(self, x):
#         one_before_last = x
#         x = self.block1(x)
#         x = self.block2(x)[:, :, :-1]
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)[:, :, :-1]
#         x = self.block7(x)
#         x = self.block8(x)
#         x = self.block9(x)
#         x = self.block10(x)[:, :, :-1]
#         x = self.block11(x)
#         x = self.block12(x)
#         x = self.batch_norm(x)
#         x = self.conv_out(x)[:, :, :-1]
#         return x, one_before_last


# class ResNet(nn.Module):
#     def __init__(self, in_channels, MECG, *args, **kwargs):
#         super().__init__()
#         self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
#         self.Mdecoder = ResnetDecoder()
#         self.Fdecoder = ResnetDecoder()
#         self.MECG = MECG

#     def forward(self, x):
#         x = self.encoder(x)
#         latent_half = x.size()[1] // 2
#         m = x[:, :latent_half, :]
#         f = x[:, latent_half:, :]

#         # Pass through decoders
#         m_hat = self.Mdecoder(m.contiguous()) + self.MECG  # Ensure m is contiguous
#         m_out, one_before_last_m = m_hat

#         f_out, one_before_last_f = self.Fdecoder(f.contiguous())  # Ensure f is contiguous

#         # Gather the tensors across processes
#         m_out = self.gather_tensor(m_out)
#         one_before_last_m = self.gather_tensor(one_before_last_m)
#         f_out = self.gather_tensor(f_out)
#         one_before_last_f = self.gather_tensor(one_before_last_f)

#         return m_out, one_before_last_m, f_out, one_before_last_f

#     def gather_tensor(self, tensor):
#         # Ensure tensor is contiguous
#         tensor = tensor.contiguous()

#         # Prepare an empty list to gather tensors across processes
#         tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]

#         # Perform the all_gather operation
#         dist.all_gather(tensors_gather, tensor)

#         # Optionally concatenate the gathered tensors along the batch dimension if needed
#         gathered = torch.cat(tensors_gather, dim=0)

#         return gathered


#######
import torch
import torch.distributed as dist
# from ResnetBasics import * # for local testing
from server.ResnetBasics import * # for docker


class ResNetEncoder(nn.Module):
    # The Encoder is made of 14 Resnet Basic Block that extracts 2048 features(channels)
    # from the input signal.
    # The output of the Encoder is a Latent Variable vector of these
    # features (both maternal and fetal)

    def __init__(self, in_channels=2, activation='leaky_relu'):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch = nn.BatchNorm1d(16)
        self.relu = activation_func(activation)

        self.block1 = ResNetBasicBlockEncoder(16, 32, downsampling=2)
        self.block2 = ResNetBasicBlockEncoder(32, 32)

        self.block3 = ResNetBasicBlockEncoder(32, 64)
        self.block4 = ResNetBasicBlockEncoder(64, 64)

        self.block5 = ResNetBasicBlockEncoder(64, 128, downsampling=2)
        self.block6 = ResNetBasicBlockEncoder(128, 128)

        self.block7 = ResNetBasicBlockEncoder(128, 256)
        self.block8 = ResNetBasicBlockEncoder(256, 256)

        self.block9 = ResNetBasicBlockEncoder(256, 512, downsampling=2)
        self.block10 = ResNetBasicBlockEncoder(512, 512)

        self.block11 = ResNetBasicBlockEncoder(512, 1024)
        self.block12 = ResNetBasicBlockEncoder(1024, 1024)

        self.block13 = ResNetBasicBlockEncoder(1024, 2048, downsampling=2)
        self.block14 = ResNetBasicBlockEncoder(2048, 2048)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)

        return x


class ResnetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        self.conv_out = nn.ConvTranspose1d(16, out_channels, kernel_size=3, stride=2, padding=2, output_padding=0,
                                           bias=False)
        self.batch_norm = nn.BatchNorm1d(16)

        self.block1 = ResNetBasicBlockDecoder(1024, 1024)
        self.block2 = ResNetBasicBlockDecoder(1024, 512, upsampling=2)

        self.block3 = ResNetBasicBlockDecoder(512, 512)
        self.block4 = ResNetBasicBlockDecoder(512, 256)

        self.block5 = ResNetBasicBlockDecoder(256, 256)
        self.block6 = ResNetBasicBlockDecoder(256, 128, upsampling=2)

        self.block7 = ResNetBasicBlockDecoder(128, 128)
        self.block8 = ResNetBasicBlockDecoder(128, 64, upsampling=2)

        self.block9 = ResNetBasicBlockDecoder(64, 64)
        self.block10 = ResNetBasicBlockDecoder(64, 32, upsampling=2)

        self.block11 = ResNetBasicBlockDecoder(32, 32)
        self.block12 = ResNetBasicBlockDecoder(32, 16)

    def forward(self, x):
        one_before_last = x
        x = self.block1(x)
        x = self.block2(x)[:, :, :-1]
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)[:, :, :-1]
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)[:, :, :-1]
        x = self.block11(x)
        x = self.block12(x)
        x = self.batch_norm(x)
        x = self.conv_out(x)[:, :, :-1]
        return x, one_before_last


class ResNet(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.Mdecoder = ResnetDecoder()
        self.Fdecoder = ResnetDecoder()

    def forward(self, x):
        # Added this to make it work on CPU
        os.environ['RANK'] = '0'  # Process rank (e.g., 0 for master process)
        os.environ['WORLD_SIZE'] = '1'  # Total number of processes
        os.environ['MASTER_ADDR'] = '127.0.0.1'  # Address for rendezvous
        os.environ['MASTER_PORT'] = '12355'  # Port for rendezvous
        dist.init_process_group(backend='gloo')

        x = self.encoder(x)
        latent_half = x.size()[1] // 2
        m = x[:, :latent_half, :]
        f = x[:, latent_half:, :]

        m_out, one_before_last_m = self.Mdecoder(m)
        f_out, one_before_last_f = self.Fdecoder(f)

        one_before_last_m = self.gather_tensor(one_before_last_m)
        one_before_last_f = self.gather_tensor(one_before_last_f)

        return m_out, one_before_last_m, f_out, one_before_last_f

    def gather_tensor(self, tensor):
        # Ensure tensor is contiguous
        tensor = tensor.contiguous()


        # Prepare an empty list to gather tensors across processes
        tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]

        # Perform the all_gather operation
        dist.all_gather(tensors_gather, tensor)

        # Optionally concatenate the gathered tensors along the batch dimension if needed
        gathered = torch.cat(tensors_gather, dim=0)

        return gathered

######
# from ResnetBasics import *
# from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler


# class ResNetEncoder(nn.Module):
# # The Encoder is made of 14 Resnet Basic Block that extracts 2048 features(channels)
# # from the input signal.
# # The output of the Encoder is a Latent Variable vector of these
# # features (both maternal and fetal)

#     def __init__(self, in_channels=2, activation='leaky_relu'):
#         super().__init__()

#         self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
#         self.batch = nn.BatchNorm1d(16)
#         self.relu = activation_func(activation)

#         self.block1 = ResNetBasicBlockEncoder(16, 32, downsampling=2)
#         self.block2 = ResNetBasicBlockEncoder(32, 32)

#         self.block3 = ResNetBasicBlockEncoder(32, 64)
#         self.block4 = ResNetBasicBlockEncoder(64, 64)

#         self.block5 = ResNetBasicBlockEncoder(64, 128, downsampling=2)
#         self.block6 = ResNetBasicBlockEncoder(128, 128)

#         self.block7 = ResNetBasicBlockEncoder(128, 256)
#         self.block8 = ResNetBasicBlockEncoder(256, 256)

#         self.block9 = ResNetBasicBlockEncoder(256, 512, downsampling=2)
#         self.block10 = ResNetBasicBlockEncoder(512, 512)

#         self.block11 = ResNetBasicBlockEncoder(512, 1024)
#         self.block12 = ResNetBasicBlockEncoder(1024, 1024)

#         self.block13 = ResNetBasicBlockEncoder(1024, 2048, downsampling=2)
#         self.block14 = ResNetBasicBlockEncoder(2048, 2048)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batch(x)
#         x = self.relu(x)

#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         x = self.block7(x)
#         x = self.block8(x)
#         x = self.block9(x)
#         x = self.block10(x)
#         x = self.block11(x)
#         x = self.block12(x)
#         x = self.block13(x)
#         x = self.block14(x)

#         return x

# class ResnetDecoder(nn.Module):
#     def __init__(self, out_channels=1):
#         super().__init__()

#         self.conv_out = nn.ConvTranspose1d(16, out_channels, kernel_size=3, stride=2, padding=2, output_padding=0,
#                                            bias=False)
#         self.batch_norm = nn.BatchNorm1d(16)

#         self.block1 = ResNetBasicBlockDecoder(1024, 1024)
#         self.block2 = ResNetBasicBlockDecoder(1024, 512, upsampling=2)

#         self.block3 = ResNetBasicBlockDecoder(512, 512)
#         self.block4 = ResNetBasicBlockDecoder(512, 256)

#         self.block5 = ResNetBasicBlockDecoder(256, 256)
#         self.block6 = ResNetBasicBlockDecoder(256, 128, upsampling=2)

#         self.block7 = ResNetBasicBlockDecoder(128, 128)
#         self.block8 = ResNetBasicBlockDecoder(128, 64, upsampling=2)

#         self.block9 = ResNetBasicBlockDecoder(64, 64)
#         self.block10 = ResNetBasicBlockDecoder(64, 32, upsampling=2)

#         self.block11 = ResNetBasicBlockDecoder(32, 32)
#         self.block12 = ResNetBasicBlockDecoder(32, 16)

#     def forward(self, x):
#         one_before_last = x
#         x = self.block1(x)
#         x = self.block2(x)[:, :, :-1]
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)[:, :, :-1]
#         x = self.block7(x)
#         x = self.block8(x)
#         x = self.block9(x)
#         x = self.block10(x)[:, :, :-1]
#         x = self.block11(x)
#         x = self.block12(x)
#         x = self.batch_norm(x)
#         x = self.conv_out(x)[:, :, :-1]
#         return x, one_before_last

# class ResNet(nn.Module):
#     def __init__(self, in_channels, *args, **kwargs):
#         super().__init__()
#         self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
#         self.Mdecoder = ResnetDecoder()
#         self.Fdecoder = ResnetDecoder()
#         # self.Mdiffusion = DiffusionModel(
#         #         net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
#         #         in_channels=4, # U-Net: number of input/output (audio) channels
#         #         channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
#         #         factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
#         #         items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
#         #         attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
#         #         attention_heads=8, # U-Net: number of attention heads per attention item
#         #         attention_features=64, # U-Net: number of attention features per attention item
#         #         diffusion_t=VDiffusion, # The diffusion method used
#         #         sampler_t=VSampler, # The diffusion sampler used
#         #     )
#         # self.Mdiffusion = DiffusionModel(
#         #     net_t=UNetV0,
#         #     in_channels=4,  # Keeping the in_channels to match your data
#         #     channels=[8, 16, 32, 64, 128],  # Reduced number of channels
#         #     factors=[1, 2, 2, 2, 2],  # Less aggressive downsampling
#         #     items=[1, 1, 1, 1, 1],  # Fewer blocks per stage
#         #     attentions=[0, 0, 0, 1, 1],  # Disable attention
#         #     attention_heads=8,  # No attention heads needed
#         #     attention_features=64,  # No attention features needed
#         #     diffusion_t=VDiffusion,  # Keeping the diffusion method
#         #     sampler_t=VSampler,  # Keeping the sampler
#         # )
#         self.Mdiffusion = DiffusionModel(
#         net_t=UNetV0,
#         in_channels=4,  # Keeping the in_channels to match your data
#         channels=[8, 32, 64, 128, 256],  # Increased number of channels
#         factors=[1, 2, 2, 2, 2],  # Less aggressive downsampling
#         items=[1, 1, 1, 1, 1],  # Fewer blocks per stage
#         attentions=[1, 1, 1, 1, 1],  # Enable attention for all stages
#         attention_heads=8,  # Number of attention heads per stage
#         attention_features=64,  # Number of attention features per head
#         diffusion_t=VDiffusion,  # Keeping the diffusion method
#         sampler_t=VSampler,  # Keeping the sampler
#     )


#         self.Fdiffusion = DiffusionModel(
#                 net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
#                 in_channels=4, # U-Net: number of input/output (audio) channels
#                 channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
#                 factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
#                 items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
#                 attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
#                 attention_heads=8, # U-Net: number of attention heads per attention item
#                 attention_features=64, # U-Net: number of attention features per attention item
#                 diffusion_t=VDiffusion, # The diffusion method used
#                 sampler_t=VSampler, # The diffusion sampler used
#             )

#     def forward(self, x):
#         # x_diff = x
#         x = self.encoder(x)
#         latent_half = x.size()[1] // 2
#         m = x[:, :latent_half, :]
#         f = x[:, latent_half:, :]

#         m_out, one_before_last_m = self.Mdecoder(m)
#         f_out, one_before_last_f = self.Fdecoder(f)

#         if torch.cuda.device_count() > 1:
#             m = self.gather_tensor(m)
#             f = self.gather_tensor(f)


#         # m_diff = m.transpose(0, 2).transpose(1, 2)
#         # f_diff = f.transpose(0, 2).transpose(1, 2)
#         # m_ouf_diff = self.Mdiffusion(m_diff)# [batch_size, in_channels, length]
#         # f_out_diff = self.Fdiffusion(f_diff)


#         return m_out, one_before_last_m, f_out, one_before_last_f, m, f

#     def gather_tensor(self, tensor):
#         tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.cuda.device_count())]
#         torch.distributed.all_gather(tensors_gather, tensor)
#         return torch.cat(tensors_gather, dim=0)
