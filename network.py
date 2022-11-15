import torch.nn as nn
from torchinfo import summary as infosummary
import torch.nn.functional as F
import torch


def _3d_conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv3d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm3d(out_f),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
    )


class ResBlock(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(hidden_size)
        self.batchnorm2 = nn.BatchNorm3d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    def forward(self, x):
        return x + self.convblock(x)

# self-attention layer from https://github.com/guoyii/SACNN
class SelfAttention_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gama = nn.Parameter(torch.tensor([0.0]))

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True),
        )

    @classmethod
    def Cal_Patt(cls, k_x, q_x, v_x, N, C, D, H, W):
        k_x_flatten = k_x.reshape((N, C, D, 1, H * W))
        q_x_flatten = q_x.reshape((N, C, D, 1, H * W))
        v_x_flatten = v_x.reshape((N, C, D, 1, H * W))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 4, 3), k_x_flatten)
        r_x = F.softmax(sigma_x, dim=4)
        Patt = torch.matmul(v_x_flatten, r_x).reshape(N, C, D, H, W)
        return Patt

    @classmethod
    def Cal_Datt(cls, k_x, q_x, v_x, N, C, D, H, W):
        k_x_flatten = k_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        q_x_flatten = q_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        v_x_flatten = v_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 3, 5, 4), k_x_flatten)
        r_x = F.softmax(sigma_x, dim=5)
        Datt = torch.matmul(v_x_flatten, r_x).reshape(N, C, H, W, D)
        return Datt.permute(0, 1, 4, 2, 3)

    def forward(self, x):
        N, C, D, H, W, = x.size()
        v_x = self.conv3d_3(x)
        k_x = self.conv3d_1(x)
        q_x = self.conv3d_1(x)

        Patt = self.Cal_Patt(k_x, q_x, v_x, N, C, D, H, W)
        Datt = self.Cal_Datt(k_x, q_x, v_x, N, C, D, H, W)

        Y = self.gama * (Patt + Datt) + x
        return Y


class network_3dcnn_res_attention(nn.Module):
    def __init__(self):
        super(network_3dcnn_res_attention, self).__init__()
        self.adaptive = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((16, 64, 64))
            # nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.attention_model = SelfAttention_block(1, 1)
        self.conv_block1 = _3d_conv_block(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.conv_block2 = _3d_conv_block(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv_block3 = _3d_conv_block(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16384, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 136)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout3d(0.2)
        res_block = []
        for i in range(2):
            res_block += [ResBlock(128, 256, 128)]
        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        x = self.adaptive(x)
        x = self.attention_model(x)
        x = self.dropout(self.conv_block1(x))
        x = self.dropout(self.conv_block2(x))
        x = self.dropout(self.conv_block3(x))
        x = self.res_block(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


if __name__ == '__main__':
    batch_size = 2
    model = network_3dcnn_res_attention().cuda()
    infosummary(model, (batch_size, 1, 16, 18, 56))
