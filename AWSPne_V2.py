import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from fightingcv_attention.attention.CBAM import CBAMBlock
# 如果尚未安装 pytorch_wavelets，请先安装：
# pip install pytorch_wavelets

from pytorch_wavelets import DTCWTForward
from torchvision import models

################################################################################
# 第一步：加载数据，并将标签转换为One-Hot编码
################################################################################
def load_data_and_onehot_label(data_path="TrainData.mat", label_path="TrainLabel.mat"):
    """
    假设 TrainData.mat 文件中包含键名 'TrainData', 形状为 (8096, 1, 64, 481)，
    TrainLabel.mat 文件中包含键名 'TrainLabel', 形状为 (8096,1)，这里的标签
    是从 0 开始的整数标签，如果不是，请自行调整。
    """

    # 加载 .mat 文件
    data_np = sio.loadmat('./Data/TrainData.mat')['AwspNet_Train_Data']  # 形状 (8096, 1, 64, 481)
    label_np = sio.loadmat('./Data/TrainLabel.mat')['AwspNet_Train_Label']

    # 取出numpy数组
    # 假设Mat文件内的键名分别是 'TrainData' 和 'TrainLabel'
    # data_np = data_mat['TrainData']  # (8096, 1, 64, 481)
    # label_np = label_mat['TrainLabel']  # (8096, 1)

    # 转换为 PyTorch Tensor
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    # 形状 (8096, 1) 转为一维 (8096,)
    label_tensor = torch.tensor(label_np.squeeze(), dtype=torch.long)  

    # 假设标签从 0 开始并连续，获取类别总数
    num_classes = len(torch.unique(label_tensor))

    # 将标签转换为 One-Hot
    # label_tensor shape：(N,) -> one_hot shape：(N, num_classes)
    label_onehot = F.one_hot(label_tensor, num_classes=num_classes).float()

    return data_tensor, label_onehot, num_classes

################################################################################
# 第二步：定义一个简单的注意力模块（示例：通道注意力 + 空间注意力）
################################################################################
class SimpleAttention(nn.Module):
    """
    这是一个示例注意力模块，可自行替换为更复杂的注意力机制。
    包含通道注意力和空间注意力，用于演示如何插入到主网络中。
    """
    def __init__(self, in_channels, reduction=4):
        super(SimpleAttention, self).__init__()
        # 通道注意力
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False) #  # 2通道输入，1通道输出，2通道是平均和最大池化的结果
        # self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # x shape: [N, C, H, W]
        n, c, h, w = x.size()

        # 1) 通道注意力
        avg_out = self.avg_pool(x).view(n, c)       # -> [N, C]
        max_out = self.max_pool(x).view(n, c)       # -> [N, C]
        channel_att = self.fc(avg_out).view(n, c, 1, 1)  # -> [N, C, 1, 1]
        channel_att2 = self.fc(max_out).view(n, c, 1, 1)  # -> [N, C, 1, 1]
        output=self.sigmoid(channel_att+channel_att2)
        x = x * output

        # 2) 空间注意力
        avg_channel = torch.mean(x, dim=1, keepdim=True)  # -> [N, 1, H, W]
        max_channel, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_channel, max_channel], dim=1)  # -> [N, 2, H, W]
        spatial_att = self.sigmoid(self.conv_spatial(spatial_in))
        x = x * spatial_att

        return x

################################################################################
# 第三步：定义使用多种小波进行DTCWT特征提取、拼接、注意力、ResNet网络的整体框架
################################################################################
class DTCWT_Attention_ResNet(nn.Module):
    def __init__(self, wavelet_list, num_classes=10, J=3, Size = (224, 224)):

        super(DTCWT_Attention_ResNet, self).__init__()

        self.wavelet_list = wavelet_list
        self.J = J

        # 定义多个DTCWTForward
        self.dtcwt_list = nn.ModuleList([
            DTCWTForward(J=J, biort=w[0], qshift=w[1]) for w in wavelet_list
        ])

        # 自适应池化层，用于标准化尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d(Size) # 后续需要替换平均池化为最大池化

        # 注意力模块（稍后在 forward 中动态定义）
        self.attention_module = None

        # 通道降维模块（稍后在 forward 中动态定义）
        self.channel_reducer = None

        # 加载预训练的ConvNeXt模型
        self.ConvNeXt = models.ConvNeXt(pretrained=True)
        # 替换最后的分类层
        in_features = self.ConvNeXt.fc.in_features
        self.ConvNeXt.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        x: 输入形状 [N, 1, 64, W]，如 W=481
        """
        # 多个DTCWT的特征拼接
        extracted_features = []
        for xfm in self.dtcwt_list:
            # Yl, Yh 为低频和高频系数
            Yl, Yh = xfm(x)  # Yl shape: [N, 1, H', W'], Yh是多个尺度的列表
            # 低频部分先保留
            feat_list = [Yl]

            # 高频 Yh: tuple, len=J，每个是 [N, 1, 6, H'', W'', 2]
            # 将 6 和 2 合并到通道维度中
            for scale in Yh:
                # scale shape: [N, 1, 6, H'', W'', 2]
                N, C, Ori, Hh, Wh, RI = scale.shape
                # 合并 ori 和 RI 到通道
                merged = scale.view(N, C * Ori * RI, Hh, Wh)  # -> [N, 1*(6*2), H'', W'']
                feat_list.append(merged)
            
            # 拼接同一个 xfm 的低频+多层高频
            # 低频维度是 [N, 1, H', W']，和高频大小不同，可以先进行自适应池化再拼
            pooled_feats = []
            for f in feat_list:
                pooled_feats.append(self.adaptive_pool(f))
            # 在通道维度拼接
            all_scales_features = torch.cat(pooled_feats, dim=1)
            extracted_features.append(all_scales_features)

        # 不同小波的结果再在通道维度拼接
        merged_features = torch.cat(extracted_features, dim=1)  # [N, big_C, 224, 224]

        # 如果第一次forward，需要根据拼接后的通道数初始化注意力模块和通道降维模块
        if self.attention_module is None:
            big_C = merged_features.shape[1]
            self.attention_module = SimpleAttention(big_C).to(merged_features.device)
            self.channel_reducer = nn.Sequential(
                nn.Conv2d(big_C, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
            ).to(merged_features.device)

        # 注意力
        att_features = self.attention_module(merged_features)  # [N, big_C, 224, 224]

        # 通道降维：big_C → 3
        reduced_features = self.channel_reducer(att_features)  # [N, 3, 224, 224]

        # 送入ResNet
        out = self.ConvNeXt(reduced_features)  # [N, num_classes]
        return out

################################################################################
# 第四步：组装并示例如何调用以上模块
################################################################################
if __name__ == "__main__":

    # 1. 加载数据、标签
    data_tensor, label_onehot, num_classes = load_data_and_onehot_label(
        data_path="TrainData.mat",
        label_path="TrainLabel.mat"
    )
    Size = data_tensor.shape
    Size = tuple(Size[-2:])
    print("Data shape:", data_tensor.shape)           # 预期 (8096, 1, 64, 481)
    print("Label one-hot shape:", label_onehot.shape) # 预期 (8096, num_classes)

    # 2. 定义网络：使用若干不同的 (biort, qshift) 参数代表不同的DTCWT母小波
    #    这些名称可以参考 pytorch_wavelets 文档   
    ''' 可替换为其他小波组合
        wavelet_list = [
        ('antonini', 'qshift_06'),
        ('legall', 'qshift_c'),
        ('near_sym_a', 'qshift_a'),
        ('near_sym_b', 'qshift_b')
    ]
    '''
    wavelet_list = [
        ('near_sym_a', 'qshift_a'),
        ('near_sym_b', 'qshift_b')
    ]
    model = DTCWT_Attention_ResNet(wavelet_list=wavelet_list, num_classes=num_classes, J=3)

    # 3. 做一个演示的前向传播
    #    取 batch_size=4 演示
    sample_data = data_tensor[:4]  # [4, 1, 64, 481]
    sample_label = label_onehot[:4]  # [4, num_classes]

    # 前向
    output = model(sample_data)
    print("Network output shape:", output.shape)  
    # 预期 [4, num_classes]

    # 4. 简要的训练示例（伪代码）
    # 假设使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 擬表示例的前向和反向
    # (真实训练需要完整的DataLoader和epoch循环，这里仅示意)
    pred = model(sample_data)
    # 注意我们当前的 label 仍是 one-hot，需要转回类别索引
    _, class_idx = torch.max(sample_label, dim=1)  # 转成类别索引
    loss = criterion(pred, class_idx)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Sample training step done. Loss:", loss.item())