import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from fightingcv_attention.attention.CBAM import CBAMBlock
from pytorch_wavelets import DTCWTForward
from torchvision import models
from typing import Optional
from fightingcv_attention.backbone.MobileViT import *
################################################################################
# 第一步：加载数据，并将标签转换为One-Hot编码
################################################################################
def load_data_and_onehot_label(data_path="./Data/TrainData.mat", 
                             label_path="./Data/TrainLabel.mat"):
    """
    加载.mat文件并转换为PyTorch Tensor
    """
    # 加载 .mat 文件
    data_np = sio.loadmat(data_path)
    label_np = sio.loadmat(label_path)

    # 转换为 PyTorch Tensor
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    label_tensor = torch.tensor(label_np.squeeze(), dtype=torc
                                h.long)  

    # 获取类别总数并转换为One-Hot
    num_classes = len(torch.unique(label_tensor))
    label_onehot = F.one_hot(label_tensor, num_classes=num_classes).float()

    return data_tensor, label_onehot, num_classes

################################################################################
# 第二步：定义使用多种小波进行DTCWT特征提取、拼接、注意力、预训练网络的整体框架
################################################################################
class DTCWT_Attention_Pretrained(nn.Module):
    def __init__(self, 
                 wavelet_list, 
                 num_classes=10, 
                 J=3, 
                 Size=(224, 224), 
                 model_type='convnext_v2',
                 pretrained=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(DTCWT_Attention_Pretrained, self).__init__()
        
        self.device = device
        self.wavelet_list = wavelet_list
        self.J = J
        self.model_type = model_type
        self.pretrained = pretrained

        # 定义多个DTCWTForward
        self.dtcwt_list = nn.ModuleList([
            DTCWTForward(J=J, biort=w[0], qshift=w[1]).to(device) for w in wavelet_list
        ])

        # 自适应池化层，用于标准化尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d(Size).to(device)

        # 初始化预训练模型
        self._init_pretrained_model(num_classes)
        self.to(device)

    def _init_pretrained_model(self, num_classes):
        """根据选择的模型类型初始化预训练模型"""
        if self.model_type == 'convnext_v2':
            weights = models.ConvNeXt_V2_Small_Weights.IMAGENET1K_V1 if self.pretrained else None
            self.backbone = models.convnext_v2_small(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            
        elif self.model_type == 'efficientnet_v2':
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if self.pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            
        elif self.model_type == 'resnext':
            weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if self.pretrained else None
            self.backbone = models.resnext50_32x4d(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)            
            
        elif self.model_type == 'shufflenet':
            weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if self.pretrained else None
            self.backbone = models.shufflenet_v2_x1_0(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif self.model_type == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if self.pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            
        elif self.model_type == 'mobilevit':
            # MobileViT is not in torchvision, using a custom implementation
            try:    
                self.backbone = mobilevit_s()
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            except ImportError:
                raise ImportError("Please install mobilevit package: pip install mobilevit")
                
                
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.backbone = self.backbone.to(self.device)

    def forward(self, x):
        # Move input to device
        x = x.to(self.device)
        
        # 多个DTCWT的特征拼接
        extracted_features = []
        for xfm in self.dtcwt_list:
            Yl, Yh = xfm(x)  # Yl shape: [N, 1, H', W'], Yh是多个尺度的列表
            feat_list = [Yl]

            for scale in Yh:
                N, C, Ori, Hh, Wh, RI = scale.shape
                merged = scale.view(N, C * Ori * RI, Hh, Wh)
                feat_list.append(merged)
            
            pooled_feats = []
            for f in feat_list:
                pooled_feats.append(self.adaptive_pool(f))
            all_scales_features = torch.cat(pooled_feats, dim=1)
            extracted_features.append(all_scales_features)

        # 不同小波的结果再在通道维度拼接
        merged_features = torch.cat(extracted_features, dim=1)

        # 应用CBAM注意力机制
        if not hasattr(self, 'attention_module'):
            big_C = merged_features.shape[1]
            self.attention_module = CBAMBlock(channel=big_C, reduction=8, kernel_size=7).to(self.device)
        
        att_features = self.attention_module(merged_features)

        # 通道降维：big_C → 3 (适应预训练模型的输入通道数)
        if not hasattr(self, 'channel_reducer'):
            big_C = att_features.shape[1]
            self.channel_reducer = nn.Sequential(
                nn.Conv2d(big_C, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
            ).to(self.device)
        
        reduced_features = self.channel_reducer(att_features)

        # 送入预训练模型
        out = self.backbone(reduced_features)
        return out

################################################################################
# 第四步：训练函数和主程序
################################################################################
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, class_idx = torch.max(labels, dim=1)
            loss = criterion(outputs, class_idx)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

if __name__ == "__main__":
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据、标签
    data_tensor, label_onehot, num_classes = load_data_and_onehot_label()
    print("Data shape:", data_tensor.shape)
    print("Label one-hot shape:", label_onehot.shape)

    # 2. 定义网络
    wavelet_list = [('near_sym_a', 'qshift_a'), ('near_sym_b', 'qshift_b')]
    
    # 可选的模型类型: 'convnext_v2', 'efficientnet_v2', 'resnext', 'senet', 
    # 'shufflenet', 'mobilenet_v3', 'mobilevit', 'maxvit'
    model_type = 'resnext'  
    
    model = DTCWT_Attention_Pretrained(
        wavelet_list=wavelet_list, 
        num_classes=num_classes, 
        J=3,
        Size=(224, 224),
        model_type=model_type,
        pretrained=True,
        device=device
    )
    
    # 3. 创建简单的数据加载器 (实际使用时应该更完整)
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(data_tensor, label_onehot)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True) # 如果数据集较大，可以设置更大的batch_size，但是要注意内存限制,如果GPU内存不足，可能会导致OOM错误
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)  # 使用多线程加载数据
    
    # 4. 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 5. 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=5, device=device)