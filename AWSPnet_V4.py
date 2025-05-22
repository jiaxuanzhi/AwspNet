import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py  # Required for v7.3 MAT files
import numpy as np
import scipy.io as sio
from fightingcv_attention.attention.CBAM import CBAMBlock
from pytorch_wavelets import DTCWTForward
from torchvision import models
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
from fightingcv_attention.backbone.MobileViT import *
################################################################################
# 使用现成的预训练模型、注意力机制的块等搭建代码、能够在GPU上运行，预训练模型可以有多种选择，分割训练验证集

# 第一步：加载数据，并将标签转换为One-Hot编码
################################################################################
def load_data_and_onehot_label(
    data_path="./Data/Test_Data.mat",
    label_path="./Data/Test_Label.mat",
    test_split=0.2,
):
    """
    Load .mat files (including v7.3 format) and convert to PyTorch Tensors,
    automatically detect variable names, then split into train/test sets.note input should be complex value

    Args:
        data_path (str): Path to the training data .mat file
        label_path (str): Path to the training labels .mat file
        test_split (float): Fraction of data to use for testing (0-1)

    Returns:
        tuple: (train_dataset, test_dataset, num_classes)
    """

    def load_mat(file_path):
        try:
            # Try standard .mat loading (non-v7.3)
            mat = sio.loadmat(file_path)
            # Remove MATLAB metadata keys (like '__header__', '__version__')
            variables = [k for k in mat.keys() if not k.startswith("__")]
            if len(variables) != 1:
                raise ValueError(f"Expected 1 variable in {file_path}, found {variables}")
            return mat[variables[0]]
        except NotImplementedError:
            # Fall back to HDF5 for v7.3 files
            with h5py.File(file_path, "r") as f:
                variables = list(f.keys())
                if len(variables) != 1:
                    raise ValueError(f"Expected 1 variable in {file_path}, found {variables}")
                return np.array(f[variables[0]]).T  # Transpose for correct shape

    # Load data and labels
    data_np = load_mat(data_path)
    label_np = load_mat(label_path)

    # Handle complex data by converting to real representation (stack real and imaginary parts)
    if np.iscomplexobj(data_np):
        data_np = np.concatenate((data_np.real, data_np.imag), axis=1)  # Creates an extra dimension for real/imag parts
    elif data_np.dtype == np.void:
        # Handle MATLAB complex data stored as void type
        real_part = data_np['real'].squeeze()
        imag_part = data_np['imag'].squeeze()
        data_np = np.concatenate((real_part, imag_part), axis=1)
    elif data_np.dtype == np.dtype([('real', '<f4'), ('imag', '<f4')]):
        # Extract real and imaginary parts
        real_part = data_np['real']
        imag_part = data_np['imag']
        
        # Combine into complex array then split to real/imag components
        # complex_data = real_part + 1j * imag_part
        data_np = np.concatenate((real_part, imag_part), axis=1)
    else:
        raise ValueError(f"Unexpected data type: {data_np.dtype}. Expected structured array with real/imag fields.")
    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    label_tensor = torch.tensor(label_np.squeeze(), dtype=torch.long)
    # label_tensor = label_tensor-1
    # One-hot encoding
    num_classes = len(torch.unique(label_tensor))
    label_onehot = F.one_hot(label_tensor, num_classes=num_classes).float()
    
    # Train/test split
    dataset = TensorDataset(data_tensor, label_onehot)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, validate_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )

    return train_dataset, validate_dataset, num_classes

def Train_data_Gen(data_path="./Data/Train_Data.mat", 
                             label_path="./Data/Train_Label.mat"):
    """
    Load .mat files (including v7.3 format) and convert to PyTorch Tensors,
    automatically detect variable names, then split into train/test sets.

    Args:
        data_path (str): Path to the training data .mat file
        label_path (str): Path to the training labels .mat file
        test_split (float): Fraction of data to use for testing (0-1)

    Returns:
        tuple: (train_dataset, test_dataset, num_classes)
    """
    def load_mat(file_path):
        try:
            # Try standard .mat loading (non-v7.3)
            mat = sio.loadmat(file_path)
            # Remove MATLAB metadata keys (like '__header__', '__version__')
            variables = [k for k in mat.keys() if not k.startswith("__")]
            if len(variables) != 1:
                raise ValueError(f"Expected 1 variable in {file_path}, found {variables}")
            return mat[variables[0]]
        except NotImplementedError:
            # Fall back to HDF5 for v7.3 files
            with h5py.File(file_path, "r") as f:
                variables = list(f.keys())
                if len(variables) != 1:
                    raise ValueError(f"Expected 1 variable in {file_path}, found {variables}")
                return np.array(f[variables[0]]).T  # Transpose for correct shape

    # Load data and labels
    data_np = load_mat(data_path)
    label_np = load_mat(label_path)

    # Handle complex data by converting to real representation (stack real and imaginary parts)
    if np.iscomplexobj(data_np):
        data_np = np.concatenate((data_np.real, data_np.imag), axis=1)  # Creates an extra dimension for real/imag parts
    elif data_np.dtype == np.void:
        # Handle MATLAB complex data stored as void type
        real_part = data_np['real'].squeeze()
        imag_part = data_np['imag'].squeeze()
        data_np = np.concatenate((real_part, imag_part), axis=1)
    elif data_np.dtype == np.dtype([('real', '<f4'), ('imag', '<f4')]):
        # Extract real and imaginary parts
        real_part = data_np['real']
        imag_part = data_np['imag']
        
        # Combine into complex array then split to real/imag components
        # complex_data = real_part + 1j * imag_part
        data_np = np.concatenate((real_part, imag_part), axis=1)
    else:
        raise ValueError(f"Unexpected data type: {data_np.dtype}. Expected structured array with real/imag fields.")
    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    label_tensor = torch.tensor(label_np.squeeze(), dtype=torch.long)
    label_tensor = label_tensor-1
    # One-hot encoding
    num_classes = len(torch.unique(label_tensor))
    label_onehot = F.one_hot(label_tensor, num_classes=num_classes).float()

    # test dataset
    Train_dataset = TensorDataset(data_tensor, label_onehot)

    return Train_dataset, num_classes

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
            try:
                self.backbone = mobilevit_s(pretrained=self.pretrained)
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            except ImportError:
                raise ImportError("Please install mobilevit package: pip install mobilevit")
                
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.backbone = self.backbone.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        
        # 多个DTCWT的特征拼接
        extracted_features = []
        for xfm in self.dtcwt_list:
            Yl, Yh = xfm(x)
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

        merged_features = torch.cat(extracted_features, dim=1)

        # 应用CBAM注意力机制
        if not hasattr(self, 'attention_module'):
            big_C = merged_features.shape[1]
            self.attention_module = CBAMBlock(channel=big_C, reduction=16, kernel_size=7).to(self.device)
        
        att_features = self.attention_module(merged_features)

        # 通道降维
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
# 训练和评估函数
################################################################################
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', model_name='model'):
    # 创建保存模型的目录
    os.makedirs('saved_models', exist_ok=True)
    
    # 记录训练过程中的指标
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
            
            # 统计训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == class_idx).sum().item()
            
            running_loss += loss.item()
            
        # 计算训练集上的平均损失和准确率
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 在验证集上评估
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'saved_models/{model_name}_best.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # 保存最后一个epoch的模型
    torch.save(model.state_dict(), f'saved_models/{model_name}_last.pth')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    return model

def evaluate_model(model, test_loader, criterion=None, device='cuda'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, class_idx = torch.max(labels, dim=1)
            
            if criterion is not None:
                loss = criterion(outputs, class_idx)
                total_loss += loss.item()
            
            # 统计预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == class_idx).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(class_idx.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader) if criterion is not None else None
    accuracy = 100 * correct / total
    
    # 计算其他指标
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f'\nEvaluation Metrics:')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # 绘制混淆矩阵
    # cm = confusion_matrix(all_labels, all_preds)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.savefig('confusion_matrix.png')
    # plt.show()
    
    return avg_loss, accuracy

def predict_with_model(model, input_data, device='cuda'):
    model.eval()
    with torch.no_grad():
        input_data = input_data.to(device)
        output = model(input_data)
        _, predicted = torch.max(output.data, 1)
    return predicted.cpu().numpy()

################################################################################
# 主程序
################################################################################
if __name__ == "__main__":
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据、标签
    test_dataset, val_dataset, num_classes = load_data_and_onehot_label(test_split=0.4)
    train_dataset, num_classes = Train_data_Gen()
    # 从测试集中划分验证集
    val_size = int(0.5 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. 定义网络
    wavelet_list = [('near_sym_a', 'qshift_a'), ('near_sym_b', 'qshift_b')]
    model_type = 'resnext'  # 可选的模型类型
    
    model = DTCWT_Attention_Pretrained(
        wavelet_list=wavelet_list, 
        num_classes=num_classes, 
        J=3,
        Size=(224, 224),
        model_type=model_type,
        pretrained=True,
        device=device
    )
    
    # 3. 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 4. 训练模型
    print("Starting training...")
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=10, 
        device=device,
        model_name=f'{model_type}_wavelet_model'
    )
    
    # 5. 在测试集上评估模型
    print("\nEvaluating on test set...")
    evaluate_model(trained_model, test_loader, criterion, device)
    
    # 6. 示例预测
    print("\nExample prediction:")
    sample_data, _ = next(iter(test_loader))
    sample_input = sample_data[0:1]  # 取第一个样本
    prediction = predict_with_model(trained_model, sample_input, device)
    print(f"Predicted class: {prediction[0]}")
    
    # 7. 加载保存的最佳模型进行预测
    print("\nLoading best model for prediction...")
    best_model = DTCWT_Attention_Pretrained(
        wavelet_list=wavelet_list, 
        num_classes=num_classes, 
        J=3,
        Size=(224, 224),
        model_type=model_type,
        pretrained=False,  # 不加载预训练权重，因为我们有自己的权重
        device=device
    )
    best_model.load_state_dict(torch.load(f'./saved_models/{model_type}_wavelet_model_best.pth'))
    best_model.to(device)
    
    # 在测试集上评估最佳模型
    print("Evaluating best model on test set...")
    evaluate_model(best_model, test_loader, criterion, device)