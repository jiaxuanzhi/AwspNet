import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py  # Required for v7.3 MAT files
import numpy as np
import scipy.io as sio
from fightingcv_attention.attention.CBAM import CBAMBlock # Assuming this is available
# from pytorch_wavelets import DTCWTForward # Not used in current version of DTCWT_Attention_Pretrained
from torchvision import models
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import seaborn as sns # Not used in current evaluate_model plot, will keep it commented
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
# from fightingcv_attention.backbone.MobileViT import mobilevit_s # Keep this for potential use
from pytorch_wavelets import ScatLayer,ScatLayerj2 # These are used

# Helper for MobileViT if fightingcv_attention.backbone.MobileViT.mobilevit_s is intended to be used
# Since it's commented out and causes issues if not present, I'll define a placeholder
# if it's truly needed and the library is present, the user can uncomment the import.
try:
    from fightingcv_attention.backbone.MobileViT import mobilevit_s
except ImportError:
    def mobilevit_s(pretrained=True, **kwargs): # Placeholder
        print("Warning: mobilevit_s not found, using a placeholder. Functionality will be limited.")
        # Create a dummy model that has a 'classifier' attribute for the init logic to run
        dummy_model = nn.Sequential(nn.Conv2d(3, 10, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        dummy_model.classifier = nn.Sequential(nn.Linear(10,10), nn.Linear(10,5)) # Dummy classifier
        return dummy_model

################################################################################
# 使用现成的预训练模型、注意力机制的块等搭建代码、能够在GPU上运行，预训练模型可以有多种选择，分割训练验证集
# WSN used, DWT used  TrainDatasize = 8096*1*64*481 (N, C 一般2通道，将复数转化为两个实数, H 脉冲数, W 快拍数)
# 这个代码采用了无监督对比学习的方式来训练模型，与雷达的特性不太符合，需要考虑改进为监督对比学习。
# 第一步：加载数据，并将标签转换为One-Hot编码
################################################################################
def load_data_and_onehot_label(
    data_path="./Data/Test_Data.mat",
    label_path="./Data/Test_Label.mat",
    test_split=0.2
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
        # Assuming data_np is (N, C_orig, H, W) or (N, H, W)
        # If (N, H, W), real/imag parts are (N, H, W). Stack to (N, 2, H, W)
        if data_np.ndim == 3: # (N, H, W)
            data_np = np.stack((data_np.real, data_np.imag), axis=1)
        elif data_np.ndim == 4: # (N, C_orig, H, W)
             # Concatenate along channel axis: real parts then imag parts
            data_np = np.concatenate((data_np.real, data_np.imag), axis=1) # Results in (N, 2*C_orig, H, W)
        else:
            raise ValueError(f"Unsupported complex data ndim: {data_np.ndim}")

    elif data_np.dtype == np.void and 'real' in data_np.dtype.names and 'imag' in data_np.dtype.names:
        real_part = data_np['real']
        imag_part = data_np['imag']
        # Ensure parts are correctly shaped, e.g., (N, 1, H, W) or (N, H, W)
        # If original was (N, H, W) complex, real_part might be (N, H, W)
        if real_part.ndim == 3: # (N, H, W)
            real_part = np.expand_dims(real_part, axis=1) # (N, 1, H, W)
            imag_part = np.expand_dims(imag_part, axis=1) # (N, 1, H, W)
        data_np = np.concatenate((real_part, imag_part), axis=1) # (N, 2, H, W)

    elif data_np.dtype == np.dtype([('real', '<f4'), ('imag', '<f4')]) or \
         data_np.dtype == np.dtype([('real', '<f8'), ('imag', '<f8')]):
        real_part = data_np['real']
        imag_part = data_np['imag']
        if real_part.ndim == 3: 
            real_part = np.expand_dims(real_part, axis=1)
            imag_part = np.expand_dims(imag_part, axis=1)
        data_np = np.concatenate((real_part, imag_part), axis=1)
    # else: # If data is already real and correctly formatted, no changes needed for non-complex
    #    pass # Assuming real data is already in desired shape e.g. (N,C,H,W)

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    label_tensor = torch.tensor(label_np.squeeze(), dtype=torch.long)
    
    # One-hot encoding & ensure 0-indexed labels
    min_label = torch.min(label_tensor)
    if min_label > 0: # Adjust labels to be 0-indexed if they start from 1 or higher
        label_tensor = label_tensor - min_label
    
    num_classes = len(torch.unique(label_tensor))
    if num_classes <= 0 : # Should not happen with valid labels
        raise ValueError("Number of classes is zero or negative. Check labels.")
    label_onehot = F.one_hot(label_tensor, num_classes=num_classes).float()
    
    dataset = TensorDataset(data_tensor, label_onehot)

    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    
    if train_size <= 0 and test_size <= 0 and len(dataset)>0: # Edge case if dataset is very small
        if test_split > 0.5: # Prioritize test set
            test_dataset = dataset
            train_dataset = TensorDataset(torch.empty(0, *data_tensor.shape[1:]), torch.empty(0, *label_onehot.shape[1:]))
        else: # Prioritize train set
            train_dataset = dataset
            test_dataset = TensorDataset(torch.empty(0, *data_tensor.shape[1:]), torch.empty(0, *label_onehot.shape[1:]))
        return train_dataset, test_dataset, num_classes

    if test_size == 0 :
        train_dataset = dataset
        # Create an empty dataset for validate_dataset if test_size is 0
        empty_data_sample = torch.empty(0, *data_tensor.shape[1:]) if data_tensor.numel() > 0 else torch.empty(0)
        empty_label_sample = torch.empty(0, *label_onehot.shape[1:]) if label_onehot.numel() > 0 else torch.empty(0)
        validate_dataset = TensorDataset(empty_data_sample, empty_label_sample)
    elif train_size == 0:
        validate_dataset = dataset
        empty_data_sample = torch.empty(0, *data_tensor.shape[1:]) if data_tensor.numel() > 0 else torch.empty(0)
        empty_label_sample = torch.empty(0, *label_onehot.shape[1:]) if label_onehot.numel() > 0 else torch.empty(0)
        train_dataset = TensorDataset(empty_data_sample, empty_label_sample)
    else:
        train_dataset, validate_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    return train_dataset, validate_dataset, num_classes

def Train_data_Gen(data_path="./Data/Train_Data.mat", 
                   label_path="./Data/Train_Label.mat"):
    """
    Load .mat files (including v7.3 format) and convert to PyTorch Tensors,
    automatically detect variable names for training data.
    """
    def load_mat(file_path):
        try:
            mat = sio.loadmat(file_path)
            variables = [k for k in mat.keys() if not k.startswith("__")]
            if len(variables) != 1:
                raise ValueError(f"Expected 1 variable in {file_path}, found {variables}")
            return mat[variables[0]]
        except NotImplementedError:
            with h5py.File(file_path, "r") as f:
                variables = list(f.keys())
                if len(variables) != 1:
                    raise ValueError(f"Expected 1 variable in {file_path}, found {variables}")
                return np.array(f[variables[0]]).T

    data_np = load_mat(data_path)
    label_np = load_mat(label_path)

    if np.iscomplexobj(data_np):
        if data_np.ndim == 3: 
            data_np = np.stack((data_np.real, data_np.imag), axis=1)
        elif data_np.ndim == 4: 
            data_np = np.concatenate((data_np.real, data_np.imag), axis=1)
        else:
            raise ValueError(f"Unsupported complex data ndim: {data_np.ndim}")
    elif data_np.dtype == np.void and 'real' in data_np.dtype.names and 'imag' in data_np.dtype.names:
        real_part = data_np['real']
        imag_part = data_np['imag']
        if real_part.ndim == 3: 
            real_part = np.expand_dims(real_part, axis=1)
            imag_part = np.expand_dims(imag_part, axis=1)
        data_np = np.concatenate((real_part, imag_part), axis=1)
    elif data_np.dtype == np.dtype([('real', '<f4'), ('imag', '<f4')]) or \
         data_np.dtype == np.dtype([('real', '<f8'), ('imag', '<f8')]):
        real_part = data_np['real']
        imag_part = data_np['imag']
        if real_part.ndim == 3:
            real_part = np.expand_dims(real_part, axis=1)
            imag_part = np.expand_dims(imag_part, axis=1)
        data_np = np.concatenate((real_part, imag_part), axis=1)

    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    label_tensor = torch.tensor(label_np.squeeze(), dtype=torch.long)
    
    min_label = torch.min(label_tensor)
    if min_label > 0:
        label_tensor = label_tensor - min_label
    num_classes = len(torch.unique(label_tensor))
    if num_classes <= 0:
        raise ValueError("Number of classes is zero or negative in Train_data_Gen. Check labels.")

    label_onehot = F.one_hot(label_tensor, num_classes=num_classes).float()

    Train_dataset = TensorDataset(data_tensor, label_onehot)
    return Train_dataset, num_classes

################################################################################
# 第二步：定义编码器网络 (原DTCWT_Attention_Pretrained modified)
################################################################################
class EncoderNetwork(nn.Module):
    def __init__(self, 
                 Size=(224, 224), 
                 model_type='convnext_v2',
                 pretrained=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(EncoderNetwork, self).__init__()
        
        self.device = device
        self.model_type = model_type
        self.pretrained = pretrained
        self.encoder_feature_dim = 0 # Will be set in _init_encoder_backbone
        self.target_size = Size # Store target size for adaptive pooling

        self.scat = ScatLayer().to(device)
        # self.adaptive_pool = nn.AdaptiveAvgPool2d(Size).to(device) # Moved adaptive_pool to forward if needed
        self.scat2a = ScatLayerj2().to(device)
        self._init_encoder_backbone()
        self.to(device)

    def _init_encoder_backbone(self):
        """根据选择的模型类型初始化预训练模型作为编码器 (移除分类头)"""
        if self.model_type == 'convnext_v2':
            weights = models.ConvNeXt_V2_Small_Weights.IMAGENET1K_V1 if self.pretrained else None
            model_temp = models.convnext_v2_small(weights=weights)
            self.encoder_feature_dim = model_temp.classifier[-1].in_features
            model_temp.classifier[-1] = nn.Identity()
            self.backbone = model_temp
        
        elif self.model_type == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
            model_temp = models.resnet50(weights=weights)
            self.encoder_feature_dim = model_temp.fc.in_features
            model_temp.fc = nn.Identity()
            self.backbone = model_temp
            
        elif self.model_type == 'efficientnet_v2':
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if self.pretrained else None
            model_temp = models.efficientnet_v2_s(weights=weights)
            self.encoder_feature_dim = model_temp.classifier[-1].in_features
            model_temp.classifier[-1] = nn.Identity()
            self.backbone = model_temp
            
        elif self.model_type == 'resnext':
            weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if self.pretrained else None
            model_temp = models.resnext50_32x4d(weights=weights)
            self.encoder_feature_dim = model_temp.fc.in_features
            model_temp.fc = nn.Identity()
            self.backbone = model_temp
            
        elif self.model_type == 'shufflenet':
            weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if self.pretrained else None
            model_temp = models.shufflenet_v2_x1_0(weights=weights)
            self.encoder_feature_dim = model_temp.fc.in_features
            model_temp.fc = nn.Identity()
            self.backbone = model_temp
            
        elif self.model_type == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if self.pretrained else None
            model_temp = models.mobilenet_v3_small(weights=weights)
            self.encoder_feature_dim = model_temp.classifier[-1].in_features
            model_temp.classifier[-1] = nn.Identity()
            self.backbone = model_temp
            
        elif self.model_type == 'densenet': # Using densenet121 as an example
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if self.pretrained else None
            model_temp = models.densenet121(weights=weights)
            self.encoder_feature_dim = model_temp.classifier.in_features
            model_temp.classifier = nn.Identity() # DenseNet's classifier is a single Linear layer
            self.backbone = model_temp
        
            
        elif self.model_type == 'mobilevit':
            try:
                model_temp = mobilevit_s(pretrained=self.pretrained) 
                if hasattr(model_temp, 'classifier') and isinstance(model_temp.classifier, nn.Linear):
                    self.encoder_feature_dim = model_temp.classifier.in_features
                    model_temp.classifier = nn.Identity()
                elif hasattr(model_temp, 'classifier') and isinstance(model_temp.classifier, nn.Sequential) and \
                     isinstance(model_temp.classifier[-1], nn.Linear):
                    self.encoder_feature_dim = model_temp.classifier[-1].in_features
                    model_temp.classifier[-1] = nn.Identity()
                elif hasattr(model_temp, 'head') and isinstance(model_temp.head, nn.Linear):
                    self.encoder_feature_dim = model_temp.head.in_features
                    model_temp.head = nn.Identity()
                else: # Placeholder specific logic
                    if hasattr(model_temp, 'classifier') and isinstance(model_temp.classifier, nn.Sequential) and len(model_temp.classifier) > 1 and isinstance(model_temp.classifier[-2], nn.Linear):
                         self.encoder_feature_dim = model_temp.classifier[-2].out_features 
                         model_temp.classifier[-1] = nn.Identity()
                    else:
                        self.encoder_feature_dim = 512 
                        print(f"Warning: MobileViT using placeholder or unknown structure. Defaulting encoder_feature_dim to {self.encoder_feature_dim}. The final layer of the classifier sequence might need manual adjustment to nn.Identity().")


                self.backbone = model_temp
            except ImportError:
                raise ImportError("Please install mobilevit package: pip install mobilevit or ensure fightingcv_attention is installed.")
            except Exception as e:
                raise RuntimeError(f"Error initializing MobileViT encoder: {e}")
                
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.backbone = self.backbone.to(self.device)
        if self.encoder_feature_dim == 0:
             raise ValueError(f"encoder_feature_dim was not set for model type {self.model_type}. Check _init_encoder_backbone.")

    def forward(self, x):
        x = x.to(self.device)
        Z = self.scat2a(x) 

        current_channels = Z.shape[1]
        # Ensure reduction is an int and valid for CBAMBlock
        # reduction = current_channels // 2 if current_channels > 3 else 8
        # reduction = max(1, int(reduction)) # CBAM typically expects reduction >= 1
        self.attention_module = CBAMBlock(channel=current_channels, reduction=8 , kernel_size=7).to(self.device)
        
        att_features = self.attention_module(Z)
        # att_features = Z # Uncomment this line and comment above if CBAM is not desired (as in original code)

        if not hasattr(self, 'channel_reducer') or self.channel_reducer[0].in_channels != att_features.shape[1]:
            current_channels_att = att_features.shape[1]
            if current_channels_att == 0:
                 raise ValueError("Attention features have 0 channels.")
            self.channel_reducer = nn.Sequential(
                nn.Conv2d(current_channels_att, 32, kernel_size=3, padding='same'), # Kernel 3 for wider applicability
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=3, padding='same'),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            ).to(self.device)
        
        reduced_features = self.channel_reducer(att_features)

        # Adaptive pooling if the backbone expects a fixed input size (e.g., 224x224)
        # Most backbones have their own global pooling, but this ensures compatibility.
        if reduced_features.shape[2:] != self.target_size:
            if not hasattr(self, 'adaptive_pool'):
                self.adaptive_pool = nn.AdaptiveAvgPool2d(self.target_size).to(self.device)
            features_for_backbone = self.adaptive_pool(reduced_features)
        else:
            features_for_backbone = reduced_features
        
        out_features = self.backbone(features_for_backbone)
        # Ensure output is (Batch, encoder_feature_dim)
        if out_features.ndim == 4 and out_features.shape[2] == 1 and out_features.shape[3] == 1:
            out_features = torch.flatten(out_features, 1)
        elif out_features.ndim != 2 or out_features.shape[1] != self.encoder_feature_dim:
            # This case might indicate an issue in how the backbone's classifier was replaced
            # or if the backbone doesn't output (B, F) or (B, F, 1, 1) before the Identity layer.
            print(f"Warning: Backbone output shape is {out_features.shape}, expected (B, {self.encoder_feature_dim}) or (B, {self.encoder_feature_dim}, 1, 1). Flattening.")
            out_features = torch.flatten(out_features, 1)
            # If after flatten, a Linear layer is needed to get to self.encoder_feature_dim, that's an architectural issue.
            # For now, assume the Identity replacement was correct and this flatten is a safeguard.
            if out_features.shape[1] != self.encoder_feature_dim:
                 print(f"Error: Flattened output dim {out_features.shape[1]} != encoder_feature_dim {self.encoder_feature_dim}")
                 # This would be an error. For robust execution, could add a linear layer here, but it implies a flaw in _init_encoder_backbone.
                 # For now, rely on _init_encoder_backbone being correct.
        return out_features

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.head(x.to(self.device)) # Ensure input x is on the same device

class ContrastiveLearningModel(nn.Module):
    def __init__(self, encoder: EncoderNetwork, projection_hidden_dim: int = 512, projection_output_dim: int = 128):
        super(ContrastiveLearningModel, self).__init__()
        self.encoder = encoder
        self.projection_head = ProjectionHead(
            encoder.encoder_feature_dim, 
            projection_hidden_dim, 
            projection_output_dim,
            device=encoder.device 
        )
        self.device = encoder.device
        self.to(self.device)

    def forward(self, x1, x2):
        # Encoder itself handles moving data to its device
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Projection head also handles moving data to its device in its forward
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        return z1, z2

class DownstreamClassifier(nn.Module):
    def __init__(self, encoder: EncoderNetwork, num_classes: int):
        super(DownstreamClassifier, self).__init__()
        self.encoder = encoder
        # To freeze encoder weights:
        # for param in self.encoder.parameters():
        #     param.requires_grad = False 
        self.classifier_head = nn.Linear(encoder.encoder_feature_dim, num_classes)
        self.device = encoder.device
        self.to(self.device)

    def forward(self, x):
        # x = x.to(self.device) # Encoder's forward should handle this.
        # Determine if encoder should be in eval mode if frozen, or train mode if fine-tuning
        # For simplicity, rely on model.train() or model.eval() called on DownstreamClassifier instance.
        # If encoder is frozen: self.encoder.eval() can be called once after init.
        # If fine-tuning: self.encoder.train(self.training) will be set by self.train(mode).
        
        # features = self.encoder(x) # This will use the mode set on self.encoder

        # Control gradient computation for the encoder based on its training mode (fine-tuning or frozen)
        # This ensures that if requires_grad is False for encoder params, no_grad isn't strictly needed,
        # but it's a good practice if you want to be absolutely sure for frozen parts.
        is_encoder_training = self.encoder.training
        if not any(p.requires_grad for p in self.encoder.parameters()): # If all encoder params are frozen
            with torch.no_grad():
                features = self.encoder(x)
        else: # Encoder is being fine-tuned or some parts are trainable
            self.encoder.train(mode=is_encoder_training) # Ensure encoder's mode is synced
            features = self.encoder(x)

        logits = self.classifier_head(features.to(self.device)) # Ensure features are on device for classifier_head
        return logits

################################################################################
# Contrastive Loss Function
################################################################################
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        # Using log_softmax and NLLLoss for numerical stability with CrossEntropy logic
        self.criterion = nn.CrossEntropyLoss().to(device) 

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        if batch_size == 0: return torch.tensor(0.0, device=self.device, requires_grad=True)

        z_i = F.normalize(z_i.to(self.device), p=2, dim=1)
        z_j = F.normalize(z_j.to(self.device), p=2, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels: positive pair for z_i[k] is z_j[k] (at index k+batch_size), and vice-versa
        labels = torch.arange(2 * batch_size, device=self.device)
        labels = (labels + batch_size) % (2 * batch_size) 
        
        loss = self.criterion(similarity_matrix, labels)
        return loss

################################################################################
# Training and Evaluation Functions
################################################################################

def get_augmented_views(batch_tensor: torch.Tensor, device: str, sigma: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    view1 = batch_tensor.to(device) # Ensure original is also on device
    # Generate noise on the target device, with the same shape as batch_tensor
    noise = torch.randn_like(view1, device=device) * sigma # Use view1 for randn_like to ensure consistent device and shape
    # Perform addition with tensors on the same device
    view2 = view1 + noise 
    # view1 and view2 are now on the target device.
    # The .to(device) on view2 in return is redundant but harmless.
    return view1, view2

def train_contrastive_phase(
    contrastive_model: ContrastiveLearningModel, 
    train_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    num_epochs: int, 
    device: str, 
    temperature: float = 0.1,
    model_name_prefix: str = "contrastive_encoder"
):
    os.makedirs('saved_models', exist_ok=True)
    contrastive_model.to(device) # Ensure model is on device
    criterion = NTXentLoss(temperature=temperature, device=device)
    
    train_losses = []
    print(f"Starting Contrastive Pre-training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        contrastive_model.train() 
        running_loss = 0.0
        
        for i, (inputs, _) in enumerate(train_loader): 
            # inputs are already on device from DataLoader if pin_memory=True and GPU is used,
            # but explicit .to(device) is safer if not guaranteed.
            # get_augmented_views will handle .to(device)
            inputs1, inputs2 = get_augmented_views(inputs, device) 
            
            optimizer.zero_grad()
            
            z1, z2 = contrastive_model(inputs1, inputs2) 
            loss = criterion(z1, z2)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}, batch {i+1}. Skipping update.")
                # Potentially log inputs or state for debugging
                continue # Skip backpropagation for this batch

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % (max(1, len(train_loader) // 10)) == 0: # Log ~10 times per epoch, ensure divisor is at least 1
                 print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Contrastive Loss: {loss.item():.4f}")

        if len(train_loader) > 0:
            epoch_loss = running_loss / len(train_loader)
        else:
            epoch_loss = 0.0 # Avoid division by zero if train_loader is empty
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Contrastive Loss: {epoch_loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs: 
            encoder_save_path = f'saved_models/{model_name_prefix}_epoch_{epoch+1}.pth'
            torch.save(contrastive_model.encoder.state_dict(), encoder_save_path)
            print(f"Saved contrastively trained encoder to {encoder_save_path}")

    plt.figure(figsize=(6, 5))
    plt.plot(train_losses, label='Contrastive Training Loss')
    plt.title('Contrastive Pre-training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('contrastive_training_loss.png')
    plt.close() # Close plot to free memory
    print("Contrastive pre-training finished.")
    return contrastive_model.encoder 

def train_classification_model(
    model: DownstreamClassifier, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    num_epochs: int = 10, 
    device: str = 'cuda', 
    model_name: str = 'classification_model'
):
    os.makedirs('saved_models', exist_ok=True)
    model.to(device)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    print(f"\nStarting Classification Training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        correct_train, total_train = 0, 0
        
        for i, (inputs, labels_onehot) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels_onehot = labels_onehot.to(device)
            _, class_idx = torch.max(labels_onehot, dim=1) 
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, class_idx)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected during classification training at epoch {epoch+1}, batch {i+1}. Skipping update.")
                continue
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += class_idx.size(0)
            correct_train += (predicted_train == class_idx).sum().item()

        if len(train_loader) > 0:
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct_train / total_train if total_train > 0 else 0.0
        else: # Handle empty train_loader
            train_loss = 0.0
            train_acc = 0.0

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss_epoch, val_acc_epoch = 0.0, 0.0
        if len(val_loader.dataset) > 0: # Check if val_loader has data
             val_loss_epoch, val_acc_epoch = evaluate_classification_model(model, val_loader, criterion, device, is_eval_mode=False) # is_eval_mode=False to keep model in train for BN if needed, though evaluate_classification_model sets eval mode internally.
        val_losses.append(val_loss_epoch)
        val_accs.append(val_acc_epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%')
        
        if val_acc_epoch > best_val_acc and len(val_loader.dataset) > 0 :
            best_val_acc = val_acc_epoch
            torch.save(model.state_dict(), f'saved_models/{model_name}_best.pth')
            print(f"Saved best classification model to saved_models/{model_name}_best.pth (Val Acc: {best_val_acc:.2f}%)")

    torch.save(model.state_dict(), f'saved_models/{model_name}_last.pth')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if len(val_loader.dataset) > 0: plt.plot(val_losses, label='Val Loss')
    plt.title('Classification Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    if len(val_loader.dataset) > 0: plt.plot(val_accs, label='Val Accuracy')
    plt.title('Classification Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('classification_training_metrics.png')
    plt.close()
    print("Classification training finished.")
    return model

def evaluate_classification_model(
    model: DownstreamClassifier, 
    test_loader: DataLoader, 
    criterion: Optional[nn.Module] = None, 
    device: str = 'cuda',
    is_eval_mode: bool = True # Controls if model.eval() is called
):
    if is_eval_mode: model.eval() 
    model.to(device)

    total_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels_idx = [], []
    
    if len(test_loader.dataset) == 0: # Check if dataloader is empty
        print("Evaluation loader is empty. Skipping evaluation.")
        return 0.0, 0.0 # Return default loss and accuracy

    with torch.no_grad():
        for inputs, labels_onehot in test_loader:
            inputs = inputs.to(device)
            labels_onehot = labels_onehot.to(device)
            _, class_idx = torch.max(labels_onehot, dim=1) 
            
            outputs = model(inputs) 
            
            if criterion:
                loss = criterion(outputs, class_idx)
                total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += class_idx.size(0)
            correct += (predicted == class_idx).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels_idx.extend(class_idx.cpu().numpy())
            
    avg_loss = total_loss / len(test_loader) if criterion and len(test_loader) > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    # Only print and calculate detailed metrics if not called from within training loop for val stats
    if is_eval_mode:
        precision = precision_score(all_labels_idx, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels_idx, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels_idx, all_preds, average='weighted', zero_division=0)
        
        print(f'\nClassification Model Evaluation Metrics:')
        if criterion: print(f'Average Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
    
        # Confusion Matrix
        if len(all_labels_idx) > 0 and len(all_preds) > 0 and hasattr(model.classifier_head, 'out_features'):
            try:
                import seaborn as sns 
                cm = confusion_matrix(all_labels_idx, all_preds)
                plt.figure(figsize=(max(6, model.classifier_head.out_features), max(5, model.classifier_head.out_features*0.8))) # Adjust size
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=range(model.classifier_head.out_features), 
                            yticklabels=range(model.classifier_head.out_features))
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
                plt.savefig('classification_confusion_matrix.png')
                plt.close()
            except ImportError:
                print("Seaborn not installed, skipping confusion matrix plot.")
            except Exception as e:
                print(f"Error plotting confusion matrix: {e}")

    return avg_loss, accuracy

def predict_with_classifier(model: DownstreamClassifier, input_data: torch.Tensor, device: str = 'cuda'):
    model.eval()
    model.to(device)
    input_data = input_data.to(device) # Ensure input is on device
    with torch.no_grad():
        output = model(input_data)
        _, predicted_idx = torch.max(output.data, 1)
    return predicted_idx.cpu().numpy()

################################################################################
# 主程序
################################################################################
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading training data...")
    train_dataset_full, num_classes = Train_data_Gen(
        data_path="./Data/Train_Data.mat",
        label_path="./Data/Train_Label.mat"
    )
    print(f"Full training dataset size: {len(train_dataset_full)}, Num classes: {num_classes}")

    print("Loading data for validation and testing sets...")
    # Load data from TestData_1.mat and split it into validation and test sets.
    # load_data_and_onehot_label's test_split is used here to mean "portion for the second part of its return"
    # Let's make it load all, then split manually. test_split=0 means all data goes to 'ds1', 'ds2' is empty.
    ds1, _, num_classes_val_test = load_data_and_onehot_label(
        data_path="./Data/Test_Data.mat", 
        label_path="./Data/Test_Label.mat",
        test_split=0.0 
    )
    if num_classes != num_classes_val_test:
        print(f"Warning: num_classes mismatch ({num_classes} vs {num_classes_val_test}). Using {num_classes} from main training data.")

    val_test_dataset = ds1 
    val_dataset, test_dataset = TensorDataset(torch.empty(0), torch.empty(0)), TensorDataset(torch.empty(0), torch.empty(0)) # Initialize empty

    if len(val_test_dataset) > 1 :
        val_size = int(0.5 * len(val_test_dataset))
        test_size = len(val_test_dataset) - val_size
        if val_size > 0 and test_size > 0:
            val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(42))
        elif val_size > 0 : # Only val
            val_dataset = val_test_dataset
        elif test_size > 0 : # Only test
            test_dataset = val_test_dataset
    elif len(val_test_dataset) == 1: # Only one sample, put in test
         test_dataset = val_test_dataset


    print(f"Main training set size: {len(train_dataset_full)}")
    print(f"Validation set size: {len(val_dataset.indices) if isinstance(val_dataset, torch.utils.data.Subset) else len(val_dataset)}")
    print(f"Test set size: {len(test_dataset.indices) if isinstance(test_dataset, torch.utils.data.Subset) else len(test_dataset)}")
    
    contrastive_batch_size = min(128, len(train_dataset_full) if len(train_dataset_full)>0 else 128) # Ensure batch size <= dataset size
    classification_batch_size = min(64, len(train_dataset_full) if len(train_dataset_full)>0 else 64)

    if len(train_dataset_full) == 0: raise ValueError("Main training dataset is empty.")
    
    contrastive_train_loader = DataLoader(train_dataset_full, batch_size=contrastive_batch_size, shuffle=True, drop_last=True)
    classification_train_loader = DataLoader(train_dataset_full, batch_size=classification_batch_size, shuffle=True)
    
    val_loader_bs = min(classification_batch_size, len(val_dataset) if len(val_dataset)>0 else 1)
    val_loader = DataLoader(val_dataset, batch_size=val_loader_bs, shuffle=False) if len(val_dataset) > 0 else DataLoader(TensorDataset(torch.empty(0), torch.empty(0))) # Empty loader if no val data

    test_loader_bs = min(classification_batch_size, len(test_dataset) if len(test_dataset)>0 else 1)
    test_loader = DataLoader(test_dataset, batch_size=test_loader_bs, shuffle=False) if len(test_dataset) > 0 else DataLoader(TensorDataset(torch.empty(0), torch.empty(0)))

    model_type = 'resnet50' 
    use_pretrained_encoder_weights = True 

    contrastive_epochs = 10 
    contrastive_lr = 3e-4 # Common for SimCLR
    projection_hidden_dim = 512 
    projection_output_dim = 128 
    temperature = 0.07 

    classification_epochs = 10 
    classification_lr = 1e-4

    print(f"\n--- Phase 1: Contrastive Pre-training ---")
    encoder = EncoderNetwork(
        Size=(64, 64), # Example size, adjust based on ScatLayer output and backbone needs. Original was (224,224)
        model_type=model_type,
        pretrained=use_pretrained_encoder_weights,
        device=device
    )
    print(f"Initialized EncoderNetwork with {model_type}. Backbone feature dimension: {encoder.encoder_feature_dim}")

    contrastive_model = ContrastiveLearningModel(
        encoder=encoder,
        projection_hidden_dim=projection_hidden_dim,
        projection_output_dim=projection_output_dim
    ).to(device)
    
    optimizer_contrastive = torch.optim.AdamW(contrastive_model.parameters(), lr=contrastive_lr, weight_decay=1e-6) # AdamW often better
    
    trained_encoder = train_contrastive_phase(
        contrastive_model=contrastive_model,
        train_loader=contrastive_train_loader,
        optimizer=optimizer_contrastive,
        num_epochs=contrastive_epochs,
        device=device,
        temperature=temperature,
        model_name_prefix=f"{model_type}_contrastive_encoder"
    )
    
    print(f"\n--- Phase 2: Downstream Classification ---")
    print(f"Loading best contrastively trained encoder for downstream task...")
    downstream_encoder = EncoderNetwork( # Re-init for safety, could also use trained_encoder directly
        Size=(64, 64), model_type=model_type, pretrained=False, device=device
    )
    # Use the path to the *final* saved encoder from contrastive phase
    best_encoder_path = f'saved_models/{model_type}_contrastive_encoder_epoch_{contrastive_epochs}.pth' 
    if os.path.exists(best_encoder_path):
        downstream_encoder.load_state_dict(torch.load(best_encoder_path, map_location=device))
        print(f"Loaded weights from {best_encoder_path} into new encoder instance.")
    else:
        print(f"Warning: Saved encoder weights not found at {best_encoder_path}. Using the encoder from memory (trained_encoder).")
        downstream_encoder.load_state_dict(trained_encoder.state_dict()) # Use state_dict from in-memory one

    downstream_encoder.to(device)

    classifier_model = DownstreamClassifier(
        encoder=downstream_encoder, 
        num_classes=num_classes
    ).to(device)
    
    # Fine-tuning: Set a smaller LR for the encoder, larger for the classifier head
    # Or optimize all with one LR if desired.
    # Example of differential learning rates:
    # optimizer_classification = torch.optim.AdamW([
    #         {'params': classifier_model.encoder.parameters(), 'lr': classification_lr / 10}, # Smaller LR for encoder
    #         {'params': classifier_model.classifier_head.parameters(), 'lr': classification_lr} # Larger LR for new head
    #     ], weight_decay=1e-5)
    optimizer_classification = torch.optim.AdamW(filter(lambda p: p.requires_grad, classifier_model.parameters()), lr=classification_lr, weight_decay=1e-5)
    criterion_classification = nn.CrossEntropyLoss()
    
    print("Starting downstream classification training...")
    final_trained_classifier = train_classification_model(
        model=classifier_model,
        train_loader=classification_train_loader,
        val_loader=val_loader, 
        criterion=criterion_classification,
        optimizer=optimizer_classification,
        num_epochs=classification_epochs,
        device=device,
        model_name=f'{model_type}_contrastive_downstream_classifier'
    )
    
    if len(test_loader.dataset) > 0:
        print("\nEvaluating the best classification model on the test set...")
        best_classifier_model_path = f'saved_models/{model_type}_contrastive_downstream_classifier_best.pth'
        if os.path.exists(best_classifier_model_path):
            eval_encoder = EncoderNetwork(
                 Size=(64,64), model_type=model_type, pretrained=False, device=device
            )
            if os.path.exists(best_encoder_path): # Load contrastive weights for encoder
                 eval_encoder.load_state_dict(torch.load(best_encoder_path, map_location=device))
            
            evaluation_classifier = DownstreamClassifier(encoder=eval_encoder, num_classes=num_classes).to(device)
            evaluation_classifier.load_state_dict(torch.load(best_classifier_model_path, map_location=device))
            
            evaluate_classification_model(model=evaluation_classifier, test_loader=test_loader, criterion=criterion_classification, device=device)
        else:
            print(f"Best classification model not found at {best_classifier_model_path}. Evaluating last model instead.")
            evaluate_classification_model(model=final_trained_classifier, test_loader=test_loader, criterion=criterion_classification, device=device)
    else:
        print("Test dataset is empty. Skipping evaluation on test set.")

    if len(test_loader.dataset) > 0:
        print("\nExample prediction using the final trained classification model:")
        try:
            sample_data, _ = next(iter(test_loader)) # Get a batch
            sample_input = sample_data[0:1]  # Take the first sample
            prediction_idx = predict_with_classifier(final_trained_classifier, sample_input, device)
            print(f"Sample input shape: {sample_input.shape}")
            print(f"Predicted class index for the sample: {prediction_idx[0]}")
        except StopIteration:
            print("Test loader became empty, cannot get a sample for prediction.")
    else:
        print("Test dataset is empty, cannot perform example prediction.")

    print("\nRun completed. All fixes for this round are done. You can test the code or move to the next item.")