
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py  # Required for v7.3 MAT files
import numpy as np
import scipy.io as sio
from fightingcv_attention.attention.CBAM import CBAMBlock # Assuming this is available
# from pytorch_wavelets import DTCWTForward # Not used
from torchvision import models
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import seaborn as sns # Not used in current evaluate_model plot, will keep it commented
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
# from fightingcv_attention.backbone.MobileViT import mobilevit_s # Keep this for potential use
from pytorch_wavelets import ScatLayer,ScatLayerj2 # These are used
import random # Added for new augmentation

# Helper for MobileViT if fightingcv_attention.backbone.MobileViT.mobilevit_s is intended to be used
try:
    from fightingcv_attention.backbone.MobileViT import mobilevit_s
except ImportError:
    def mobilevit_s(pretrained=True, **kwargs): # Placeholder
        print("Warning: mobilevit_s not found, using a placeholder. Functionality will be limited.")
        dummy_model = nn.Sequential(nn.Conv2d(3, 10, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        dummy_model.classifier = nn.Sequential(nn.Linear(10,10), nn.Linear(10,5)) # Dummy classifier
        return dummy_model

################################################################################
# 第一步：加载数据，并将标签转换为整数索引 (不再是One-Hot)
################################################################################
def load_data_and_labels( # Renamed for clarity, was load_data_and_onehot_label
    data_path="./Data/Test_Data.mat",
    label_path="./Data/Test_Label.mat",
    test_split=0.2
):
    """
    Load .mat files (including v7.3 format) and convert to PyTorch Tensors,
    automatically detect variable names, then split into train/test sets.
    Labels are returned as integer class indices.

    Args:
        data_path (str): Path to the training data .mat file
        label_path (str): Path to the training labels .mat file
        test_split (float): Fraction of data to use for testing (0-1)

    Returns:
        tuple: (train_dataset, test_dataset, num_classes)
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
    label_tensor = torch.tensor(label_np.squeeze(), dtype=torch.long) # Integer labels
    
    min_label = torch.min(label_tensor)
    if min_label > 0: 
        label_tensor = label_tensor - min_label # Ensure 0-indexed
    
    num_classes = len(torch.unique(label_tensor))
    if num_classes <= 0 :
        raise ValueError("Number of classes is zero or negative. Check labels.")
    
    # Store integer labels directly in the dataset
    dataset = TensorDataset(data_tensor, label_tensor) 

    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    
    if train_size <= 0 and test_size <= 0 and len(dataset)>0:
        if test_split > 0.5: 
            test_dataset = dataset
            train_dataset = TensorDataset(torch.empty(0, *data_tensor.shape[1:]), torch.empty(0, dtype=torch.long))
        else: 
            train_dataset = dataset
            test_dataset = TensorDataset(torch.empty(0, *data_tensor.shape[1:]), torch.empty(0, dtype=torch.long))
        return train_dataset, test_dataset, num_classes

    if test_size == 0 :
        train_dataset = dataset
        empty_data_sample = torch.empty(0, *data_tensor.shape[1:]) if data_tensor.numel() > 0 else torch.empty(0)
        empty_label_sample = torch.empty(0, dtype=torch.long)
        validate_dataset = TensorDataset(empty_data_sample, empty_label_sample)
    elif train_size == 0:
        validate_dataset = dataset
        empty_data_sample = torch.empty(0, *data_tensor.shape[1:]) if data_tensor.numel() > 0 else torch.empty(0)
        empty_label_sample = torch.empty(0, dtype=torch.long)
        train_dataset = TensorDataset(empty_data_sample, empty_label_sample)
    else:
        train_dataset, validate_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
    return train_dataset, validate_dataset, num_classes

def Train_data_Gen_with_labels(data_path="./Data/Train_Data.mat", # Renamed for clarity
                               label_path="./Data/Train_Label.mat"):
    """
    Load .mat files for training data, returning integer labels.
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
    label_tensor = torch.tensor(label_np.squeeze(), dtype=torch.long) # Integer labels
    
    min_label = torch.min(label_tensor)
    if min_label > 0:
        label_tensor = label_tensor - min_label # Ensure 0-indexed
    num_classes = len(torch.unique(label_tensor))
    if num_classes <= 0:
        raise ValueError("Number of classes is zero or negative in Train_data_Gen. Check labels.")

    # Store integer labels directly
    Train_dataset = TensorDataset(data_tensor, label_tensor) 
    return Train_dataset, num_classes

################################################################################
# Data Augmentation Functions (for Supervised Contrastive Learning)
################################################################################
def horizontal_shift_tensor(tensor_batch: torch.Tensor, max_shift_percent: float = 0.1, padding_value: float = 0.0) -> torch.Tensor:
    """
    Applies random horizontal shift with specified padding to a batch of tensors.
    Args:
        tensor_batch (torch.Tensor): Input batch of shape (N, C, H, W).
        max_shift_percent (float): Maximum shift as a percentage of width.
        padding_value (float): Value used for padding.
    Returns:
        torch.Tensor: Horizontally shifted batch.
    """
    n, c, h, w = tensor_batch.shape
    shifted_batch = torch.full_like(tensor_batch, padding_value)
    
    for i in range(n):
        max_pixels = int(w * max_shift_percent)
        # Ensure max_pixels is at least 0, can be negative if w or max_shift_percent is 0
        if max_pixels < 0: max_pixels = 0

        if max_pixels == 0: # Handle case where no shift is possible or desired
            shift_pixels = 0
        else:
            shift_pixels = random.randint(-max_pixels, max_pixels)

        if shift_pixels == 0:
            shifted_batch[i] = tensor_batch[i]
        elif shift_pixels > 0:  # Shift content right, pad left
            shifted_batch[i, :, :, shift_pixels:] = tensor_batch[i, :, :, :-shift_pixels]
        else:  # Shift content left (shift_pixels is negative), pad right
            shifted_batch[i, :, :, :w + shift_pixels] = tensor_batch[i, :, :, -shift_pixels:]
            
    return shifted_batch

def get_supcon_augmented_views(batch_tensor: torch.Tensor, 
                               device: str, 
                               noise_sigma: float = 0.2, 
                               shift_percent: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates two augmented views for supervised contrastive learning.
    View 1: Original + Gaussian Noise.
    View 2: Original + Horizontal Shift with zero-padding.
    """
    batch_tensor_on_device = batch_tensor.to(device)

    # View 1: Gaussian Noise
    noise = torch.randn_like(batch_tensor_on_device) * noise_sigma
    view1 = batch_tensor_on_device + noise

    # View 2: Horizontal Shift
    # Ensure batch_tensor_on_device is passed, as horizontal_shift_tensor expects a tensor
    view2 = horizontal_shift_tensor(batch_tensor_on_device, max_shift_percent=shift_percent, padding_value=0.0)
    
    return view1, view2

################################################################################
# Encoder Network 
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
        self.encoder_feature_dim = 0 
        self.target_size = Size 

        self.scat2a = ScatLayerj2().to(device) # User switched to this version
        
        # Initialize attention_module and channel_reducer with placeholder/default configurations
        # This ensures the module names exist when load_state_dict is called.
        # The forward pass will re-initialize them correctly based on actual input dimensions.
        
        # Placeholder for channels after scat2a. 
        # This should be a value that ScatLayerj2 might output or a small valid default.
        # For example, if ScatLayerj2 typically outputs many channels, start with a reasonable guess
        # or a minimum valid channel count for CBAM (e.g., CBAM might need >=1 or >= some_value_for_reduction).
        # Let's use a small, plausible number of channels, e.g., 3 (common in image processing).
        # Adjust if ScatLayerj2 has a known default or minimum output channel count.

        # initial_scat_output_channels = 14 
        # # Configure placeholder attention_module
        # # Use the reduction logic from your forward pass for consistency
        # reduction_placeholder_attn = initial_scat_output_channels // 2 if initial_scat_output_channels > 16 else 8
        # reduction_placeholder_attn = max(1, int(reduction_placeholder_attn))
        
        # self.attention_module = CBAMBlock(
        #     channel=initial_scat_output_channels, 
        #     reduction=reduction_placeholder_attn, 
        #     kernel_size=7
        # ).to(device)
        # self.attention_module_configured_channels = initial_scat_output_channels

        # # Placeholder for channels after attention_module (CBAM outputs same channels as input)
        # initial_attn_output_channels = initial_scat_output_channels
        
        # # Configure placeholder channel_reducer
        # self.channel_reducer = nn.Sequential(
        #     nn.Conv2d(initial_attn_output_channels, 32, kernel_size=3, padding='same'), 
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 3, kernel_size=3, padding='same'),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(inplace=True),
        # ).to(device)
        # self.channel_reducer_input_channels = initial_attn_output_channels
        
        # Initialize adaptive_pool here if its size is fixed and known at init
        # Or keep it in forward if self.target_size can change (though it's from __init__)
        # For consistency with the forward pass logic, it can be initialized here as well.
        # self.adaptive_pool = nn.AdaptiveAvgPool2d(self.target_size).to(self.device)
        # However, the current forward pass initializes it on demand if shapes don't match,
        # which is fine. Adding it here means its state would also be in state_dict.
        # For this specific error, only attention_module and channel_reducer are the issue.

        self._init_encoder_backbone() # Initializes self.backbone and self.encoder_feature_dim
        self.to(device) # Move the entire EncoderNetwork to the specified device
    
    # The _init_encoder_backbone method remains the same.
    # The forward method (with its conditional re-initialization logic for 
    # attention_module and channel_reducer) remains the same.

    def _init_encoder_backbone(self):
        """Initializes the backbone model as an encoder, removing the final classification layer."""
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
                elif hasattr(model_temp, 'head') and isinstance(model_temp.head, nn.Linear): # Common in ViT models
                    self.encoder_feature_dim = model_temp.head.in_features
                    model_temp.head = nn.Identity()
                else: 
                    if hasattr(model_temp, 'classifier') and isinstance(model_temp.classifier, nn.Sequential) and \
                       len(model_temp.classifier) > 1 and isinstance(model_temp.classifier[-2], nn.Linear):
                         self.encoder_feature_dim = model_temp.classifier[-2].out_features 
                         model_temp.classifier[-1] = nn.Identity()
                    else: # Fallback for placeholder or unknown structure
                        self.encoder_feature_dim = 512 
                        print(f"Warning: MobileViT ('{self.model_type}') using placeholder or unknown classifier structure. Defaulting encoder_feature_dim to {self.encoder_feature_dim}.")
                self.backbone = model_temp
            except ImportError:
                raise ImportError("MobileViT not available. Please install fightingcv_attention or choose another model_type.")
            except Exception as e:
                raise RuntimeError(f"Error initializing MobileViT encoder: {e}")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.backbone = self.backbone.to(self.device)
        if self.encoder_feature_dim == 0:
             raise ValueError(f"encoder_feature_dim was not set for model type {self.model_type}. Check _init_encoder_backbone.")

    def forward(self, x):
        x = x.to(self.device)
        Z = self.scat2a(x) # Using ScatLayerj2 as per user's previous implication

        # Conditional initialization for attention_module
        if not hasattr(self, 'attention_module') or \
           not hasattr(self, 'attention_module_configured_channels') or \
           self.attention_module_configured_channels != Z.shape[1]:
            current_channels_Z = Z.shape[1]
            if current_channels_Z == 0: 
                raise ValueError("Scattering layer (scat2a) output has 0 channels.")
            
            # Determine reduction factor for CBAM
            reduction = current_channels_Z // 2 if current_channels_Z > 16 else 8 # Example logic for reduction
            reduction = max(1, int(reduction)) 
            
            self.attention_module = CBAMBlock(channel=current_channels_Z, reduction=reduction, kernel_size=7).to(self.device)
            self.attention_module_configured_channels = current_channels_Z # Store configured channels
        
        att_features = self.attention_module(Z)

        # Conditional initialization for channel_reducer
        if not hasattr(self, 'channel_reducer') or \
           not hasattr(self, 'channel_reducer_input_channels') or \
           self.channel_reducer_input_channels != att_features.shape[1]:
            current_channels_att = att_features.shape[1]
            if current_channels_att == 0:
                 raise ValueError("Attention features have 0 channels.")
            self.channel_reducer = nn.Sequential(
                nn.Conv2d(current_channels_att, 32, kernel_size=3, padding='same'), 
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=3, padding='same'),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            ).to(self.device)
            self.channel_reducer_input_channels = current_channels_att # Store configured channels
        
        reduced_features = self.channel_reducer(att_features)

        # Adaptive pooling to target size if needed
        if reduced_features.shape[2:] != self.target_size:
            if not hasattr(self, 'adaptive_pool'): # Initialize adaptive_pool on demand
                self.adaptive_pool = nn.AdaptiveAvgPool2d(self.target_size).to(self.device)
            features_for_backbone = self.adaptive_pool(reduced_features)
        else:
            features_for_backbone = reduced_features
        
        out_features = self.backbone(features_for_backbone)
        
        # Ensure output is 2D: (Batch, encoder_feature_dim)
        if out_features.ndim == 4: # Common for CNNs before flatten
            out_features = torch.flatten(out_features, 1) 
        
        # if out_features.ndim != 2 or out_features.shape[1] != self.encoder_feature_dim:
        #     print(f"Warning: Output features dim {out_features.shape} after backbone and flatten "
        #           f"does not match expected encoder_feature_dim {self.encoder_feature_dim}. "
        #           f"This may cause issues. Attempting to reshape or re-evaluate encoder structure.")
        #     # Attempting a more aggressive flatten if still not matching (e.g. ViT might output [B, N_patches, D_hidden])
        #     # This is a fallback; ideally, backbone modification in _init_encoder_backbone is perfect.
        #     if out_features.ndim > 2:
        #         out_features = torch.flatten(out_features, 1)
            
        #     # If still not matching, this is a critical issue.
        #     if out_features.shape[1] != self.encoder_feature_dim:
        #          print(f"CRITICAL Warning: Final output dim {out_features.shape[1]} still != encoder_feature_dim {self.encoder_feature_dim}.")
        #          # As a last resort, if a linear projection is missing to get to encoder_feature_dim,
        #          # it implies an issue in _init_encoder_backbone. Forcing it here is a patch:
        #          # if not hasattr(self, 'final_projection_to_encoder_dim'):
        #          #     self.final_projection_to_encoder_dim = nn.Linear(out_features.shape[1], self.encoder_feature_dim).to(self.device)
        #          # out_features = self.final_projection_to_encoder_dim(out_features)
        #          # However, this is not ideal and indicates a setup problem. For now, we rely on previous steps being correct.

        return out_features

################################################################################
# Projection Head (for Supervised Contrastive Learning)
################################################################################
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        self.device = device
        self.to(self.device)

    def forward(self, x):
        # Input x should already be on self.device if coming from an encoder on the same device
        return self.head(x)

################################################################################
# Model for Supervised Contrastive Learning (Encoder + Projection Head)
################################################################################
class ContrastiveLearningModel(nn.Module):
    def __init__(self, encoder: EncoderNetwork, projection_hidden_dim: int = 512, projection_output_dim: int = 256):
        super(ContrastiveLearningModel, self).__init__()
        self.encoder = encoder # This encoder instance will be trained
        # The projection head is part of the SupCon training, but NOT used by ProtoNet directly
        self.projection_head = ProjectionHead(
            encoder.encoder_feature_dim, 
            projection_hidden_dim, 
            projection_output_dim,
            device=encoder.device 
        )
        self.device = encoder.device
        self.to(self.device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes two augmented views through encoder and projection head.
        Args:
            x1: First batch of augmented views.
            x2: Second batch of augmented views.
        Returns:
            Tuple of projected features (z1, z2).
        """
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        return z1, z2

################################################################################
# Supervised Contrastive Loss Function
################################################################################
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode # 'all' or 'one'
        self.base_temperature = base_temperature # Typically same as temperature
        self.device = device

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: Hidden_vector of shape [bsz, n_views, ...]. Assumed to be L2 normalized. n_views
            labels: Ground truth of shape [bsz]. Ignored if mask is provided.
            mask: Contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                is positive for sample i.
        Returns:
            A loss scalar.
        """
        features = features.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             ' e.g. [batch_size, 2, feature_dim].')
        if len(features.shape) > 3: # Flatten feature dimensions if more than 1D
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        
        # Create mask for positive pairs based on labels
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features (batch_size)')
            # mask[i, j] is 1 if sample i and sample j have the same label
            mask = torch.eq(labels, labels.T).float() 
        elif mask is None: # Unsupervised case (SimCLR)
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        # If mask is provided, use it directly (already on device)

        n_views = features.shape[1]
        # Concatenate all features from different views: (bsz * n_views, feature_dim)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature # Each view of each image is an anchor
            anchor_count = n_views
        elif self.contrast_mode == 'one': # Use only the first view of each image as an anchor
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:
            raise ValueError('Unknown contrast_mode: {}'.format(self.contrast_mode))

        # Compute similarity: anchor_feature @ contrast_feature.T
        # anchor_dot_contrast will be (bsz * anchor_count, bsz * n_views)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # For numerical stability in softmax
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile the original mask to match the shape of logits
        # Original mask: (bsz, bsz)
        # Tiled mask: (bsz * anchor_count, bsz * n_views)
        mask = mask.repeat(anchor_count, n_views)
        
        # Mask out self-contrast cases (diagonal elements in the sim matrix of all views)
        # logits_mask is 1 everywhere except diagonal
        # logits_mask = torch.ones_like(mask) - torch.eye(batch_size * anchor_count, device=self.device).repeat(
        #     1, n_views // anchor_count if n_views > anchor_count else 1)[:,:batch_size*n_views] # Ensure correct shape for eye
        # if logits_mask.shape != mask.shape: # More robust eye creation for all cases
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size * anchor_count, device=self.device).view(-1,1), 0)


        mask = mask * logits_mask # Positive pairs, excluding self-similarity with the exact same view instance

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # Denominator sum, zero out self-contrasts
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # Add small epsilon

        # Compute mean of log-likelihood over positive pairs
        # mask.sum(1) is the number of positive pairs for each anchor
        # Add epsilon to avoid division by zero if an anchor has no positive pairs (other than itself)
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Final loss (negative mean log-likelihood)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # Average loss over all anchors in the batch
        loss = loss.view(anchor_count, batch_size).mean() 

        return loss

################################################################################
# Training Function for Supervised Contrastive Encoder
################################################################################
def train_supervised_contrastive_phase(
    contrastive_model: ContrastiveLearningModel, 
    train_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    num_epochs: int, 
    device: str, 
    temperature: float = 0.1, # temperature for SupCon loss
    model_name_prefix: str = "supcon_encoder"
):
    os.makedirs('saved_models', exist_ok=True)
    contrastive_model.to(device)
    criterion = SupConLoss(temperature=temperature, device=device) 
    
    train_losses = []
    print(f"Starting Supervised Contrastive Pre-training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        contrastive_model.train() 
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader): # labels are integer class indices
            inputs = inputs.to(device) 
            labels = labels.to(device)

            inputs1, inputs2 = get_supcon_augmented_views(inputs, device) # Generate two augmented views
            
            optimizer.zero_grad()
            
            # Get projected features. ContrastiveLearningModel's forward can be used if it returns (z1, z2)
            # Or, call encoder and projection_head explicitly if more control is needed.
            # The SupConLoss expects features in shape [bsz, n_views, f_dim].
            h1 = contrastive_model.encoder(inputs1)
            h2 = contrastive_model.encoder(inputs2)
            z1 = contrastive_model.projection_head(h1)
            z2 = contrastive_model.projection_head(h2)
            
            # Normalize features (often done before SupConLoss, though SupCon can also handle it)
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)

            # Combine features for SupConLoss: [bsz, n_views, feature_dim]
            features_for_loss = torch.stack([z1, z2], dim=1) # Shape: (bsz, 2, projection_output_dim)
            
            loss = criterion(features_for_loss, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf SupCon loss detected at epoch {epoch+1}, batch {i+1}. Loss: {loss.item()}. Skipping update.")
                continue

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % (max(1, len(train_loader) // 10)) == 0:
                 print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], SupCon Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average SupCon Loss: {epoch_loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs: 
            encoder_save_path = f'saved_models/{model_name_prefix}_epoch_{epoch+1}.pth'
            torch.save(contrastive_model.encoder.state_dict(), encoder_save_path) # Save only the encoder
            print(f"Saved supervised contrastively trained encoder to {encoder_save_path}")

    plt.figure(figsize=(8, 6)) # Adjusted figure size
    plt.plot(train_losses, label='Supervised Contrastive Training Loss')
    plt.title('Supervised Contrastive Pre-training Loss Curve') # More descriptive title
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss') # More descriptive label
    plt.grid(True) # Add grid for better readability
    plt.legend()
    plt.savefig('supcon_training_loss.png')
    plt.close()
    print("Supervised contrastive pre-training finished.")
    return contrastive_model.encoder # Return the trained encoder module

################################################################################
# Prototypical Network for Classification
################################################################################
class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder: EncoderNetwork, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
        self.encoder.eval() # Encoder is typically frozen for prototypical networks
        for param in self.encoder.parameters(): # Explicitly set requires_grad to False
            param.requires_grad = False
            
        self.device = device
        self.prototypes = None # Shape: [num_classes, encoder_feature_dim]
        self.prototype_class_labels = None # Stores integer labels corresponding to each prototype row
        self.to(self.device)

    def compute_prototypes(self, support_loader: DataLoader, num_total_classes: int):
        """
        Computes class prototypes from the support set (e.g., training data).
        Args:
            support_loader: DataLoader for the support set, yielding (data, integer_labels).
            num_total_classes: The total number of unique classes in the dataset (e.g., from Train_data_Gen).
        """
        self.encoder.eval() 
        print("Computing prototypes for Prototypical Network...")
        
        all_features_list = []
        all_labels_list = [] # Integer labels
        
        with torch.no_grad():
            for inputs, labels_idx in support_loader:
                inputs = inputs.to(self.device)
                # labels_idx are already integer class indices
                features = self.encoder(inputs)
                all_features_list.append(features.cpu()) 
                all_labels_list.append(labels_idx.cpu())
        
        all_features = torch.cat(all_features_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        feature_dim = all_features.shape[1]
        # Initialize prototypes tensor for all potential classes
        # This ensures consistent indexing if some classes are missing in the current support_loader batch
        # but were present in the overall dataset definition (num_total_classes)
        self.prototypes = torch.zeros(num_total_classes, feature_dim).to(self.device)
        prototype_counts = torch.zeros(num_total_classes, dtype=torch.long).to(self.device)

        unique_present_labels = torch.unique(all_labels)
        self.prototype_class_labels = unique_present_labels.to(self.device) # Store labels for which prototypes are actually computed

        # It's better to create prototypes only for classes present in the support set if num_total_classes is an upper bound.
        # Or, if num_total_classes is exact, iterate 0 to num_total_classes-1.
        # Assuming labels are 0 to num_total_classes-1.
        
        computed_prototypes_list = []
        computed_prototype_labels_list = []

        for i in range(num_total_classes): # Iterate through all possible class indices
            class_mask = (all_labels == i)
            if class_mask.any(): 
                class_features = all_features[class_mask].to(self.device)
                prototype = class_features.mean(dim=0)
                self.prototypes[i] = prototype # Store at index i
                prototype_counts[i] = class_features.shape[0]
                computed_prototypes_list.append(prototype.unsqueeze(0))
                computed_prototype_labels_list.append(torch.tensor([i], device=self.device))


        if not computed_prototypes_list: # No prototypes computed (e.g., empty support_loader)
            print("Warning: No prototypes were computed. Support set might be empty or lack labels.")
            self.prototypes = torch.empty(0, feature_dim).to(self.device) # Empty prototypes
            self.prototype_class_labels = torch.empty(0, dtype=torch.long).to(self.device)
        else:
            # If you want prototypes to be dense only for existing classes:
            # self.prototypes = torch.cat(computed_prototypes_list, dim=0)
            # self.prototype_class_labels = torch.cat(computed_prototype_labels_list, dim=0)
            # For now, using the fixed-size self.prototypes indexed by class label 0..N-1
            pass


        print(f"Prototypes computed. Shape: {self.prototypes.shape if self.prototypes is not None else 'None'}")
        for i in range(num_total_classes):
            if prototype_counts[i] == 0:
                 print(f"Warning: Class {i} had 0 samples in support set for prototype computation. Its prototype is zero vector.")


    def forward(self, query_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.prototypes is None or self.prototypes.nelement() == 0: # Check if prototypes are valid
            raise RuntimeError("Prototypes are not computed or are empty. Call compute_prototypes() with a valid support_loader.")
        
        self.encoder.eval()
        query_x = query_x.to(self.device)
        
        with torch.no_grad():
            query_features = self.encoder(query_x) # (N_query, D_feat)
        
        # Calculate squared Euclidean distances: (a-b)^2 = a^2 - 2ab + b^2
        # query_features: (N_query, D_feat)
        # self.prototypes: (N_classes_with_prototypes, D_feat)
        qf_sq = torch.sum(query_features**2, dim=1, keepdim=True) # (N_query, 1)
        p_sq = torch.sum(self.prototypes**2, dim=1, keepdim=True).t() # (1, N_classes_with_prototypes)
        dot_prod = torch.matmul(query_features, self.prototypes.t()) # (N_query, N_classes_with_prototypes)
        
        dists = qf_sq - 2 * dot_prod + p_sq # (N_query, N_classes_with_prototypes)
        dists = torch.clamp(dists, min=0.0) # Ensure non-negative distances

        # Convert distances to probabilities (closer distance = higher probability)
        # Using -dists as logits for softmax. Higher similarity (lower distance) gives higher logit.
        log_probas = F.log_softmax(-dists, dim=1) 
        all_class_probabilities = torch.exp(log_probas) # (N_query, N_classes_with_prototypes)

        # Get the probability of the predicted class and the predicted class label
        predicted_max_probabilities, predicted_label_indices_relative_to_prototypes = torch.max(all_class_probabilities, dim=1)
        
        # If self.prototypes is indexed 0..N-1 directly by class label:
        predicted_actual_labels = predicted_label_indices_relative_to_prototypes
        # If self.prototype_class_labels was used to store actual labels for a dense prototype tensor:
        # predicted_actual_labels = self.prototype_class_labels[predicted_label_indices_relative_to_prototypes]

        return predicted_actual_labels, predicted_max_probabilities, all_class_probabilities

################################################################################
# Evaluation Function for Prototypical Network
################################################################################
def evaluate_prototypical_network(
    proto_net: PrototypicalNetwork, 
    test_loader: DataLoader, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_total_classes_for_cm: Optional[int] = None # For confusion matrix label range
):
    proto_net.to(device)
    proto_net.eval() 

    correct = 0
    total = 0
    all_preds_idx_list = []
    all_true_labels_idx_list = []
    all_pred_max_probs_list = [] 

    if len(test_loader.dataset) == 0:
        print("Evaluation loader for Prototypical Network is empty. Skipping evaluation.")
        # Return default values or raise an error
        return 0.0, 0.0, 0.0, 0.0, 0.0 

    print("\nEvaluating Prototypical Network...")
    with torch.no_grad():
        for inputs, true_labels_idx in test_loader: # true_labels_idx are integer labels
            inputs = inputs.to(device)
            true_labels_idx = true_labels_idx.to(device)
            
            predicted_labels, predicted_max_probs, _ = proto_net(inputs)

            total += true_labels_idx.size(0)
            correct += (predicted_labels == true_labels_idx).sum().item()
            
            all_preds_idx_list.append(predicted_labels.cpu())
            all_true_labels_idx_list.append(true_labels_idx.cpu())
            all_pred_max_probs_list.append(predicted_max_probs.cpu())

    if total == 0:
        print("No samples processed in the Prototypical Network evaluation.")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    accuracy = 100 * correct / total
    
    all_preds_idx_cat = torch.cat(all_preds_idx_list).numpy()
    all_true_labels_idx_cat = torch.cat(all_true_labels_idx_list).numpy()
    avg_confidence_on_prediction = torch.cat(all_pred_max_probs_list).mean().item() if all_pred_max_probs_list else 0.0

    precision = precision_score(all_true_labels_idx_cat, all_preds_idx_cat, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels_idx_cat, all_preds_idx_cat, average='weighted', zero_division=0)
    f1 = f1_score(all_true_labels_idx_cat, all_preds_idx_cat, average='weighted', zero_division=0)
    
    print(f'Prototypical Network Evaluation Results:')
    print(f'  Accuracy: {accuracy:.2f}%')
    print(f'  Avg. Confidence of Predicted Class: {avg_confidence_on_prediction:.4f}')
    print(f'  Precision (weighted): {precision:.4f}')
    print(f'  Recall (weighted): {recall:.4f}')
    print(f'  F1 Score (weighted): {f1:.4f}')
    
    # Confusion Matrix
    # Use num_total_classes_for_cm if provided and valid, otherwise infer from data or prototypes
    cm_num_labels = num_total_classes_for_cm
    if cm_num_labels is None and proto_net.prototypes is not None:
        cm_num_labels = proto_net.prototypes.shape[0]
    
    if cm_num_labels is not None and cm_num_labels > 0:
        try:
            import seaborn as sns 
            # Ensure labels for confusion_matrix are within the expected range [0, cm_num_labels-1]
            cm_labels_range = list(range(cm_num_labels))
            cm = confusion_matrix(all_true_labels_idx_cat, all_preds_idx_cat, labels=cm_labels_range)
            
            plt.figure(figsize=(max(8, cm_num_labels*0.5), max(6, cm_num_labels*0.4))) # Dynamic size
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=cm_labels_range, 
                        yticklabels=cm_labels_range)
            plt.title('Prototypical Network Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.savefig('prototypical_network_confusion_matrix.png')
            plt.close()
        except ImportError:
            print("Seaborn not installed. Skipping confusion matrix plot for prototypical network.")
        except Exception as e:
            print(f"Error plotting prototypical network confusion matrix: {e}")
            
    return accuracy, precision, recall, f1, avg_confidence_on_prediction
################################################################################
# Main Program
################################################################################
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    # Ensure data loading functions return integer labels for SupCon and ProtoNet
    print("Loading main training data (for SupCon and prototype computation)...")
    train_dataset_full, num_classes = Train_data_Gen_with_labels( # Using updated function name
        data_path="./Data/Train_Data.mat", # Path to your primary training data
        label_path="./Data/Train_Label.mat"
    )
    print(f"Full training dataset size: {len(train_dataset_full)}, Num classes from training data: {num_classes}")

    print("Loading data for Prototypical Network testing (query set)...")
    # Using load_data_and_labels for test/query set
    # test_split=0 means ds1 gets all data from the files, ds2 is empty.
    # This ds1 will be used as the query set for the prototypical network.
    query_dataset_pool, _, num_classes_query = load_data_and_labels(
        data_path="./Data/Test_Data.mat",   # Path to your test/query data
        label_path="./Data/Test_Label.mat",
        test_split=0.0 
    )
    if num_classes_query > 0 and num_classes != num_classes_query: # num_classes_query could be 0 if pool empty
        print(f"Warning: Class count mismatch between training data ({num_classes}) and query data pool ({num_classes_query}). Using {num_classes} as the primary number of classes.")
    
    # If you want a validation set for hyperparameter tuning of ProtoNet (e.g., encoder LR for SupCon, temperature),
    # you would typically split train_dataset_full or have a separate validation file.
    # For this script, query_dataset_pool is directly used as the test/query set for ProtoNet.
    test_dataset_proto = query_dataset_pool

    print(f"Supervised Contrastive training set size: {len(train_dataset_full)}")
    print(f"Prototypical Network test (query) set size: {len(test_dataset_proto)}")
    
    # --- DataLoaders ---
    supcon_batch_size = min(128, len(train_dataset_full)) if len(train_dataset_full) > 0 else 1 # Adjusted to not be 0
    proto_eval_batch_size = min(64, len(test_dataset_proto)) if len(test_dataset_proto) > 0 else 1

    if len(train_dataset_full) == 0: 
        raise ValueError("Main training dataset (train_dataset_full) is empty. Cannot proceed.")
    
    # Loader for Supervised Contrastive training
    supcon_train_loader = DataLoader(train_dataset_full, batch_size=supcon_batch_size, shuffle=True, drop_last=True)
    # Loader for computing prototypes (uses the same training data, no shuffle/drop_last needed)
    prototype_compute_loader = DataLoader(train_dataset_full, batch_size=proto_eval_batch_size, shuffle=False)

    # Loader for evaluating the prototypical network on the query set
    if len(test_dataset_proto) > 0:
        test_loader_proto = DataLoader(test_dataset_proto, batch_size=proto_eval_batch_size, shuffle=False)
    else:
        print("Warning: Prototypical network test (query) set is empty. Evaluation will be skipped.")
        # Create an empty loader to prevent crashes if code expects it, though it won't be used meaningfully
        test_loader_proto = DataLoader(TensorDataset(torch.empty(0, *(train_dataset_full[0][0].shape)), torch.empty(0, dtype=torch.long)), batch_size=1)


    # --- Model & Training Hyperparameters ---
    model_type = 'resnet50'  # Example: 'resnet50', 'convnext_v2', 'mobilenet_v3', 'efficientnet_v2',  'shufflenet', 'densenet'
    use_pretrained_encoder_weights = True # For initial backbone weights before SupCon
    encoder_img_size = (224,224) # Adjust based on your data preprocessing for the encoder. Was (224,224), then (64,64)

    # Supervised Contrastive Learning Phase
    supcon_epochs = 200 # E.g., 20-100 epochs, depends on dataset
    supcon_lr = 5e-4 # Tunable: 1e-3, 5e-4, 1e-4
    # projection_hidden_dim will be set after encoder init based on encoder.encoder_feature_dim
    projection_output_dim = 128 # Common for contrastive embeddings  256
    supcon_temperature = 0.1  # Tunable: 0.07, 0.1, 0.5

    # --- Phase 1: Supervised Contrastive Encoder Training ---
    print(f"\n--- Phase 1: Supervised Contrastive Encoder Training ---")
    # Initialize Encoder (this instance will be trained)
    encoder_for_supcon = EncoderNetwork(
        Size=encoder_img_size, 
        model_type=model_type,
        pretrained=use_pretrained_encoder_weights,
        device=device
    )
    projection_hidden_dim = encoder_for_supcon.encoder_feature_dim # Set based on actual encoder output

    print(f"Initialized EncoderNetwork for SupCon with {model_type}. Backbone feature dim: {encoder_for_supcon.encoder_feature_dim}")

    # ContrastiveLearningModel wraps the encoder and projection head for SupCon training
    supcon_training_model = ContrastiveLearningModel(
        encoder=encoder_for_supcon, 
        projection_hidden_dim=projection_hidden_dim,
        projection_output_dim=projection_output_dim
    ).to(device)
    
    optimizer_supcon = torch.optim.AdamW(
        supcon_training_model.parameters(), # Train both encoder and projection head
        lr=supcon_lr, 
        weight_decay=1e-5 # AdamW often benefits from some weight decay
    )
    
    # Start Supervised Contrastive training
    # This function trains supcon_training_model.encoder and saves its state_dict
    trained_encoder_instance = train_supervised_contrastive_phase(
        contrastive_model=supcon_training_model,
        train_loader=supcon_train_loader,
        optimizer=optimizer_supcon,
        num_epochs=supcon_epochs,
        device=device,
        temperature=supcon_temperature,
        model_name_prefix=f"{model_type}_supcon_encoder"
    )
    # trained_encoder_instance is the Python object of the trained EncoderNetwork

# --- Phase 2: Prototypical Network Classification ---
    print(f"\n--- Phase 2: Prototypical Network Classification ---")
    
    # Create a new encoder instance for the prototypical network
    # and load the weights trained via SupCon.
    # This ensures separation and that we are using the correct state.
    print(f"Loading supervised contrastively trained encoder for Prototypical Network...")
    encoder_for_prototyping = EncoderNetwork(
        Size=encoder_img_size, 
        model_type=model_type, 
        pretrained=False, # Weights will be loaded manually
        device=device
    )
    
    # Path to the saved encoder weights from the SupCon phase
    supcon_encoder_final_path = f'saved_models/{model_type}_supcon_encoder_epoch_{supcon_epochs}.pth'
    if os.path.exists(supcon_encoder_final_path):
        # <<<< ADD THE FOLLOWING LINES FOR DUMMY FORWARD PASS >>>>
        if len(train_dataset_full) > 0:
            print("Performing a dummy forward pass to initialize dynamic encoder modules...")
            # Create a dummy input tensor with the same shape characteristics as training data
            # Batch size of 1 is sufficient for shape inference.
            # train_dataset_full[0][0] is the first data sample (tensor)
            sample_data_shape = train_dataset_full[0][0].shape 
            # Expected shape: (C, H, W). Add batch dimension: (1, C, H, W)
            dummy_input = torch.randn(1, *sample_data_shape, device=device)
            
            # Perform a forward pass in eval mode to correctly initialize internal modules
            # without affecting batch norm statistics or requiring gradients.
            encoder_for_prototyping.eval() 
            try:
                with torch.no_grad():
                    _ = encoder_for_prototyping(dummy_input)
                print("Dummy forward pass completed. Modules should now be correctly sized.")
            except Exception as e:
                print(f"Error during dummy forward pass: {e}")
                print("Proceeding with load_state_dict, but size mismatches might still occur if dummy pass failed to set sizes correctly.")
        else:
            print("Warning: train_dataset_full is empty, cannot perform dummy forward pass. Module sizes might not match state_dict.")
        # <<<< END OF ADDED LINES >>>>

        encoder_for_prototyping.load_state_dict(torch.load(supcon_encoder_final_path, map_location=device))
        print(f"Loaded SupCon encoder weights from {supcon_encoder_final_path} for Prototypical Network.")
    else:
        # Fallback: if the saved file isn't found, use the in-memory trained_encoder_instance.
        # This might happen if saving failed or path is incorrect.
        print(f"Warning: Saved SupCon encoder weights not found at {supcon_encoder_final_path}.")
        print("Attempting to use the encoder instance directly from SupCon training phase for ProtoNet.")
        # trained_encoder_instance was returned by train_supervised_contrastive_phase
        if 'trained_encoder_instance' in locals() and trained_encoder_instance is not None:
            # Before loading state_dict from an in-memory instance, also ensure its modules are correctly sized
            # (though they should be if it was trained, this is more for the freshly created encoder_for_prototyping)
            # If trained_encoder_instance is directly assigned, this dummy pass logic might be less critical for it,
            # but the error occurs with a new EncoderNetwork instance.
            encoder_for_prototyping.load_state_dict(trained_encoder_instance.state_dict())
        else:
            raise FileNotFoundError(f"Encoder weights not found at {supcon_encoder_final_path} and no in-memory instance (trained_encoder_instance) available.")

    encoder_for_prototyping.to(device).eval() # Ensure it's on device and in eval mode

    # Initialize Prototypical Network
    prototypical_net = PrototypicalNetwork(encoder=encoder_for_prototyping, device=device)    
    
    # Compute prototypes using the training data
    if len(prototype_compute_loader.dataset) > 0:
        prototypical_net.compute_prototypes(prototype_compute_loader, num_classes)
    else:
        # This should ideally not happen if train_dataset_full was non-empty
        print("Warning: Prototype computation loader is empty. Prototypes may not be computed correctly.")
        # Attempt to compute with an empty loader will likely lead to empty prototypes.
        # PrototypicalNetwork.compute_prototypes handles this by creating empty/zero prototypes.

    # Evaluate the Prototypical Network on the test/query set
    if len(test_loader_proto.dataset) > 0:
        evaluate_prototypical_network(
            proto_net=prototypical_net,
            test_loader=test_loader_proto,
            device=device,
            num_total_classes_for_cm=num_classes # Pass total classes for confusion matrix range
        )
        
        # Example Prediction with the Prototypical Network
        print("\nExample prediction using the Prototypical Network:")
        try:
            # Fetch a batch from the test loader for prediction
            sample_data_proto_batch, sample_true_label_proto_batch = next(iter(test_loader_proto))
            # Take the first sample from the batch for individual prediction display
            sample_input_proto = sample_data_proto_batch[0:1].to(device) 
            sample_true_label_proto = sample_true_label_proto_batch[0].item()
            
            pred_label_idx, pred_max_prob, all_class_probs_dist = prototypical_net(sample_input_proto)
            
            print(f"  Sample input shape: {sample_input_proto.shape}")
            print(f"  True label for sample: {sample_true_label_proto}")
            print(f"  Predicted class index by ProtoNet: {pred_label_idx.item()}")
            print(f"  Probability of this predicted class: {pred_max_prob.item():.4f}")
            # For debugging, show the full probability distribution for the sample
            # print(f"  All class probabilities for sample: {all_class_probs_dist.squeeze().cpu().numpy()}")

        except StopIteration: # Should not happen if len(test_loader_proto.dataset) > 0 check passed
            print("Test loader for Prototypical Network unexpectedly empty, cannot get a sample for prediction.")
    else:
        print("Test (query) dataset for Prototypical Network is empty. Skipping evaluation and example prediction.")

    print("\nRun completed. All changes for Supervised Contrastive Learning and Prototypical Networks are incorporated.")
    print("This round of fixes is complete. You can test the code or move to the next item.")