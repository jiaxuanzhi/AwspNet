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
from pytorch_wavelets import ScatLayer,ScatLayerj2 # These are used
import random # Added for new augmentation
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

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
                               shift_percent: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates two augmented views for supervised contrastive learning.
    View 1: Original + Gaussian Noise.
    View 2: Original + Horizontal Shift with zero-padding.
    """
    batch_tensor_on_device = batch_tensor.to(device)

    # View 1: Gaussian Noise
    noise = torch.randn_like(batch_tensor_on_device) * noise_sigma
    view1 = batch_tensor_on_device + noise
    # # View 2: Gaussian Noise
    # noise = torch.randn_like(batch_tensor_on_device) * noise_sigma
    # view2 = batch_tensor_on_device + noise

    # View 2: Horizontal Shift
    # Ensure batch_tensor_on_device is passed, as horizontal_shift_tensor expects a tensor
    view2 = horizontal_shift_tensor(batch_tensor_on_device, max_shift_percent=shift_percent, padding_value=0.0)
    
    return view1, view2
    # return view1

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

        self._init_encoder_backbone() # Initializes self.backbone and self.encoder_feature_dim
        self.to(device) # Move the entire EncoderNetwork to the specified device
    
    # The _init_encoder_backbone method remains the same.
    # The forward method (with its conditional re-initialization logic for 
    # attention_module and channel_reducer) remains the same.

    def _init_encoder_backbone(self):
        """Initializes the backbone model as an encoder, removing the final classification layer."""
        if self.model_type == 'convnext_v2':
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if self.pretrained else None
            model_temp = models.convnext_small(weights=weights)
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

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.backbone = self.backbone.to(self.device)
        if self.encoder_feature_dim == 0:
             raise ValueError(f"encoder_feature_dim was not set for model type {self.model_type}. Check _init_encoder_backbone.")

    def forward(self, x):
        x = x.to(self.device)
        # print(f"Input shape to EncoderNetwork: {x.shape}") # 打印输入形状
        Z = self.scat2a(x) # Using ScatLayerj2 as per user's previous implication
        # print(f"Shape after scat2a (Z): {Z.shape}") # <<--- 关键：检查 Z.shape[1] (通道数)

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
        # print(f"Shape after attention_module (att_features): {att_features.shape}") # <<--- 关键

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
        # print(f"Shape after channel_reducer (reduced_features): {reduced_features.shape}")

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
    plt.savefig('./ResultFig/supcon_training_loss.pdf', dpi=300, bbox_inches='tight') # Save with tight bounding box
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
    num_total_classes_for_cm: Optional[int] = None # For confusion matrix and ROC/PR label range
):
    proto_net.to(device)
    proto_net.eval()

    correct = 0
    total = 0
    all_preds_idx_list = []
    all_true_labels_idx_list = []
    all_pred_max_probs_list = []
    all_class_probabilities_list = [] # 新增：用于存储所有类别的预测概率

    if len(test_loader.dataset) == 0:
        print("Evaluation loader for Prototypical Network is empty. Skipping evaluation.")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    print("\nEvaluating Prototypical Network...")
    with torch.no_grad():
        for inputs, true_labels_idx in test_loader: # true_labels_idx are integer labels
            inputs = inputs.to(device)
            true_labels_idx = true_labels_idx.to(device)
            
            # 从原型网络获取预测标签、最大概率以及所有类别的概率分布
            predicted_labels, predicted_max_probs, all_probs_batch = proto_net(inputs)

            total += true_labels_idx.size(0)
            correct += (predicted_labels == true_labels_idx).sum().item()
            
            all_preds_idx_list.append(predicted_labels.cpu())
            all_true_labels_idx_list.append(true_labels_idx.cpu())
            all_pred_max_probs_list.append(predicted_max_probs.cpu())
            all_class_probabilities_list.append(all_probs_batch.cpu()) # 新增：收集所有类别的概率

    if total == 0:
        print("No samples processed in the Prototypical Network evaluation.")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    accuracy = 100 * correct / total
    
    all_preds_idx_cat = torch.cat(all_preds_idx_list).numpy()
    all_true_labels_idx_cat = torch.cat(all_true_labels_idx_list).numpy()
    avg_confidence_on_prediction = torch.cat(all_pred_max_probs_list).mean().item() if all_pred_max_probs_list else 0.0
    
    # 新增：合并所有类别的概率为一个Numpy数组
    all_class_probs_cat = torch.cat(all_class_probabilities_list, dim=0).numpy()


    precision_metric = precision_score(all_true_labels_idx_cat, all_preds_idx_cat, average='weighted', zero_division=0)
    recall_metric = recall_score(all_true_labels_idx_cat, all_preds_idx_cat, average='weighted', zero_division=0)
    f1_metric = f1_score(all_true_labels_idx_cat, all_preds_idx_cat, average='weighted', zero_division=0)
    
    print(f'Prototypical Network Evaluation Results:')
    print(f'  Accuracy: {accuracy:.2f}%')
    print(f'  Avg. Confidence of Predicted Class: {avg_confidence_on_prediction:.4f}')
    print(f'  Precision (weighted): {precision_metric:.4f}')
    print(f'  Recall (weighted): {recall_metric:.4f}')
    print(f'  F1 Score (weighted): {f1_metric:.4f}')
    
    # --- ROC Curve Generation ---
    n_classes = num_total_classes_for_cm
    if n_classes is None or n_classes <= 1: # ROC/PR meaningful for >= 2 classes
        print("Skipping ROC/PR curve generation: num_total_classes_for_cm not set appropriately for multi-class ROC/PR.")
    elif all_class_probs_cat.shape[1] != n_classes:
        print(f"Skipping ROC/PR: Mismatch between probability columns ({all_class_probs_cat.shape[1]}) and n_classes ({n_classes}).")
    else:
        y_true_binarized = label_binarize(all_true_labels_idx_cat, classes=range(n_classes))

        # ROC Curves
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], all_class_probs_cat[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC Class {i} (AUC = {roc_auc[i]:.2f})')

        # Micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), all_class_probs_cat.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr])) # Ensure key exists
        mean_tpr = np.zeros_like(all_fpr)
        valid_classes_for_macro = 0
        for i in range(n_classes):
            if i in fpr and i in tpr: # Check if class i had ROC computed
                 mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                 valid_classes_for_macro +=1
        if valid_classes_for_macro > 0:
            mean_tpr /= valid_classes_for_macro
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            plt.plot(fpr["macro"], tpr["macro"],
                     label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
                     color='navy', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2) # Chance line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves - Prototypical Network')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('./ResultFig/prototypical_network_roc_curves.pdf', dpi=300, bbox_inches='tight') 
        plt.close()
        print("Prototypical Network ROC curves plotted and saved to prototypical_network_roc_curves.png")

        # --- PR Curve Generation ---
        precision = dict()
        recall = dict()
        average_precision = dict()
        plt.figure(figsize=(10, 8))

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], all_class_probs_cat[:, i])
            average_precision[i] = average_precision_score(y_true_binarized[:, i], all_class_probs_cat[:, i])
            plt.plot(recall[i], precision[i], lw=2, label=f'PR Class {i} (AP = {average_precision[i]:.2f})')
        
        # Micro-average PR
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_binarized.ravel(), all_class_probs_cat.ravel())
        # Note: average_precision_score with average='micro' directly computes micro-AP
        average_precision["micro"] = average_precision_score(y_true_binarized, all_class_probs_cat, average="micro")
        plt.plot(recall["micro"], precision["micro"],
                 label=f'Micro-average PR (AP = {average_precision["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)
        
        # (Optional: Add a no-skill line, which is the prevalence of the positive class)
        # Example for class 0, if useful for interpretation:
        # if n_classes > 0 and y_true_binarized.shape[0] > 0:
        #     no_skill_class0 = y_true_binarized[:, 0].sum() / len(y_true_binarized)
        #     plt.plot([0, 1], [no_skill_class0, no_skill_class0], linestyle='--', color='grey', label=f'No skill Class 0 ({no_skill_class0:.2f})')


        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall (PR) Curves - Prototypical Network')
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig('./ResultFig/prototypical_network_pr_curves.pdf', dpi=300, bbox_inches='tight') 
        plt.close()
        print("Prototypical Network PR curves plotted and saved to prototypical_network_pr_curves.png")

    # Confusion Matrix plotting (remains as in your last full code version)
    cm_num_labels = num_total_classes_for_cm
    if cm_num_labels is not None and cm_num_labels > 0:
        try:
            import seaborn as sns 
            cm_labels_range = list(range(cm_num_labels))
            cm = confusion_matrix(all_true_labels_idx_cat, all_preds_idx_cat, labels=cm_labels_range)
            
            plt.figure(figsize=(max(8, cm_num_labels*0.5), max(6, cm_num_labels*0.4)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=cm_labels_range, 
                        yticklabels=cm_labels_range)
            plt.title('Prototypical Network Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.savefig('./ResultFig/prototypical_network_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            print("Seaborn not installed. Skipping confusion matrix plot for prototypical network.")
        except Exception as e:
            print(f"Error plotting prototypical network confusion matrix: {e}")
            
    return accuracy, precision_metric, recall_metric, f1_metric, avg_confidence_on_prediction
################################################################################
# Main Program
################################################################################

def load_mat_data(filepath: str, data_key_name: str) -> np.ndarray:
    """
    从 .mat 文件加载指定的单个数据变量，并处理复数数据。
    该函数主要依赖 scipy.io.loadmat，并期望能够处理多种 MAT 文件版本。
    加载后，它会应用复数到实数的转换逻辑。

    Args:
        filepath (str): .mat 文件的路径。
        data_key_name (str): 要从 .mat 文件中加载的数据变量的名称。

    Returns:
        np.ndarray: 加载并处理后的数据，作为 float32 类型的 NumPy 数组。
    
    Raises:
        FileNotFoundError: 如果指定路径的文件不存在。
        IOError: 如果文件无法读取或不是有效的 MAT 文件。
        KeyError: 如果在 MAT 文件中没有找到指定的 data_key_name。
        ValueError: 如果加载的变量为空或 MAT 文件不包含任何用户变量，或复数数据维度不支持。
    """
    loaded_data_np = None # 初始化变量
    try:
        # 使用 variable_names 参数仅加载指定变量，提高效率
        # squeeze_me=True: 移除单维度条目。
        mat_contents = sio.loadmat(filepath, variable_names=[data_key_name], squeeze_me=False)
        
        if data_key_name not in mat_contents:
            raise KeyError(f"错误：指定的键 '{data_key_name}' 未在 MAT 文件 {filepath} 中通过 sio.loadmat 找到。")
        
        loaded_data_np = mat_contents[data_key_name]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：MAT 文件未在路径中找到: {filepath}")
    except NotImplementedError as e:
        print(f"scipy.io.loadmat 对 {filepath} (特定变量 '{data_key_name}') 失败 (NotImplementedError: {e})，尝试使用 h5py 直接读取 v7.3 格式...")
        try:
            with h5py.File(filepath, 'r') as f:
                if data_key_name not in f:
                    raise KeyError(f"错误：在 HDF5 文件 {filepath} 中未找到数据键 '{data_key_name}'。")
                # 将 h5py.Dataset 转换为 NumPy 数组
                data_from_h5 = f[data_key_name][()]                 
                loaded_data_np = data_from_h5.T

        except Exception as h5_err:
            raise IOError(f"使用 h5py 读取 MAT 文件 {filepath} 中的变量 '{data_key_name}' 失败。错误: {h5_err}")
    except Exception as e:
        raise IOError(f"无法读取 MAT 文件 {filepath}。错误: {e}")

    if loaded_data_np is None:
        raise ValueError(f"未能从 {filepath} 中加载变量 '{data_key_name}'。")

    # 确保 loaded_data_np 是一个 NumPy 数组，即使 squeeze_me=True 后它可能变成标量
    if not isinstance(loaded_data_np, np.ndarray):
        loaded_data_np = np.array(loaded_data_np)

    # --- 应用您提供的复数数据处理逻辑 ---
    data_to_process = loaded_data_np # 使用一个新变量以免混淆
    
    if np.iscomplexobj(data_to_process):
        print(f"  处理复数数据 (iscomplexobj) from {filepath} for key '{data_key_name}', ndim: {data_to_process.ndim}")
        if data_to_process.ndim == 3: 
            # 假设输入是 (N, H, W) 复数, 输出 (N, 2, H, W) 实数
            data_to_process = np.stack((data_to_process.real, data_to_process.imag), axis=1)
        elif data_to_process.ndim == 4: 
            data_to_process = np.concatenate((data_to_process.real, data_to_process.imag), axis=1)
        elif data_to_process.ndim == 5: # 对应 Batch_SNR_JammerNum_Channel_Pulse_Range 中的 Channel_Pulse_Range 部分
            print(f"    ndim={data_to_process.ndim}, applying concatenate along axis=-3 for real/imag parts.")
            data_to_process = np.concatenate((data_to_process.real, data_to_process.imag), axis=-3) # 与您代码一致
        elif data_to_process.ndim == 6: # 对应 Batch_SNR_JammerNum_Channel_Pulse_Range 中的 Channel_Pulse_Range 部分
            print(f"    ndim={data_to_process.ndim}, applying concatenate along axis=-3 for real/imag parts.")
            data_to_process = np.concatenate((data_to_process.real, data_to_process.imag), axis=-3) # 与您代码一致
        elif data_to_process.ndim == 7: # 对应 Batch_SNR_JammerNum_Channel_Pulse_Range 中的 Channel_Pulse_Range 部分
            print(f"    ndim={data_to_process.ndim}, applying concatenate along axis=-3 for real/imag parts.")
            data_to_process = np.concatenate((data_to_process.real, data_to_process.imag), axis=-3) # 与您代码一致
        else:
            raise ValueError(f"不支持的复数数据维度 (iscomplexobj): {data_to_process.ndim} for key '{data_key_name}' in {filepath}")

    elif data_to_process.dtype == np.void and 'real' in data_to_process.dtype.names and 'imag' in data_to_process.dtype.names:
        print(f"  处理复数数据 (void type) from {filepath} for key '{data_key_name}'")
        real_part = data_to_process['real']
        imag_part = data_to_process['imag']
        data_to_process = np.concatenate((real_part, imag_part), axis=-3)

    elif data_to_process.dtype == np.dtype([('real', '<f4'), ('imag', '<f4')]) or \
         data_to_process.dtype == np.dtype([('real', '<f8'), ('imag', '<f8')]):
        print(f"  处理复数数据 (structured array with real/imag fields) from {filepath} for key '{data_key_name}'")
        real_part = data_to_process['real']
        imag_part = data_to_process['imag']
        data_to_process = np.concatenate((real_part, imag_part), axis=-3)
    # --- 复数数据处理逻辑结束 ---
        
    return data_to_process.astype(np.float32)

if __name__ == "__main__":
    # --- 基本设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('./ResultFig/', exist_ok=True)  # 自动创建图片保存目录（如果不存在）
    mode = input("Choose mode (train/evaluate): ").strip().lower()
    # --- Data Loading ---
    # Ensure data loading functions return integer labels for SupCon and ProtoNet
    print("Loading main training data (for SupCon and prototype computation)...")
    train_dataset_full, num_classes = Train_data_Gen_with_labels( # Using updated function name
        data_path="./Data/Train_Data_Com.mat", # Path to your primary training data
        label_path="./Data/Train_Label_identify.mat"
    )
    print(f"Full training dataset size: {len(train_dataset_full)}, Num classes from training data: {num_classes}")

    print("Loading data for Prototypical Network testing (query set)...")
    # Using load_data_and_labels for test/query set
    # test_split=0 means ds1 gets all data from the files, ds2 is empty.
    # This ds1 will be used as the query set for the prototypical network.
    query_dataset_pool, _, num_classes_query = load_data_and_labels(
        data_path="./Data/Test_Data.mat",   # Path to your test/query data
        label_path="./Data/Test_Label_identify.mat",
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
    supcon_batch_size = min(64, len(train_dataset_full)) if len(train_dataset_full) > 0 else 1 # Adjusted to not be 0
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
        '''
    **MobileNetV3**: Lightweight characteristics, reduced model parameters, and depthwise separable convolution significantly decrease computational load and parameter count, 
    enabling CNNs to operate efficiently on resource-constrained devices. The inverted residual structure expands first and then compresses,
    maintaining expressive power while reducing computation. It combines Neural Architecture Search (NAS) with manual design, 
    and network pruning removes layers and operations with minimal performance contributions. 
    The design also considers actual hardware characteristics to optimize model performance on specific devices.

    **EfficientNetV2**: Systematically balances network width, depth, and resolution for more efficient model scaling. 
    It uses Mobile Inverted Bottleneck Convolution as a basic building block, incorporates the Swish activation function, 
    and introduces Fused-MBConv blocks and a progressive learning strategy to improve training efficiency and accuracy.

    **ConvNeXtV2**: An improved version of ConvNeXt, introducing the Global Response Normalization (GRN) module to further enhance model performance. 
    The GRN module models global context in features, improving the model's ability to capture long-range dependencies while maintaining computational efficiency.

    **ShuffleNet (2018)**: Combines pointwise group convolution and channel shuffle, significantly reducing computational load while preserving accuracy.
    '''
    # --- 定义要训练和评估的模型类型列表 ---
    # model_types_to_train = ['efficientnet_v2']  # Example: 'resnet50', 'convnext_v2', 'mobilenet_v3', 'efficientnet_v2',  'shufflenet', 'densenet','resnext',
    model_types_to_train = [ 'efficientnet_v2','mobilenet_v3', 'shufflenet', 'convnext_v2', 'resnext','resnet50'] # 'resnet50', 'efficientnet_v2', 'densenet',
    use_pretrained_encoder_weights = True # For initial backbone weights before SupCon
    encoder_img_size = (224,224) # Adjust based on your data preprocessing for the encoder. Was (224,224), then (64,64)

    # Supervised Contrastive Learning Phase
    supcon_epochs = 50 # E.g., 20-100 epochs, depends on dataset
    supcon_lr = 5e-4 # Tunable: 1e-3, 5e-4, 1e-4
    # projection_hidden_dim will be set after encoder init based on encoder.encoder_feature_dim
    projection_output_dim = 256 # Common for contrastive embeddings  256
    supcon_temperature = 0.2  # Tunable: 0.07, 0.1, 0.5

    # --- Phase 1: Supervised Contrastive Encoder Training ---
    print(f"\n--- Phase 1: Supervised Contrastive Encoder Training ---")
    # Initialize Encoder (this instance will be trained)
    # --- 遍历模型类型进行训练和评估 ---
    for model_type in model_types_to_train:
        print(f"\n\n{'='*30}\n   STARTING PROCESSING FOR MODEL TYPE: {model_type.upper()}\n{'='*30}")

        current_model_name_prefix_for_saving = f"{model_type}_supcon_encoder"
        current_model_identifier_tag = f"{model_type}_Prototypical"

        # --- Phase 1: Supervised Contrastive Encoder Training ---
        print(f"\n--- Phase 1: Supervised Contrastive Encoder Training for {model_type} ---")
        encoder_for_supcon = EncoderNetwork(
            Size=encoder_img_size, 
            model_type=model_type, # 当前循环中的 model_type
            pretrained=use_pretrained_encoder_weights,
            device=device
        )
        projection_hidden_dim = encoder_for_supcon.encoder_feature_dim
        print(f"Initialized EncoderNetwork ({model_type}). Backbone feature dim: {encoder_for_supcon.encoder_feature_dim}")

        supcon_training_model = ContrastiveLearningModel(
            encoder=encoder_for_supcon, 
            projection_hidden_dim=projection_hidden_dim,
            projection_output_dim=projection_output_dim
        ).to(device)
        
        # --- 设置分层学习率 (根据您的代码片段调整) ---
        lr_pretrained_backbone = supcon_lr * 0.2  # 预训练骨干的学习率是 supcon_lr 的 0.2 倍
        lr_new_parts = supcon_lr                 # 其他部分（自定义前端、投影头）使用基础学习率

        pretrained_backbone_params = list(supcon_training_model.encoder.backbone.parameters())
        encoder_frontend_params = []
        # 添加编码器中非骨干部分的参数 (scat2a, attention_module, channel_reducer)
        # if hasattr(supcon_training_model.encoder, 'scat2a'): # scat2a 是 EncoderNetwork 的一个属性
        #     encoder_frontend_params.extend(list(supcon_training_model.encoder.scat2a.parameters()))
        if hasattr(supcon_training_model.encoder, 'attention_module'):
            encoder_frontend_params.extend(list(supcon_training_model.encoder.attention_module.parameters()))
        if hasattr(supcon_training_model.encoder, 'channel_reducer'):
            encoder_frontend_params.extend(list(supcon_training_model.encoder.channel_reducer.parameters()))
        
        projection_head_params = list(supcon_training_model.projection_head.parameters())

        optimizer_parameter_groups = []
        if pretrained_backbone_params:
            optimizer_parameter_groups.append({'params': pretrained_backbone_params, 'lr': lr_pretrained_backbone})
            # print(f"Optimizer ({model_type}): Pretrained backbone LR: {lr_pretrained_backbone}") # 日志移到后面统一打印
        if encoder_frontend_params:
            optimizer_parameter_groups.append({'params': encoder_frontend_params, 'lr': lr_new_parts})
            # print(f"Optimizer ({model_type}): Encoder frontend LR: {lr_new_parts}")
        if projection_head_params:
            optimizer_parameter_groups.append({'params': projection_head_params, 'lr': lr_new_parts})
            # print(f"Optimizer ({model_type}): Projection head LR: {lr_new_parts}")
        
        if not optimizer_parameter_groups:
             raise ValueError(f"No parameters found to optimize for model {model_type}.")

        optimizer_supcon = torch.optim.AdamW(optimizer_parameter_groups, weight_decay=1e-5)
        print(f"Optimizer for {model_type} configured with LLR (Backbone LR: {lr_pretrained_backbone}, New Parts LR: {lr_new_parts}).")
        
        if mode == 'train':
            trained_encoder_instance = train_supervised_contrastive_phase(
                contrastive_model=supcon_training_model,
                train_loader=supcon_train_loader,
                optimizer=optimizer_supcon,
                num_epochs=supcon_epochs,
                device=device,
                temperature=supcon_temperature,
                model_name_prefix=current_model_name_prefix_for_saving 
            )
        
        # --- Phase 2: Prototypical Network Classification ---
        print(f"\n--- Phase 2: Prototypical Network Classification for {model_type} ---")
        encoder_for_prototyping = EncoderNetwork(
            Size=encoder_img_size, 
            model_type=model_type, 
            pretrained=False, 
            device=device
        )
        
        # supcon_encoder_final_path = f'saved_models/{current_model_name_prefix_for_saving}_epoch_{supcon_epochs}.pth'

        numTpSample = 48  # 滑窗大小
        supcon_encoder_final_path = f'saved_models/{current_model_name_prefix_for_saving}_epoch_{supcon_epochs}.pth'
        if os.path.exists(supcon_encoder_final_path):
            if len(train_dataset_full) > 0:
                print(f"Performing dummy forward pass for {model_type} encoder_for_prototyping...")
                # temp = train_dataset_full[0][0]
                # sample_data_shape = temp[:,:,0:numTpSample].shape

                sample_data_shape = train_dataset_full[0][0].shape
                # sample_data_shape = raw_data_np.shape[-3:]
                # sample_data_shape = raw_data_np[0,0,0,:,:,:].shape
                # sample_data_shape = train_dataset_full[0,0,0,:,:,0:numTpSample].shape
                
                dummy_input = torch.randn(1, *sample_data_shape, device=device)
                encoder_for_prototyping.eval() 
                try:
                    with torch.no_grad():
                        _ = encoder_for_prototyping(dummy_input)
                    print("Dummy forward pass completed.")
                except Exception as e:
                    print(f"Error during dummy forward pass for {model_type}: {e}")
            
            encoder_for_prototyping.load_state_dict(torch.load(supcon_encoder_final_path, map_location=device))
            print(f"Loaded SupCon encoder weights from {supcon_encoder_final_path} for {model_type}.")

        encoder_for_prototyping.to(device).eval()

        prototypical_net = PrototypicalNetwork(encoder=encoder_for_prototyping, device=device)   
        # numTpSample = 48  # 滑窗大小
        output_plot_dir = './AWSPNet_Test_Results_2'
        os.makedirs(output_plot_dir, exist_ok=True)
        # --- 定义要测试的文件和参数 ---
        # 假设 num_Tar 是一个已知参数，这里用 1 作为示例    
        test_scenarios = [
            {
                'filename_template': 'Target{}_Clutter0_Jammer_Data2.mat', # 无杂波数据
                'data_key': 'AwspNet_IndentifyTestData',
                'description': 'NoClutter_6D' # 假设这个是6D
            },
            {
                'filename_template': 'Target{}_Clutter1_Jammer_Data2.mat', # 有杂波数据
                'data_key': 'AwspNet_IndentifyTestData_addClutter',
                'description': 'WithClutter_7D' # 假设这个是7D
            }
        ]

        print(f"Using device: {device}")
        snr_db_values = np.linspace(-5, 10, int((10 - (-5)) / 5) + 1)
        cnr_db_values_for_7d = np.linspace(-5, 10, int((10 - (-5)) / 5) + 1) # 仅用于7D情况
        jammer_num_values = np.arange(0, num_classes_query + 1, 1) # MATLAB 0:1:10
        numTpSample = 48
        num_Tar_example = 2 # 你可能需要根据实际情况调整或循环处理不同的 num_Tar
        # raw_data_np = load_mat_data('./Data/Target1_Clutter0_Jammer_Data2.mat', 'AwspNet_IndentifyTestData') # <--- 使用新的加载函数
        # raw_data_np_clutter = load_mat_data('./Data/Target1_Clutter1_Jammer_Data.mat', 'AwspNet_IndentifyTestData_Clutter') # <--- 使用新的加载函数
        for scenario in test_scenarios:

            try:
                # raw_data_np = load_mat_data(filename, data_key) # <--- 使用新的加载函数
                # print(f"Successfully loaded data for key '{data_key}' from {filename}. Shape: {raw_data_np.shape}")
                filename = scenario['filename_template'].format(round(num_Tar_example))
                raw_data_np = load_mat_data('./Data/'+filename, scenario['data_key']) # <--- 使用新的加载函数
                # 后续代码现在使用这里定义的 snr_db_values, cnr_db_values_for_7d, jammer_num_values
                # 而不是从 Format_Explanation 解析
        
                data_tensor = torch.from_numpy(raw_data_np).to(device)
        
                # 确定当前场景的 CNR 值列表 (如果是6D场景则为 None)
                current_cnr_values = None
                if scenario['description'] == 'WithClutter_7D':                
                    cnr_db_values = cnr_db_values_for_7d
                    
            except FileNotFoundError as e:
                print(e)
            except (IOError, ValueError, KeyError) as e:
                print(f"Error processing file {filename}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with {filename}: {e}")

            data_tensor = torch.from_numpy(raw_data_np).to(device)
            # 预期数据维度: Batch, SNR, [CNR], JammerNum, Channel, Pulse, Range

            num_batch_items = data_tensor.shape[0]

            num_channel = data_tensor.shape[-3]
            num_pulse = data_tensor.shape[-2]
            num_range_full = data_tensor.shape[-1]

            print(f"  Data tensor shape: {data_tensor.shape}")
            print(f"  Num Batch Items: {num_batch_items}, Channels: {num_channel}, Pulses: {num_pulse}, Full Range Points: {num_range_full}")
            prototypical_net.compute_prototypes(prototype_compute_loader, num_classes)

            if data_tensor.shape[-1] >= numTpSample:
                L_prob_seq = num_range_full - numTpSample + 1
            else:
                L_prob_seq = 0 # 如果 Range 维度本身就小于滑窗大小，则序列长度为0
                print(f"  Warning: Full range dimension ({data_tensor.shape[-1]}) is smaller than numTpSample ({numTpSample}). Probability sequences will be empty.")

            if scenario['description'] == 'NoClutter_6D':
                # 预期维度: Batch, SNR, JammerNum, L_prob_seq
                all_prob_sequences_class0_np = np.zeros(
                    (num_batch_items, len(snr_db_values), len(jammer_num_values), L_prob_seq),
                    dtype=np.float32
                )
                print(f"  Initialized results array for NoClutter_6D with shape: {all_prob_sequences_class0_np.shape}")
            elif scenario['description'] == 'WithClutter_7D' :
                # 预期维度: Batch, SNR, CNR, JammerNum, L_prob_seq
                num_cnr_levels = len(cnr_db_values_for_7d)
                all_prob_sequences_class0_np = np.zeros(
                    (num_batch_items, len(snr_db_values), len(cnr_db_values_for_7d), len(jammer_num_values), L_prob_seq),
                    dtype=np.float32
                )
                print(f"  Initialized results array for WithClutter_7D with shape: {all_prob_sequences_class0_np.shape}")
            else:
                # 处理未预期的 scenario['description'] 或 cnr_db_values 为 None 的7D情况
                print(f"  Error: Could not determine shape for results array for scenario '{scenario['description']}'. Skipping result storage for this scenario.")
                all_prob_sequences_class0_np = None # 设置为None，后续保存时会跳过

            # --- 主循环 ---
            for snr_idx, snr_val in enumerate(snr_db_values):
                if scenario['description'] == 'WithClutter_7D' and cnr_db_values is not None:
                    cnr_iter = enumerate(cnr_db_values)
                else:
                    cnr_iter = [(0, None)] # 虚拟循环，执行一次

                for cnr_idx, cnr_val in cnr_iter: # cnr_val will be None for 6D case
                    for jammer_idx, jammer_val in enumerate(jammer_num_values):
                        print(f"  Testing: SNR={snr_val}dB, JammerNum={jammer_val}" + (f", CNR={cnr_val}dB" if cnr_val is not None else ""))

                        # 数据切片获取 (Batch, Channel, Pulse, Range_full)
                        if scenario['description'] == 'NoClutter_6D':
                            # Batch, SNR, JammerNum, Channel, Pulse, Range
                            current_data_for_setting = data_tensor[:, snr_idx, jammer_idx, :, :, :]
                        else: # WithClutter_7D
                            # Batch, SNR, CNR, JammerNum, Channel, Pulse, Range
                            current_data_for_setting = data_tensor[:, snr_idx, cnr_idx, jammer_idx, :, :, :]
                        
                        # current_data_for_setting shape: (Batch, Channel, Pulse, Range_full)

                        for batch_item_idx in range(current_data_for_setting.shape[0]):
                            data_item = current_data_for_setting[batch_item_idx] # Shape: (Channel, Pulse, Range_full)
                            
                            prob_sequence_class0 = []
                            window_start_indices = []
                            Rlocate = round(num_range_full//2)
                            # 滑窗处理 Range 维度
                            for r_start in range(data_item.shape[-1] - numTpSample + 1):
                                range_window = data_item[:, :, r_start : r_start + numTpSample]
                                # range_window shape: (Channel, Pulse, numTpSample)
                                padded_window = torch.zeros_like(data_item, device=data_item.device)
                                padded_window[:, :, Rlocate : Rlocate + numTpSample] = range_window

                                
                                # 准备网络输入: (1, Channel, Pulse, numTpSample)
                                network_input = padded_window.unsqueeze(0) 
                                # Data is already on device from data_tensor slicing

                                with torch.no_grad():
                                    # 假设 AWSPNet_model 输出形状为 (batch_size, num_classes) 的概率
                                    _,_,output_probs = prototypical_net.forward(network_input) 
                                
                                prob_class0 = output_probs[0, 0].item() # 假设类别0在索引0
                                prob_sequence_class0.append(prob_class0)
                                window_start_indices.append(r_start)

                            current_prob_seq_np = np.array(prob_sequence_class0, dtype=np.float32)
                            if L_prob_seq > 0 and len(current_prob_seq_np) == L_prob_seq:
                                if scenario['description'] == 'NoClutter_6D':
                                    all_prob_sequences_class0_np[batch_item_idx, snr_idx, jammer_idx, :] = current_prob_seq_np
                                elif scenario['description'] == 'WithClutter_7D' and cnr_val is not None: # 确保 cnr_idx 有效
                                    all_prob_sequences_class0_np[batch_item_idx, snr_idx, cnr_idx, jammer_idx, :] = current_prob_seq_np
                            elif L_prob_seq > 0 and len(current_prob_seq_np) != L_prob_seq and len(current_prob_seq_np) > 0 :
                                print(f"    Warning: Length mismatch for P(Class 0) sequence at "
                                        f"b{batch_item_idx}, snr{snr_val}, "
                                        f"{f'cnr{cnr_val}, ' if cnr_val is not None else ''}"
                                        f"jam{jammer_val}. Expected {L_prob_seq}, got {len(current_prob_seq_np)}. "
                                        f"Storing truncated or zero-padded portion.")
                                len_to_store = min(len(current_prob_seq_np), L_prob_seq)
                                if scenario['description'] == 'NoClutter_6D':
                                    all_prob_sequences_class0_np[batch_item_idx, snr_idx, jammer_idx, :len_to_store] = current_prob_seq_np[:len_to_store]
                                elif scenario['description'] == 'WithClutter_7D' and cnr_val is not None:
                                        all_prob_sequences_class0_np[batch_item_idx, snr_idx, cnr_idx, jammer_idx, :len_to_store] = current_prob_seq_np[:len_to_store]
                            # 绘制并保存该序列的图像
                            if prob_sequence_class0 and num_batch_items-1==batch_item_idx: # 仅当有数据时绘图
                                plt.figure(figsize=(9, 6)) # 调整图像大小
                                plt.rcParams.update({
                                    "font.family": "serif",          # 使用衬线字体
                                    "font.serif": ["Times New Roman"],  # 指定 Times New Roman
                                    "font.size": 16,                 # 基础字号
                                    "axes.titlesize": 16,            # 标题字号
                                    "axes.labelsize": 16,            # 坐标轴标签字号
                                })
                                plt.plot(window_start_indices, prob_sequence_class0, marker='.', linestyle='-', markersize=4)
                                plt.axvline(x=28, color='r', linestyle='--', linewidth=2, label='Target range cell')
                                title_str = f'SNR: {snr_val} dB'
                                if cnr_val is not None:
                                    title_str += f', CNR: {cnr_val} dB'
                                title_str += f', JammerNum: {jammer_val}'
                                plt.title(title_str, fontsize=16)
                                
                                plt.xlabel('Range cell')
                                plt.ylabel('Probability of target')
                                plt.ylim([-0.05, 1.05])
                                plt.grid(True, linestyle='--', alpha=0.7)
                                plt.tight_layout()

                                plot_filename_parts = [
                                    f"ProbClass0_{scenario['description']}",
                                    f"b{batch_item_idx}",
                                    f"snr{snr_val}"
                                ]
                                if cnr_val is not None:
                                    plot_filename_parts.append(f"cnr{cnr_val}")
                                plot_filename_parts.append(f"jam{jammer_val}")
                                
                                plot_filename = os.path.join(output_plot_dir, "_".join(plot_filename_parts) + ".png")
                                # plt.savefig(plot_filename)
                                plt.close()
                                # print(f"    Saved plot: {plot_filename}")
                            # else:
                                # print(f"    No probability sequence generated for BatchItem {batch_item_idx} (Range too short for window size {numTpSample}).")
                        # --- 保存包含所有概率序列的 .mat 文件 ---
            if all_prob_sequences_class0_np is not None: # 确保数组已成功初始化
                output_mat_filename = os.path.join(
                    output_plot_dir, # 使用与绘图相同的输出目录
                    f"{model_type}_{scenario['description']}_Target{num_Tar_example}.mat"
                )
                
                # 准备要保存到 .mat 文件的数据字典
                mat_data_to_save = {
                    'all_prob_sequences_class0': all_prob_sequences_class0_np,
                    'snr_db_values': np.array(snr_db_values), # 确保是 numpy array
                    'jammer_num_values': np.array(jammer_num_values), # 确保是 numpy array
                    'numTpSample': numTpSample,
                    'L_prob_seq': L_prob_seq,
                    'original_data_filename': filename # 记录原始数据文件名
                }
                if scenario['description'] == 'WithClutter_7D' and cnr_db_values is not None:
                    mat_data_to_save['cnr_db_values'] = np.array(cnr_db_values) # 确保是 numpy array

                try:
                    # from scipy.io import savemat # 确保已导入
                    sio.savemat(output_mat_filename, mat_data_to_save)
                    print(f"  Successfully saved all P(Class 0) probability sequences to: {output_mat_filename}")
                except Exception as e:
                    print(f"  Error saving probability sequences .mat file ({output_mat_filename}): {e}")
            else:
                print(f"  Skipping .mat file saving for scenario '{scenario['description']}' as results array was not initialized.")
            # --- .mat 文件保存结束 ---

            print(f"Finished processing conditions for {filename}.")

    print("\n--- All test scenarios processed ---")