#!/usr/bin/env python3
"""
Enhanced 3D Reconstruction Model Architecture
Multi-scale attention mechanisms with ResNet/ViT encoders and 3D decoders
Supports multiple 3D representations: NeRF, SDF, Occupancy Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math
from einops import rearrange, repeat
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, 
                value: torch.Tensor = None, mask: Optional[torch.Tensor] = None):
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear layer
        output = self.w_o(attention_output)
        
        return output, attention_weights
    
    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                  mask: Optional[torch.Tensor] = None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class CrossAttention(nn.Module):
    """Cross-attention between 2D and 3D features"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Cross-attention
        attn_output, attention_weights = self.attention(query, key, value, mask)
        query = self.norm1(query + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(query)
        output = self.norm2(query + ff_output)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor):
        return x + self.pe[:x.size(0), :]


class ResNetEncoder(nn.Module):
    """Enhanced ResNet encoder with attention mechanisms"""
    
    def __init__(self, encoder_type: str = 'resnet50', pretrained: bool = True, 
                 attention_layers: List[int] = [2, 3, 4]):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.attention_layers = attention_layers
        
        # Load ResNet backbone
        if encoder_type == 'resnet18':
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=pretrained)
            self.feature_dims = [64, 64, 128, 256, 512]
        elif encoder_type == 'resnet34':
            from torchvision.models import resnet34
            backbone = resnet34(pretrained=pretrained)
            self.feature_dims = [64, 64, 128, 256, 512]
        elif encoder_type == 'resnet50':
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=pretrained)
            self.feature_dims = [64, 256, 512, 1024, 2048]
        elif encoder_type == 'resnet101':
            from torchvision.models import resnet101
            backbone = resnet101(pretrained=pretrained)
            self.feature_dims = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet type: {encoder_type}")
        
        # Extract layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Add attention mechanisms
        self.attention_modules = nn.ModuleDict()
        for layer_idx in attention_layers:
            if layer_idx <= 4:
                dim = self.feature_dims[layer_idx]
                self.attention_modules[f'attention_{layer_idx}'] = MultiHeadAttention(dim)
    
    def forward(self, x: torch.Tensor):
        """Forward pass through ResNet encoder with attention"""
        features = []
        attention_maps = []
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features.append(x)
        
        # Layer 1
        x = self.layer1(x)
        if 1 in self.attention_layers and 'attention_1' in self.attention_modules:
            x_attn, attn_map = self.attention_modules['attention_1'](x.flatten(2).transpose(1, 2))
            x = x_attn.transpose(1, 2).view_as(x)
            attention_maps.append(attn_map)
        features.append(x)
        
        # Layer 2
        x = self.layer2(x)
        if 2 in self.attention_layers and 'attention_2' in self.attention_modules:
            x_attn, attn_map = self.attention_modules['attention_2'](x.flatten(2).transpose(1, 2))
            x = x_attn.transpose(1, 2).view_as(x)
            attention_maps.append(attn_map)
        features.append(x)
        
        # Layer 3
        x = self.layer3(x)
        if 3 in self.attention_layers and 'attention_3' in self.attention_modules:
            x_attn, attn_map = self.attention_modules['attention_3'](x.flatten(2).transpose(1, 2))
            x = x_attn.transpose(1, 2).view_as(x)
            attention_maps.append(attn_map)
        features.append(x)
        
        # Layer 4
        x = self.layer4(x)
        if 4 in self.attention_layers and 'attention_4' in self.attention_modules:
            x_attn, attn_map = self.attention_modules['attention_4'](x.flatten(2).transpose(1, 2))
            x = x_attn.transpose(1, 2).view_as(x)
            attention_maps.append(attn_map)
        features.append(x)
        
        return {
            'features': features,
            'attention_maps': attention_maps,
            'global_features': F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        }


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer encoder"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, depth: int = 12, n_heads: int = 12, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, self.n_patches)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, x: torch.Tensor):
        """Forward pass through Vision Transformer"""
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Pass through transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, attn_map = block(x)
            attention_maps.append(attn_map)
        
        # Final layer norm
        x = self.norm(x)
        
        return {
            'features': x,
            'attention_maps': attention_maps,
            'global_features': x[:, 0]  # CLS token
        }


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention"""
    
    def __init__(self, d_model: int, n_heads: int = 8, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self-attention
        attn_output, attn_weights = self.attention(x, mask=mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class MultiScaleFeatureExtractor(nn.Module):
    """Extract multi-scale features from encoder outputs"""
    
    def __init__(self, feature_dims: List[int], output_dim: int = 512):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Feature pyramid network
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for dim in feature_dims:
            self.lateral_convs.append(nn.Conv2d(dim, output_dim, 1))
            self.output_convs.append(nn.Conv2d(output_dim, output_dim, 3, padding=1))
    
    def forward(self, features: List[torch.Tensor]):
        """Extract multi-scale features"""
        # Apply lateral connections
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        # Build feature pyramid
        outputs = []
        for i, (lateral, output_conv) in enumerate(zip(laterals, self.output_convs)):
            if i == 0:
                # Highest resolution
                output = output_conv(lateral)
            else:
                # Upsample and add
                upsampled = F.interpolate(outputs[-1], size=lateral.shape[-2:], mode='nearest')
                output = output_conv(lateral + upsampled)
            outputs.append(output)
        
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(self.ln(x))


class PointCloudDecoder(nn.Module):
    """3D point cloud decoder with attention"""
    
    def __init__(self, feature_dim: int, num_points: int = 4096, 
                 hidden_dim: int = 512, num_layers: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.attention = CrossAttention(hidden_dim)
        self.pos_projection = nn.Linear(3, hidden_dim)
        
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers)])
        self.point_predictor = nn.Linear(hidden_dim, 3)
    
    def forward(self, image_features: torch.Tensor, 
                query_points: Optional[torch.Tensor] = None):
        batch_size = image_features.size(0)
        features = self.feature_proj(image_features)
        if query_points is None:
            query_points = torch.randn(batch_size, self.num_points, 3, device=image_features.device)
            query_points = F.normalize(query_points, p=2, dim=-1)
        query_features = self.pos_projection(query_points)
        attended_features, attention_weights = self.attention(
            query_features,
            features.unsqueeze(1).expand(-1, self.num_points, -1),
            features.unsqueeze(1).expand(-1, self.num_points, -1)
        )
        h0 = attended_features
        for i, block in enumerate(self.blocks):
            attended_features = block(attended_features)
            if (i + 1) % 2 == 0:
                attended_features = attended_features + h0
        predicted_points = self.point_predictor(attended_features)
        return {
            'points': predicted_points,
            'attention_weights': attention_weights
        }


class NeRFDecoder(nn.Module):
    """Neural Radiance Fields (NeRF) decoder"""
    
    def __init__(self, feature_dim: int, pos_encoding_dim: int = 63, 
                 direction_encoding_dim: int = 27, hidden_dim: int = 256, 
                 num_layers: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.pos_encoding_dim = pos_encoding_dim
        self.direction_encoding_dim = direction_encoding_dim
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()
        
        # NeRF MLP
        layers = []
        input_dim = pos_encoding_dim + feature_dim
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers // 2:
                # Skip connection
                layers.append(nn.Linear(hidden_dim + feature_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            layers.append(nn.ReLU())
        
        # Density and color heads
        self.density_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim + direction_encoding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )
        
        self.mlp = nn.ModuleList(layers)
    
    def _create_positional_encoding(self, L: int = 10):
        """Create positional encoding function"""
        def encode(x):
            encoded = [x]
            for i in range(L):
                encoded.append(torch.sin(2 ** i * torch.pi * x))
                encoded.append(torch.cos(2 ** i * torch.pi * x))
            return torch.cat(encoded, dim=-1)
        return encode
    
    def forward(self, positions: torch.Tensor, directions: torch.Tensor, 
                image_features: torch.Tensor):
        """Forward pass through NeRF decoder"""
        batch_size, n_rays, n_samples = positions.shape[:3]
        
        # Positional encoding
        positions_encoded = self.pos_encoding(positions)
        directions_encoded = self.pos_encoding(directions)
        
        # Expand image features
        features_expanded = image_features.unsqueeze(1).unsqueeze(1).expand(
            -1, n_rays, n_samples, -1
        )
        
        # Concatenate features
        x = torch.cat([positions_encoded, features_expanded], dim=-1)
        
        # Pass through MLP
        features_list = []
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == len(self.mlp) // 2 - 1:  # Skip connection
                features_list.append(x)
                x = torch.cat([x, features_expanded], dim=-1)
        
        # Predict density and color
        density = self.density_head(x)
        color = self.color_head(torch.cat([x, directions_encoded], dim=-1))
        
        return {
            'density': density,
            'color': color
        }


class SDFDecoder(nn.Module):
    """Signed Distance Function (SDF) decoder"""
    
    def __init__(self, feature_dim: int, pos_encoding_dim: int = 63, 
                 hidden_dim: int = 512, num_layers: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.pos_encoding_dim = pos_encoding_dim
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()
        
        # SDF MLP
        layers = []
        input_dim = pos_encoding_dim + feature_dim
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            layers.append(nn.ReLU())
        
        # Final SDF prediction
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def _create_positional_encoding(self, L: int = 10):
        """Create positional encoding function"""
        def encode(x):
            encoded = [x]
            for i in range(L):
                encoded.append(torch.sin(2 ** i * torch.pi * x))
                encoded.append(torch.cos(2 ** i * torch.pi * x))
            return torch.cat(encoded, dim=-1)
        return encode
    
    def forward(self, positions: torch.Tensor, image_features: torch.Tensor):
        """Forward pass through SDF decoder"""
        # Positional encoding
        positions_encoded = self.pos_encoding(positions)
        
        # Expand image features
        features_expanded = image_features.unsqueeze(1).expand(-1, positions.size(1), -1)
        
        # Concatenate features
        x = torch.cat([positions_encoded, features_expanded], dim=-1)
        
        # Predict SDF values
        sdf_values = self.mlp(x)
        
        return {
            'sdf': sdf_values
        }


class OccupancyDecoder(nn.Module):
    """Occupancy network decoder"""
    
    def __init__(self, feature_dim: int, pos_encoding_dim: int = 63, 
                 hidden_dim: int = 512, num_layers: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.pos_encoding_dim = pos_encoding_dim
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()
        
        # Occupancy MLP
        layers = []
        input_dim = pos_encoding_dim + feature_dim
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            layers.append(nn.ReLU())
        
        # Final occupancy prediction
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def _create_positional_encoding(self, L: int = 10):
        """Create positional encoding function"""
        def encode(x):
            encoded = [x]
            for i in range(L):
                encoded.append(torch.sin(2 ** i * torch.pi * x))
                encoded.append(torch.cos(2 ** i * torch.pi * x))
            return torch.cat(encoded, dim=-1)
        return encode
    
    def forward(self, positions: torch.Tensor, image_features: torch.Tensor):
        """Forward pass through occupancy decoder"""
        # Positional encoding
        positions_encoded = self.pos_encoding(positions)
        
        # Expand image features
        features_expanded = image_features.unsqueeze(1).expand(-1, positions.size(1), -1)
        
        # Concatenate features
        x = torch.cat([positions_encoded, features_expanded], dim=-1)
        
        # Predict occupancy values
        occupancy_values = self.mlp(x)
        
        return {
            'occupancy': occupancy_values
        }


class Enhanced3DReconstructionModel(nn.Module):
    """Enhanced 3D reconstruction model with multiple representations"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.encoder_type = config.get('encoder_type', 'resnet50')
        self.representation = config.get('representation', 'pointcloud')
        self.hidden_dim = config.get('hidden_dim', 512)
        
        # Build encoder
        if 'resnet' in self.encoder_type:
            self.encoder = ResNetEncoder(
                encoder_type=self.encoder_type,
                pretrained=config.get('pretrained', True),
                attention_layers=config.get('attention_layers', [2, 3, 4])
            )
            self.feature_dim = self.encoder.feature_dims[-1]
        elif 'vit' in self.encoder_type:
            self.encoder = VisionTransformerEncoder(
                img_size=config.get('img_size', 224),
                patch_size=config.get('patch_size', 16),
                embed_dim=config.get('embed_dim', 768),
                depth=config.get('depth', 12),
                n_heads=config.get('n_heads', 12)
            )
            self.feature_dim = self.encoder.embed_dim
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")
        
        # Build decoder based on representation
        if self.representation == 'pointcloud':
            num_points = config.get('num_points', config.get('pointcloud_size', 4096))
            self.decoder = PointCloudDecoder(
                feature_dim=self.feature_dim,
                num_points=num_points,
                hidden_dim=self.hidden_dim
            )
        elif self.representation == 'nerf':
            self.decoder = NeRFDecoder(
                feature_dim=self.feature_dim,
                pos_encoding_dim=config.get('pos_encoding_dim', 63),
                direction_encoding_dim=config.get('direction_encoding_dim', 27),
                hidden_dim=self.hidden_dim
            )
        elif self.representation == 'sdf':
            self.decoder = SDFDecoder(
                feature_dim=self.feature_dim,
                pos_encoding_dim=config.get('pos_encoding_dim', 63),
                hidden_dim=self.hidden_dim
            )
        elif self.representation == 'occupancy':
            self.decoder = OccupancyDecoder(
                feature_dim=self.feature_dim,
                pos_encoding_dim=config.get('pos_encoding_dim', 63),
                hidden_dim=self.hidden_dim
            )
        else:
            raise ValueError(f"Unsupported representation: {self.representation}")
        
        # Uncertainty quantification
        if config.get('uncertainty_quantification', False):
            self.uncertainty_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, images: torch.Tensor, 
                query_points: Optional[torch.Tensor] = None,
                directions: Optional[torch.Tensor] = None):
        """Forward pass through the model"""
        # Encode images
        encoder_output = self.encoder(images)
        
        if 'resnet' in self.encoder_type:
            global_features = encoder_output['global_features']
        else:  # vit
            global_features = encoder_output['global_features']
        
        # Decode to 3D representation
        if self.representation == 'pointcloud':
            decoder_output = self.decoder(global_features, query_points)
        elif self.representation == 'nerf':
            decoder_output = self.decoder(query_points, directions, global_features)
        else:  # sdf or occupancy
            decoder_output = self.decoder(query_points, global_features)
        
        # Add uncertainty if enabled
        if hasattr(self, 'uncertainty_head'):
            uncertainty = self.uncertainty_head(global_features)
            decoder_output['uncertainty'] = uncertainty
        
        # Add encoder features for analysis
        decoder_output['encoder_features'] = encoder_output
        
        return decoder_output
    
    def get_attention_maps(self, images: torch.Tensor):
        """Get attention maps from the encoder"""
        encoder_output = self.encoder(images)
        return encoder_output.get('attention_maps', [])


def create_enhanced_model(config: Dict) -> Enhanced3DReconstructionModel:
    """Create enhanced 3D reconstruction model"""
    return Enhanced3DReconstructionModel(config)


def test_enhanced_model():
    """Test the enhanced model"""
    print("Testing enhanced 3D reconstruction model...")
    
    # Test configurations
    test_configs = [
        {
            'encoder_type': 'resnet50',
            'representation': 'pointcloud',
            'hidden_dim': 512,
            'num_points': 1024,
            'attention_layers': [2, 3, 4]
        },
        {
            'encoder_type': 'vit',
            'representation': 'sdf',
            'embed_dim': 768,
            'depth': 6,
            'n_heads': 8,
            'hidden_dim': 512,
            'pos_encoding_dim': 63
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTesting configuration {i+1}: {config}")
        
        try:
            # Create model
            model = create_enhanced_model(config)
            model.eval()
            
            # Test input
            batch_size = 2
            img_size = 224
            test_images = torch.randn(batch_size, 3, img_size, img_size)
            
            # Forward pass
            with torch.no_grad():
                if config['representation'] == 'pointcloud':
                    output = model(test_images)
                    print(f"Point cloud shape: {output['points'].shape}")
                elif config['representation'] == 'sdf':
                    n_queries = 100
                    query_points = torch.randn(batch_size, n_queries, 3)
                    output = model(test_images, query_points)
                    print(f"SDF values shape: {output['sdf'].shape}")
                elif config['representation'] == 'nerf':
                    n_rays = 10
                    n_samples = 32
                    query_points = torch.randn(batch_size, n_rays, n_samples, 3)
                    directions = torch.randn(batch_size, n_rays, n_samples, 3)
                    output = model(test_images, query_points, directions)
                    print(f"NeRF density shape: {output['density'].shape}")
                    print(f"NeRF color shape: {output['color'].shape}")
                elif config['representation'] == 'occupancy':
                    n_queries = 100
                    query_points = torch.randn(batch_size, n_queries, 3)
                    output = model(test_images, query_points)
                    print(f"Occupancy values shape: {output['occupancy'].shape}")
            
            # Test attention maps
            attention_maps = model.get_attention_maps(test_images)
            print(f"Number of attention maps: {len(attention_maps)}")
            
            print(f"Configuration {i+1} test passed!")
            
        except Exception as e:
            print(f"Configuration {i+1} test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nEnhanced model testing completed!")


if __name__ == "__main__":
    test_enhanced_model()
