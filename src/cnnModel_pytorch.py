import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for focusing on important image regions.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.w_o(output)
        
        return output, attention_weights

class SpatialAttention(nn.Module):
    """
    Spatial attention for focusing on important image regions.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention_weights = self.sigmoid(self.conv(x))
        return x * attention_weights

class ChannelAttention(nn.Module):
    """
    Channel attention for focusing on important feature channels.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        attention_weights = self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)
        return x * attention_weights

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(channels)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with attention for better gradient flow.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.cbam = CBAM(out_channels)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += residual
        out = F.relu(out)
        
        return out

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(in_channels, out_channels, 1)
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        ])
        
    def forward(self, features):
        # features should be a list of feature maps at different scales
        lateral_features = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            lateral_features.append(lateral_conv(features[i]))
        
        # Top-down pathway
        fpn_features = []
        for i in range(len(lateral_features) - 1, -1, -1):
            if i == len(lateral_features) - 1:
                fpn_feature = lateral_features[i]
            else:
                # Upsample and add
                upsampled = F.interpolate(fpn_features[-1], size=lateral_features[i].shape[-2:], mode='nearest')
                fpn_feature = lateral_features[i] + upsampled
            
            fpn_feature = self.fpn_convs[i](fpn_feature)
            fpn_features.insert(0, fpn_feature)
        
        return fpn_features

class EnhancedCNNModel(nn.Module):
    """
    Enhanced CNN model with attention mechanisms for complex image understanding.
    """
    def __init__(self, img_height, img_width, channels, output_vertices, 
                 attention_heads=8, d_model=512, dropout=0.1):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.output_vertices = output_vertices
        self.d_model = d_model
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks with attention
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(512, 256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-head attention for feature refinement
        self.feature_projection = nn.Linear(512, d_model)
        self.attention = MultiHeadAttention(d_model, attention_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_vertices * 3)  # 3 coordinates per vertex
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual layers
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        # Feature Pyramid Network
        fpn_features = self.fpn([x1, x2, x3])
        
        # Use the finest scale features
        x = fpn_features[-1]
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        # Project to attention dimension
        x = self.feature_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Multi-head attention
        attended_features, attention_weights = self.attention(x)
        x = self.layer_norm1(x + attended_features)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        # Final prediction
        x = x.squeeze(1)
        x = self.final_layers(x)
        
        return x

def create_enhanced_model(img_height, img_width, channels, output_vertices, 
                         attention_heads=8, d_model=512, device='cuda'):
    """
    Create and return an enhanced CNN model with attention.
    
    Args:
        img_height (int): Height of input images
        img_width (int): Width of input images
        channels (int): Number of input channels (3 for RGB)
        output_vertices (int): Number of output vertices
        attention_heads (int): Number of attention heads
        d_model (int): Dimension of the model
        device (str): Device to place the model on ('cuda' or 'cpu')
    
    Returns:
        EnhancedCNNModel: The created enhanced model
    """
    model = EnhancedCNNModel(
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        output_vertices=output_vertices,
        attention_heads=attention_heads,
        d_model=d_model
    )
    model = model.to(device)
    return model
