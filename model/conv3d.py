import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv3D(nn.Module):
    """简单的3D卷积网络，输入输出维度相同"""
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(SimpleConv3D, self).__init__()
        
        # 编码器
        self.enc1 = self._make_block(in_channels, base_channels)
        self.enc2 = self._make_block(base_channels, base_channels * 2)
        
        # 中间层
        self.mid = self._make_block(base_channels * 2, base_channels * 4)
        
        # 解码器
        self.dec1 = self._make_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._make_block(base_channels * 2, base_channels)
        
        # 输出层
        self.out_conv = nn.Conv3d(base_channels, out_channels, 3, padding=1)
        
        # 池化和上采样
        self.pool = nn.MaxPool3d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def _make_block(self, in_ch, out_ch):
        """创建卷积块"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)      # [B, C, D, H, W]
        enc1_pool = self.pool(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool(enc2)
        
        # 中间层
        mid = self.mid(enc2_pool)
        
        # 解码路径
        dec1 = self.dec1(mid)
        dec1_up = self.upsample(dec1)
        
        # 跳跃连接
        dec2_input = dec1_up + enc2
        dec2 = self.dec2(dec2_input)
        dec2_up = self.upsample(dec2)
        
        # 最终输出
        dec3_input = dec2_up + enc1
        output = self.out_conv(dec3_input)
        
        return output


class BasicConv3D(nn.Module):
    """基础3D卷积网络（更简单版本）"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(BasicConv3D, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 下采样
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # 上采样
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            
            # 输出
            nn.Conv3d(32, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        return self.conv_layers(x)