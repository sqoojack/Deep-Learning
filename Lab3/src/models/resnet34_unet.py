
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import DoubleConvBlock, up_block 

# Reference: https://ithelp.ithome.com.tw/m/articles/10333931

""" Upsampling then double conv """
class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, up_in_channel=None, up_out_channel=None):
        super().__init__()
        
        if up_in_channel == None:
            up_in_channel = in_channels
        if up_out_channel == None:
            up_out_channel = out_channels
            
        self.up = nn.ConvTranspose2d(up_in_channel, up_out_channel, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels, out_channels)

        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)  # 將上採樣後的輸出(x1) 與 來自編碼器的對應層輸出(x2) 拼接
        return self.conv(x)
        
# is_downsample: 是否需要進行下採樣, 如果會的話input data的維度將會縮小, 下採樣由一個1x1卷積層, 批量歸一層組成
class ResidualBlock(nn.Module): # 用於逐步提取圖像的高層特徵
    def __init__(self, in_channels, out_channels, stride=1, is_downsample=False):
        super(ResidualBlock, self).__init__()
        
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.is_downsample = is_downsample
        
        if self.is_downsample:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2), # stride = 2: 表示特徵圖的高度,寬度會減少一半
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        if self.is_downsample:
            residual = self.down_sample(x)  # 如果需要, 對x進行下採樣
        else:
            residual = x
        
        x = self.conv_x(x)
        # print(f"x: {x.shape}, residual: {residual.shape}")
        
        x = residual + x # residual connection, 目的是允許信息繞過卷積層直接傳遞到後面的層 -> 減少梯度消失問題, 增加梯度穩定度
        x = F.relu(x)
        return x
    
class ResNet_UNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_UNet, self).__init__()
        
        self.encoder1 = nn.Sequential(  # kernel_size=7: 論文敘述
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), # padding: 填充特徵圖的邊緣像素
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.encoder2 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        
        self.encoder3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, is_downsample=True),   # 縮小空間尺寸並增加通道數
            ResidualBlock(128, 128),    # ResidualBlock次數, 論文有提供
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        
        self.encoder4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, is_downsample=True),  
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        
        self.encoder5 = nn.Sequential(
            ResidualBlock(256, 512, 2, True),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )
        
        
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder1 = DecoderBlock(in_channels=1024, out_channels=512)
        self.decoder2 = DecoderBlock(512, 256)  # DecoderBlock內部會幫你拼接
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        # in_channel為128是因為: 來自前一個解碼器步驟的 upsampling 輸出(64) + encoder對應層的特徵圖(64)
        self.decoder5 = DecoderBlock(in_channels=128, out_channels=64, up_in_channel=64, up_out_channel=64)
        
        self.lastlayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1, bias=False)
        )
        self.drop_out = nn.Dropout(0.5)
    def forward(self, x):
        e1 = self.encoder1(x)
        pool1 = self.pool(e1)
        e2 = self.encoder2(pool1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        bridge = self.bridge(e5)
        
        d1 = self.decoder1(bridge, e5)  # 兩個會在通道維度上進行拼接 (torch.cat([x1, x2], dim=1))
        d1 = self.drop_out(d1)  
        d2 = self.decoder2(d1, e4)
        d2 = self.drop_out(d2)  
        d3 = self.decoder3(d2, e3)
        d3 = self.drop_out(d3)  
        d4 = self.decoder4(d3, e2)
        d4 = self.drop_out(d4)  
        d5 = self.decoder5(d4, e1)
        
        out = self.lastlayer(d5)
        
        return out
