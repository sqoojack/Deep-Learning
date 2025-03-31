# Implement your UNet model here
# Reference: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# Reference: https://medium.com/unet%E8%88%87fcn%E7%B3%BB%E5%88%97/unet%E5%B0%8F%E7%B0%A1%E4%BB%8B-%E5%AF%A6%E4%BD%9C2d-unet-5f9ef3d91e4b

""" checkpoint: 可以在forward pass中用來節省memory """
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels    # n_channels: model一開始的輸入通道數
        self.n_classes = n_classes  # n_classes: model的最終輸出通道數
        
        
        self.inC = (DoubleConvBlock(n_channels, 64))    # 先做一次DoubleConvBlock, 之後再做down
        self.down1 = (down_block(64, 128))
        self.down2 = (down_block(128, 256))
        self.down3 = (down_block(256, 512))
        self.down4 = (down_block(512, 1024))
        self.drop_out = nn.Dropout(0.5)
        self.up1 = (up_block(1024, 512))
        self.up2 = (up_block(512, 256))
        self.up3 = (up_block(256, 128))
        self.up4 = (up_block(128, 64))
        self.outC = (OutConv(64, 2))
        
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x1 = self.inC(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.drop_out(x5)  # Dropout applied to the bottleneck layer (以防止Overfitting)
        
        x = self.up1(x5, x4)    # 上採樣有兩個參數: 將下採樣過程中的特徵圖與上採樣過程中的特徵圖結合，以保留更多的空間信息
        # print(f"After up1: {x.shape}")        // debugging line
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outC(x)
        # print(f"Output shape: {output.shape}")  // debugging line
        return output
        
# 圖像維度減少2: 因為卷積核是3x3 且沒有進行零填充
""" stride: 步長, 代表卷積核每次移動 1 像素, 默認值為1. dilation: 膨脹率, 表示卷積核內部元素間的距離, 默認為1
    padding: 填充, 在輸入特徵圖的邊緣填充額外的像素(填充值為0), 默認為0 表示不填充 """
    
class DoubleConvBlock(nn.Module):   
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, padding=1): # out_channel
        super(DoubleConvBlock, self).__init__()
        
        self.in_channels = in_channels   # 用來保存初始化時傳入的參數
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        
        # bias = False: 因為BatchNorm中就提供了Bias的效果，所以這裡就不需要了
        conv1 = nn.Conv2d(int(self.in_channels), int(self.out_channels), kernel_size=3, stride=1, padding=int(self.padding), bias=False)
        bn1 = nn.BatchNorm2d(int(self.out_channels), affine=False)     # 做正則化, affine: 決定了該層是否有可學習的縮放, 平移參數(gamma and beta)
        relu1 = nn.ReLU(inplace = True)
        # 第二次convolution時, 進去跟出來的channel不變
        conv2 = nn.Conv2d(int(self.out_channels), int(self.out_channels), kernel_size=3, stride=1, padding=int(self.padding), bias=False)
        bn2 = nn.BatchNorm2d(int(self.out_channels), affine=False)
    
        relu2 = nn.ReLU(inplace = True)  # inplace: 代表是否創建一個新的Tensor來儲存ReLU後的數據, True代表直接在輸入數據上進行(inplace)
        
        UNet_block_list = []    # 創一個列表, 儲存一系列的層操作
        UNet_block_list.append(conv1)
        UNet_block_list.append(bn1)
        UNet_block_list.append(relu1)   
        UNet_block_list.append(conv2)   
        UNet_block_list.append(bn2)       
        UNet_block_list.append(relu2)
        self.net = nn.Sequential(*UNet_block_list)  # *為解包運算符, 將list中每個元素作為獨立的參數傳給nn.Sequential
        
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class down_block(nn.Module):  # 下降階段, 先做一個maxpooling 再做DoubleConv 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2) # 縮小2倍
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

    
class up_block(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
    
        if bilinear:    # 放大2倍, align_corners: 輸入和輸出tensor的角點會對齊
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)    
        else:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = DoubleConvBlock(in_channels + in_channels // 2, out_channels)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)    # 對 x1 做Upsample
        
        # print(f"x1 shape after upsample: {x1.shape}")      // debugging line
        # print(f"x2 shape from skip connection: {x2.shape}")   // debugging line
        
        # input is CHW
        diffX = x2.size()[3] - x1.size()[3] # 計算寬度差 (x) (第三個維度, width)
        diffY = x2.size()[2] - x1.size()[2] # 計算高度差 (y)
        
        # 填充x1 以匹配x2 的尺寸
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]) 
        x = torch.cat([x2, x1], dim=1)  # 將 x2 和填充後的 x1 沿著通道維度（dim=1）進行拼接
        
        return self.conv(x) # 最後做DoubleConv
    
class OutConv(nn.Module):   # 最後只做一次Conv
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 這裡卷積核大小只有1x1
    
    def forward(self, x):
        return self.conv(x)                   