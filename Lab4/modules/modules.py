import torch.nn as nn
import torch
from .layers import DepthConvBlock, ResidualBlock
from torch.autograd import Variable

# __all__ : 當使用 "from modules import * "語句進行導入時，哪些名稱會被導入
__all__ = [
    "Generator",
    "RGB_Encoder",
    "Gaussian_Predictor",
    "Decoder_Fusion",
    "Label_Encoder"
]

class Generator(nn.Sequential):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__(
            DepthConvBlock(input_nc, input_nc),
            ResidualBlock(input_nc, input_nc//2),
            DepthConvBlock(input_nc//2, input_nc//2),
            ResidualBlock(input_nc//2, input_nc//4),
            DepthConvBlock(input_nc//4, input_nc//4),
            ResidualBlock(input_nc//4, input_nc//8),
            DepthConvBlock(input_nc//8, input_nc//8),
            nn.Conv2d(input_nc//8, 3, 1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class RGB_Encoder(nn.Sequential):   # 編碼 RGB 影像, use Residualblock and 深度卷積塊來進行特徵提取
    def __init__(self, in_chans, out_chans):
        super(RGB_Encoder, self).__init__(
            ResidualBlock(in_chans, out_chans//8),
            DepthConvBlock(out_chans//8, out_chans//8),
            ResidualBlock(out_chans//8, out_chans//4),  # 出來時變2倍
            DepthConvBlock(out_chans//4, out_chans//4),
            ResidualBlock(out_chans//4, out_chans//2),
            DepthConvBlock(out_chans//2, out_chans//2),
            nn.Conv2d(out_chans//2, out_chans, 3, padding=1),   # 最後做convolution
        )  
        
    def forward(self, image):
        return super().forward(image)
    
    
class Label_Encoder(nn.Sequential):
    def __init__(self, in_chans, out_chans, norm_layer=nn.BatchNorm2d):
        super(Label_Encoder, self).__init__(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_chans, out_chans//2, kernel_size=7, padding=0),
            norm_layer(out_chans//2),
            nn.LeakyReLU(True),
            ResidualBlock(in_ch=out_chans//2, out_ch=out_chans)
        )  
        
    def forward(self, image):
        return super().forward(image)
    
    
class Gaussian_Predictor(nn.Sequential):
    def __init__(self, in_chans=48, out_chans=96):
        super(Gaussian_Predictor, self).__init__(
            ResidualBlock(in_chans, out_chans//4),
            DepthConvBlock(out_chans//4, out_chans//4),
            ResidualBlock(out_chans//4, out_chans//2),
            DepthConvBlock(out_chans//2, out_chans//2),
            ResidualBlock(out_chans//2, out_chans),
            nn.LeakyReLU(True),
            nn.Conv2d(out_chans, out_chans*2, kernel_size=1)
        )
        
    def reparameterize(self, mu, logvar):   # 重參數化, mu: 均值, logvar: log variance
        std = torch.exp(logvar / 2)  # 開根號
        eplison = torch.rand_like(std)  # 生成與標準差形狀相同的隨機數據
        return mu + eplison * std

    def forward(self, img, label):
        feature = torch.cat([img, label], dim=1)    # 沿著第一維進行拼接
        """ ex: A:[3, 2] B:[3, 4] -> A和B拼接後變 [3, 6] """
        parm = super().forward(feature)
        mu, logvar = torch.chunk(parm, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar
    
    
class Decoder_Fusion(nn.Sequential):    # Fusion: 融合
    def __init__(self, in_chans=48, out_chans=96):
        super().__init__(
            DepthConvBlock(in_chans, in_chans),
            ResidualBlock(in_chans, in_chans//4),
            DepthConvBlock(in_chans//4, in_chans//2),
            ResidualBlock(in_chans//2, in_chans//2),
            DepthConvBlock(in_chans//2, out_chans//2),
            nn.Conv2d(out_chans//2, out_chans, 1, 1)
        )
        
    def forward(self, img, label, parm):
        feature = torch.cat([img, label, parm], dim=1)
        return super().forward(feature)