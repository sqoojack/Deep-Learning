
""" Reference: https://github.com/hank891008/Deep-Learning """

import torch
import torch.nn as nn
from diffusers import UNet2DModel

class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes=24, dim=512):
        super(ConditionalDDPM, self).__init__()
        channel = dim // 4
        
        self.ddpm = UNet2DModel(
            sample_size=64,   # 輸入影像大小為64 x 64
            in_channels=3,  # RGB圖像
            out_channels=3,
            layers_per_block=2, # 每個block中包含2層
            
            # 設定每個 block 中的通道數量，依次為 channel, channel, channel*2, channel*2, channel*4, channel*4
            block_out_channels=(channel, channel, channel * 2, channel * 2, channel * 4),
            # 定義下採樣 block 的類型，包含多個 DownBlock2D 和一個 AttnDownBlock2D 
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            # class_embed_type="projection",   
            class_embed_type="identity",    # 設定嵌入類型, 表示直接使用輸入的class embedding
        )
        
        self.class_embedding = nn.Sequential(
            nn.Linear(num_classes, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim)     # 對輸出進行正規化
        )
        
    def forward(self, x, t, label):
        # 計算class embedding
        class_embed = self.class_embedding(label)   # 將類別標籤轉換為dim的向量
        return self.ddpm(x, t, class_embed).sample     # 使用ddpm model進行forward pass, 並返回生成的樣本

if __name__ == "__main__":
    model = ConditionalDDPM()   # 初始化ddpm模型
    print(model)    # 印出model結構
    
    # 創建模擬輸入
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.randint(0, 1000, (batch_size, ))
    label = torch.randint(0, 2, (batch_size, 24)).float()
    
    # 運行模型
    output = model(x, t, label)
    print(f"Output shape: {output.shape}")