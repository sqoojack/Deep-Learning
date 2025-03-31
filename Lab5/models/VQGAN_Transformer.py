
""" MaskGit 跟 VQGAN的結合:
    1.	圖像編碼：
	    首先, VQGAN 的編碼器將圖像轉換為潛在表示, 並通過量化得到離散的碼本索引(token)。這些 token 代表了圖像的壓縮表示。
	2.	圖像生成：
	•	然後，這些 token 作為 MaskGIT 的輸入。MaskGIT 使用它的 Bidirectional Transformer 根據遮罩策略逐步預測和填充這些 token, 以生成新的圖像表示。
	3.	圖像解碼：
	•	最後，這些由 MaskGIT 生成的 token 會被送回 VQGAN 的解碼器，解碼器將它們轉換回高質量的圖像。"""

""" tips: 不要亂創副本 -> 會容易導致cuda out of memory """

# Reference: https://github.com/KJLdefeated/NYCU_DLP_2024

import torch.nn as nn
import torch
import yaml # 用來讀入, 寫入yaml文件
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F

# TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs["VQ_Configs"]) # 初始化VQGAN model, 將其存儲在self.vqgan (下面會定義load_vqgan)
        
        self.num_image_tokens = configs["num_image_tokens"]
        self.mask_token_id = configs["num_codebook_vectors"]
        self.choice_temperature = configs["choice_temperature"]
        self.gamma = self.gamma_func(configs["gamma_type"])
        self.transformer = BidirectionalTransformer(configs["Transformer_param"])
        
    def load_transformer_checkpoint(self, load_ckpt_path):  # 在inpainting時會用到, 用來加載已訓練的transformer model 的權重
        self.transformer.load_state_dict(torch.load(load_ckpt_path))
        
    @staticmethod # @staticmethod: 表示load_vqgan方法是一個靜態方法 -> 與class的實例無關
    def load_vqgan(configs):
        # 打開yaml配置文件 並以read mode 讀取文件, yaml.safe_load: 將該yaml文件的內容加載為python字典cfg, cfg包含了VQGAN model的各種配置參數
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))  
        model = VQGAN(cfg['model_param'])   # 初始化了一個 VQGAN model的實例
        model.load_state_dict(torch.load(configs['VQ_CKPT_path'], weights_only=True), strict=True) # 加載以訓練的權重, torch.load: 從該路徑加載權重file
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
# 將輸入圖像轉換為VQGAN的潛在表示和量化表示
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        return codebook_mapping, codebook_indices
    
    def gamma_func(self, mode='cosine'):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == 'Linear':
            return lambda x: 1 - x  # x為輸入的ratio
        elif mode == 'cosine':
            return lambda x: np.cos(x * np.pi / 2)
        elif mode == 'square':
            return lambda x: 1 - x ** 2
        else:
            raise NotImplementedError
        
##TODO2 step1-3:
    """ 將輸入圖像進行編碼、遮罩處理，然後通過 Transformer 模型進行預測，最終返回預測的 logits 和對應的 ground truth """
    def forward(self, x, ratio):
        _, z_indices = self.encode_to_z(x)  # 編碼輸入圖片，獲取對應的量化索引
        z_indices = z_indices.view(-1, self.num_image_tokens)   # 將 z_indices 形狀調整為 (batch_size, num_image_tokens)
        
        # 創建遮罩，並將 z_indices 中被遮罩的位置設為 mask_token_id
        mask = torch.rand_like(z_indices, dtype=torch.float) < ratio  # 隨機遮罩，probability of masking is `ratio`
        z_indices_input = z_indices.masked_fill(mask, self.mask_token_id)  # 用 masked_fill 替換被遮罩的元素
        
        logits = self.transformer(z_indices_input)  # 使用 Transformer 預測被遮罩位置的 token
        logits = logits[..., :self.mask_token_id]  # 只保留與 mask_token_id 前的 logits
        
        # 使用 one-hot 編碼生成 ground truth
        ground_truth = F.one_hot(z_indices, num_classes=self.mask_token_id).float()
        
        return logits, ground_truth
    
## TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(self, x, ratio, mask_b): # mask_b: 是boolean變數, 用來標記哪些位置在當前在當前步驟中需要被遮罩或處理
        # 跟forward幾行類似        
        _, z_indices = self.encode_to_z(x)
        z_indices_input = torch.where(mask_b == 1, torch.tensor(self.mask_token_id).to(mask_b.device), z_indices)
        
        logits = self.transformer(z_indices_input)
        logits = F.softmax(logits, dim=-1)   # softmax將logits轉換為概率分佈 -> 才可以用torch.max()之類的方法, 找到每個位置上最有可能的token值
        
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)   # 尋找每個token預測值的最大概率
        
        ratio = self.gamma(ratio)   # 調整masking ratio
        # torch.log(torch.rand_like(...)): 計算的是隨機數的自然對數，這個自然對數通常是負數或零 -> 取"-"後變成正數
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob))) # 生成gumbel噪聲 -> 常用於採樣的噪聲, 並引入隨機性
        
        temperature = self.choice_temperature * (1 - ratio)  # 溫度越高, 隨機性越大, 溫度越低, 結果越確定
        confidence = z_indices_predict_prob + temperature * gumbel_noise    # 計算信心值 -> 有效避免model過度確定某些預測結果
        
        confidence[mask_b == 0] = float('inf')  # 將未遮罩的部分的信心值設為無限大
        
        # item(): 將張量（Tensor）中的單個元素提取出來, 並將其轉換為純 Python 數值類型, 如 int 或 float
        n = max(1, int(mask_b.sum().item() * ratio))
        _, idx_to_mask = torch.topk(confidence, n, largest=False)
        
        # 生成新的遮罩 mask_bc, 並對其進行更新, 以便確定哪些位置應該在當前步驟中被替換或保留
        mask_bc = torch.zeros_like(mask_b, dtype = torch.bool)   # 形狀跟mask_b相同
        mask_bc.scatter_(1, idx_to_mask, 1) # 在指定的維度 (第1維) 上進行元素更新
        mask_bc = mask_bc & mask_b  # 只有那些在 mask_b 中已經被遮罩, 且在新的遮罩中也被選中的位置，才會保留 True
        
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
