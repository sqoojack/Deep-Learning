
import torch.nn as nn
import torch
from .modules import Encoder, TokenPredictor

def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    
class BidirectionalTransformer(nn.Module):  # 雙向Transformer
    def __init__(self, configs):    # configs: 用來存儲許多超參數, 在config資料夾裡面有設定
        super(BidirectionalTransformer, self).__init__()
        self.num_image_tokens = configs['num_image_tokens']     # 從 configs 中提取出 num_image_tokens 並存儲到類別屬性中, 這邊為"256"
        
        """ token, pose Embedding. nn.Embedding: 將索引映射到向量, num_embeddings: 可映射的索引範圍, embedding_dim: 每個索引被映射到的向量的長度 """
        self.tok_emb = nn.Embedding(num_embeddings=configs['num_codebook_vectors'] + 1, embedding_dim=configs['dim'])    #  num_codebook_vectors: 可用的token數目, +1表示特殊的token 例如padding token
        # nn.Parameter: 代表它是一個可學習的參數, 會在model training中自動更新 -> 允許model學習每個token的位置信息
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(configs['num_image_tokens'], configs['dim'])), 0., 0.02)
        
        self.blocks = nn.Sequential(*[Encoder(configs['dim'], configs['hidden_dim']) for _ in range(configs['n_layers'])])
        self.Token_Prediction = TokenPredictor(configs['dim'])
        self.LN = nn.LayerNorm(configs['dim'], eps=1e-12)
        self.drop = nn.Dropout(p=0.1)
        
        self.bias = nn.Parameter(torch.zeros(self.num_image_tokens, configs['num_codebook_vectors'] + 1))
        self.apply(weights_init)

    def forward(self, x):
        # Token domain -> Latent domain
        token_embeddings = self.tok_emb(x)

        embed = self.drop(self.LN(token_embeddings + self.pos_emb))
        embed = self.blocks(embed)
        embed = self.Token_Prediction(embed)

        # Latent domain -> Token domain
        logits = torch.matmul(embed, self.tok_emb.weight.T) + self.bias

        return logits