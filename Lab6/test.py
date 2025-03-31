
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
# make_grid: 將多個圖像拼接成一個網格圖像 (grid), save_image: 將Tensor轉為圖像並保存到disk中
from torchvision.utils import make_grid, save_image

from diffusers import DDPMScheduler
from dataset import iclevrDataset
from model.DDPM import ConditionalDDPM

from evaluator import evaluation_model
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("result", exist_ok=True)
os.makedirs("result/test", exist_ok=True)
os.makedirs("result/new_test", exist_ok=True)

""" 固定seed設置, 使每次結果都相同 """
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 取得噪聲排程器, timesteps: 排程器中的總步數, 'squaredcos_cap_v2': 一種排程方法, 根據cos函數來設定每個步驟中的噪聲量
def get_scheduler(timesteps):
    return DDPMScheduler(num_train_timesteps=timesteps, beta_schedule='squaredcos_cap_v2') # 控制去噪過程中的添加和去除

def load_model(ckpt):
    model = ConditionalDDPM().to(device)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

""" 通過模型逐步去噪來生成影像, 並且在每個指定步驟保存中間結果 """
def generate_images(dataloader, noise_scheduler, model, eval_model, save_prefix):   # x: model生成的影像, y: 目標標籤
    all_results = []
    acc = []    # accuracy
    progress_bar = tqdm(dataloader, ncols=90)
    
    for idx, y in enumerate(progress_bar):  # 遍歷每個batch
        y = y.to(device)        #   生成一個隨機噪聲影像
        x = torch.randn((y.shape[0], 3, 64, 64)).to(device)
        denoising_result = []   #  用於存儲每個去噪步驟步驟的中間影像
        
        for i, t in enumerate(noise_scheduler.timesteps):   # 遍歷所有去噪過程的時間點
            with torch.no_grad():
                residual = model(x, t.to(device), y)   # model: 自己訓練好的模型, residual: 這個殘差表示如何修正當前影像x
            x = noise_scheduler.step(residual, t, x).prev_sample    # 用殘差來更新影像x
            
            if i % (len(noise_scheduler.timesteps) // 10) == 0:     # 每隔10%的步驟儲存一次中間結果
                denoising_result.append(x.cpu())   # 移除批次梯度並加到denoising_result
                
        eval_acc = eval_model.eval(x, y)
        acc.append(eval_acc)   # 評估生成的影像x是否滿足y, 並加入到acc
        progress_bar.set_postfix_str(f"image: {idx}, accuracy: {eval_acc:.4f}")   # acc[-1]: 最新的準確率
        
        denoising_result.append(x.cpu())   # 最後生成的影像也加入denoising_result
        
        # 檢查每個張量的形狀
        # for i, tensor in enumerate(denoising_result):
            # print(f"張量 {i} 形狀: {tensor.shape}")

        denoising_result = [tensor.squeeze(0) if tensor.dim() == 4 else tensor for tensor in denoising_result]   # 確保所有張量都是 3 通道的
        grid = make_grid(torch.stack(denoising_result).clamp(-1, 1), nrow=len(denoising_result))
        
        save_image((grid + 1) / 2, f"result/{save_prefix}/{save_prefix}_{idx}.png")
        all_results.append(x.cpu())    # 將最終生成影像加入all_results
        
    # 將整個去噪過程的結果保存為一張單一的網格圖像
    grid = make_grid(torch.cat(all_results, dim=0).clamp(-1, 1), nrow=8)
    save_image((grid + 1) / 2, f"result/{save_prefix}/{save_prefix}_grid.png")
    return acc

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(100)

if __name__ == "__main__":
    set_seed(100)
    # ckpt = 'checkpoints/checkpoint_23(best).pth'
    ckpt = "checkpoints/DL_lab6_313552049_鄭博涵.pth"   # 要將上傳的權重model存在checkpoint裡面
    timesteps = 1000
    model = load_model(ckpt)
    noise_scheduler = get_scheduler(timesteps)
    eval_model = evaluation_model()
    
    """ 在讀test, new_test的時候要切換這兩個 """
    datasets = { "test": DataLoader(iclevrDataset("iclevr", "test"), batch_size=1, worker_init_fn=seed_worker, generator=g)}
    # datasets = { "new_test": DataLoader(iclevrDataset("iclevr", "new_test"), batch_size=1, worker_init_fn=seed_worker, generator=g)}
            
    
    for name, loader in datasets.items():
        acc = generate_images(loader, noise_scheduler, model, eval_model, name)
        print(f"{name} accuracy: {np.mean(acc):.4f}")
            
    noise_scheduler = get_scheduler(timesteps)
    noise_scheduler.timesteps = noise_scheduler.timesteps.to(device)
    