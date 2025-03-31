
""" Annealing: 退火, 通過加熱和逐漸冷卻材料來減少其缺陷, 在ML中, Annealing是一種動態調整參數的技術
    KL Divergence: 是近似後驗分佈和先驗分佈之間的差異     """

""" Loss = Reconstruction Loss + ß * KL Divergence
    所以ß增加時, model會更注重latent distribution 接近先驗分佈, ß減小時, model更注重Reconstruction Loss """

""" Reference: https://github.com/KJLdefeated/NYCU_DLP_2024 """

import os
import argparse     # used to arg
import numpy as np
import torch
import torch.nn as nn   # provide multiple neural network
from torchvision import transforms  # torchvision: 專門用來處理計算機視覺的任務, transform: 圖像變換, 對圖像進行預處理與數據增強
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import functional as F

import torch.optim as optim
from torchvision.utils import save_image  # save_image(img, "file_name.png") : 保存Tensor格式的圖像到文件中
from torch import stack     # stack: 將一組形狀相同的tensor堆疊成一個新的tensor, 跟torch.cat不同, torch.cat是在已有維度上進行連接
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio  # 讀入和寫入圖像或視頻文件  ex: img = imageio.imread("檔名")
import random   # 用來生成隨機數
from math import log10

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder
from dataloader import Dataset_Dance

from torch.utils.tensorboard import SummaryWriter

""" PSNR值越高代表 兩張圖片越相似"""
def generate_PSNR(imgs1, imgs2, data_range=1.):   # imgs1: real image, imgs2: fake image, data_range: 圖像數據的像素值範圍 (0~1)
    mse = nn.functional.mse_loss(imgs1, imgs2)
    
    """ log10只能用在純量, torch.log10可以用在tensor上 """
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)   # 20 * log10(data_range): 信號的峰值, 10 * torch.log10(mse): 噪聲的強度
    return psnr

def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 +  logvar - mu.pow(2) - logvar.exp())  # logvar.exp(): 自然指數e 的 logvar次方
    KLD /= batch_size
    return KLD
    
""" beta 會在多個週期內從 start 到 stop 線性增加, 然後在每個週期結束時重置
    這樣的退火策略有助於在模型訓練過程中平衡Reconstruct Loss and KL Divergence  """
class kl_annealing():
    def __init__(self, args, current_epoch = 0):
        # TODO
        self.args = args
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        
        self.current_epoch = current_epoch
        n_cycle, ratio = self.kl_anneal_cycle, self.kl_anneal_ratio
        
        if self.kl_anneal_type == "Cyclical":
            self.beta_list = self.frange_cycle_linear(n_iter=args.num_epoch, n_cycle=n_cycle, ratio=ratio)   # self.beta_list: 用來存放beta列表
            self.beta = torch.tensor(self.beta_list[0]) # 初始化是[0] (第一個元素), 之後update時, beta會慢慢改變
            
        elif self.kl_anneal_type == "Monotonic":
            self.beta_list = self.frange_cycle_linear(n_iter=args.num_epoch, n_cycle=n_cycle, ratio=ratio)
            self.beta = torch.tensor(self.beta_list[0])
            
        elif self.kl_anneal_type == "Full_KL":
            n_cycle, ratio = None, None
            self.beta_list = None
            self.beta = torch.tensor(1.0)
            
        else:   # self.kl_anneal_type == "None":
            n_cycle, ratio = None, None
            self.beta_list = None
            self.beta = torch.tensor(0.0)
            
    def update(self):
        # TODO
        if self.kl_anneal_type == "None" or self.kl_anneal_type == "Full_KL":
            pass    # do nothing
        else:   # for mode = Cyclical or Monotonic
            self.beta = torch.tensor(self.beta_list[self.current_epoch])
        self.current_epoch += 1

    def get_beta(self):
        # TODO
        return self.beta
    
    """ 參數解釋: 
        n_iter: 總epoch數
        start: 每個循環的起始值
        end: 每個循環的結束值
        n_cycle: 循環次數
        ratio: 每個循環中線性增長部分所佔的比例     """
    def frange_cycle_linear(self, n_iter, start=0.0, end=1.0, n_cycle=1, ratio=1):
        # TODO
        L = np.ones(n_iter)     # n_iter = num_epoch, 初始化一個長度為n_iter的數組, 值皆為1
        period = n_iter / n_cycle
        step = (end - start) / (period * ratio)
        
        for c in range(n_cycle):
            v, i = start, 0
            while i <= period * ratio and int(i + c * period) < n_iter:
                L[int(i + c * period)] = v
                v += step
                i += 1
        # print(L)
        return L
    
class VAE_model(nn.Module):
    def __init__(self, args):
        super(VAE_model, self).__init__()   # self: 用來具體化出一個物件(object)
        self.args = args
        """ 參數皆為in_channels, out_channels"""
        # define a transform image from RGB_domain and feature_domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)  # 3: in_channels
        self.label_transformation = Label_Encoder(3, args.L_dim)
            
        # self.dropout = nn.Dropout(0.1)  # 防止overfitting
        
        """ args.F_dim: dimension of feature human frame
            args.L_dim: dimension of feature label frame
            args.N_dim: dimension of noise 
            args.D_out_dim: output in Decoder_Fusion    """
        self.Gaussian_Predictor = Gaussian_Predictor(in_chans=args.F_dim + args.L_dim,  out_chans=args.N_dim)   # 預測出z, mu, logvar
        self.Decoder_Fusion = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)     # 輸入自然會是 Decoder_Fusion的output, maybe生成RGB圖像 -> output_nc=3
        
        
        """ 調整優化器  """
        if self.args.optim == 'SGD':
            self.optim = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.optim == 'Adamax':
            self.optim = torch.optim.Adamax(self.parameters(), lr=self.args.lr)
        elif self.args.optim == 'AdamW':
            self.optim = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=0.00001)
        else:
            self.optim = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.00001)
            
            
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[5, 10], gamma=0.6)   # 到第5個epoch時 lr乘以gamma, 第10個epoch時 再乘以gamma
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.92)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.7)
        
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        """ Teacher forcing arguments   """
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step   # Decay step that tfr adopted 
        self.tfr_sde = args.tfr_sde     # the epoch that tfr start to decay
        
        self.train_vi_len = args.train_vi_len   # training video length
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.log_save_path = f"logs/lr_{args.lr}_optim_{args.optim}_tfr_{args.tfr}_kl_{args.kl_anneal_type}_{args.kl_anneal_cycle}"
        self.writer = SummaryWriter(self.log_save_path)     # 用來可視化model的訓練過程
        print(self.log_save_path)  
        
    def forward(self, img, img_last, label, val=False): # val: 在驗證階段時 將z隨機化 -> z允許model從潛在空間中抽取不同的點 -> 增加多樣性
        img_features = self.frame_transformation(img)   # 轉成RGB影像並存入img_features
        img_last = self.frame_transformation(img_last)
        label_features = self.label_transformation(label)
        
        z, mu ,logvar = self.Gaussian_Predictor(img_features, label_features)
        
        if val == True:
            z = torch.randn(z.size()).to(self.args.device)    
        kl_loss =  kl_criterion(mu, logvar, self.batch_size) # 注意在forward這邊不需要用到 kl_annealing, 是在training_one_step中才會用到
        
        decoder_output = self.Decoder_Fusion(img_last, label_features, z)   # 將上一幀的特徵與當前幀的特徵合起來解碼 -> improve model's 時序關聯性
        pred = self.Generator(decoder_output)
        
        return pred, kl_loss
    
    def training_stage(self):
        psnrs = []
        self.train()
        ewma_psnr = 0
        for i in range(self.args.num_epoch):    # 總共的epoch數
            train_loader = self.train_dataloader()  # 下面會定義
            adapt_TeacherForcing = (random.random() < self.tfr)     # random.random(): 用來生成 [0.0, 1.0) 範圍內的浮點數
                
            train_loss = 0
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=150)):  # 進度條的寬度設為150字元
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                train_loss += loss
                beta = self.kl_annealing.get_beta()
                
                if adapt_TeacherForcing:    # loss.detach().cpu():  得到loss的副本, 該副本已從計算圖分離
                    self.tqdm_bar("train [TeacherForcing: ON, {:1f}], beta: {:.2f}".format(self.tfr, beta),     # 印出進度條
                                  pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])    # 取得最近的學習率
                else:
                    self.tqdm_bar("train [TeacherForcing: OFF, {:1f}], beta: {:.2f}".format(self.tfr, beta),
                                  pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            val_loss, psnr, _ = self.eval_val()    # 呼叫eval_val()函式
            
            if self.current_epoch % self.args.per_save == 0:    # 當 目前epoch到了 儲存epoch 的倍數時, 便會save model
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))    # 第一個參數是save到哪個路徑, ckpt: checkpoint
                
            if i != 0:
                ewma_psnr = 0.99 * ewma_psnr + 0.01 * psnr.numpy()
            
            self.writer.add_scalar('train loss', train_loss / len(train_loader), self.current_epoch)    # 到時在tensorboard上觀察
            self.writer.add_scalar('val loss', val_loss, self.current_epoch)
            self.writer.add_scalar('val PSNR', psnr, self.current_epoch)
            self.writer.add_scalar('beta', self.kl_annealing.get_beta(), self.current_epoch)
            self.writer.add_scalar('tfr', self.tfr, self.current_epoch)
            
            psnrs.append(psnr.numpy())
                
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
        return ewma_psnr
            
    @torch.no_grad()
    def eval_val(self):
        self.eval()
        val_loader = self.val_dataloader()  # 下面會定義                    
        
        for (img, label) in (pbar := tqdm(val_loader, ncols=150)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr, psnr_frame = self.val_one_step(img, label)  # 在test會用到psnr_frame
            
            self.tqdm_bar(f"val | PSNR: {psnr:.2f}", pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])         
        self.train()    # 調回train mode
        return loss.cpu().detach(), psnr.cpu().detach(), psnr_frame   # 先將tensor移到cpu, 再將tensor從計算圖中分離, 表示返回的tensor不會參與梯度計算
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        img_last = img[:, 0]   # 要傳入img_last的參數, 初始化為第一幀
        KL = 0
        MSE = 0
        self.optim.zero_grad()  # 將optim裡的所有參數的梯度歸零
        for i in range(self.train_vi_len - 1):
            img_in = img[:, i+1]
            label_in = label[:, i+1]
            if adapt_TeacherForcing:
                pred, kl_loss = self(img_in, img[:, i], label_in)  # 進行forward pass, 用真實的前一幀圖像來train
            else:
                pred, kl_loss = self(img_in, img_last, label_in)   # 沒有的話就是 跟預測出來的前一幀比
                
            KL += kl_loss
            MSE += self.mse_criterion(pred, img_in)
            
            img_last = pred.detach()
        """ 退火: 在training早期, model主要關注reconstruction loss, 到後期則會關注 KL散度 """
        loss = MSE + KL * self.kl_annealing.get_beta()  # 在這邊用到退火策略, get_beta會返回當前epoch的beta值 並進行退火
        loss.backward()
        self.optimizer_step()
        return loss.detach()
        
    def val_one_step(self, img, label):
        with torch.no_grad():
            img_last = img[:, 0]
            KL = 0
            MSE = 0
            PSNR = 0
            psnr_frame = []     # 在test時會用到
            for i in range(self.val_vi_len - 1):
                img_in = img[:, i+1]
                label_in = label[:, i+1]
                pred, kl_loss = self(img_in, img_last, label_in, val=True)
                KL += kl_loss
                MSE += self.mse_criterion(pred, img_in)
                psnr = generate_PSNR(pred, img_in)
                PSNR += psnr
                psnr_frame.append(psnr.detach().cpu().numpy())  # 將tensor 轉換為numpy陣列
                img_last = pred.detach()
            PSNR /= self.val_vi_len
            loss = MSE
        return loss.detach(), PSNR.detach(), psnr_frame
    
    """     在這邊不會用到make_gif()    """
    # def make_gif(self, images_list, img_name):  # 將一系列圖像組合成gif動畫, 並保存到指定路徑
    #     new_list = []
    #     for img in images_list:
    #         new_list.append(transforms.ToPILImage()(img))
            
    #     new_list[0].save(img_name, format="GIF", append_images=new_list,
    #                 save_all=True, duration=40, loop=0) # duration = 40: 每一幀之間的時間間隔為 40 毫秒
        
    def train_dataloader(self):
        """   進行數據增強 -> 以防止overfitting   """
        # augment_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5), # 隨機水平翻轉
        # transforms.RandomRotation((-5, 5)),  # 將圖像隨機旋轉一個在-5度到5度之間的角度
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 顏色抖動
        # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # 對圖像進行仿射變換  
        # ])
        
        transform = lambda x: self.process_image(x)
        
        """ DR: your dataset path, partial: 要train的dataset 佔的比例  (其中有分是不是 fast_train) """
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                partial=self.args.fast_partial if self.args.fast_train else self.args.partial)
        
        if self.current_epoch >= self.args.fast_train_epoch: # 如果目前epoch >= fast_train_epoch, 則關閉fast_train
            self.args.fast_train = False
            
        """ drop_last = True: If最後一個 batch 的數據數量少於設定的 batch_size, 則丟棄這個不完整的 batch"""
        train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.args.num_workers, drop_last=True, shuffle=True)
        return train_loader     # num_workers: 代表加載時使用多少個子進程來處理
    
    def val_dataloader(self):
        
        transform = lambda x: self.process_image(x)

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)
        val_loader = DataLoader(dataset, batch_size=1, num_workers=self.args.num_workers, drop_last=True, shuffle=False)  # 這邊batch_size固定為1, 不然psnr會發生計算錯誤
        return val_loader
    
    def teacher_forcing_ratio_update(self):  # 用在training_stage那邊
        # TODO
        if self.current_epoch >= self.tfr_sde:   # tfr_sde: The epoch that teacher forcing ratio start to decay
            self.tfr -= self.tfr_d_step     # tfr_d_step: tfr減少的比例
            self.tfr = max(0.0, self.tfr)    # 不能少於0
            
    def tqdm_bar(self, mode, pbar, loss, lr):   # pbar: 代表這個進度條的具體實例
        pbar.set_description(f"({mode}) Epoch: {self.current_epoch}, lr: {lr:.5f}", refresh=False) # 進度條的描述內容
        pbar.set_postfix_str(f"loss={loss:.4f}", refresh=False)   # 添加進度條後的額外訊息
        pbar.refresh()  #   刷新進度條   
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),    # 模型的狀態字典, 包含了model的所有parameter
            "optimizer": self.optim.state_dict(),
            "lr": self.scheduler.get_last_lr()[0],
            "tfr": self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")   # ckpt: check point
        
    def load_checkpoint(self):  # checkpoint檔案: 定期保存你model的狀態, if訓練過程被中斷 可以從最近一個checkpoint開始train
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint["state_dict"], strict=True)
            self.args.lr = checkpoint['lr']     # 記錄lr, tfr
            self.tfr = checkpoint['tfr']
            
            """ 調整優化器  """
            if self.args.optim == 'SGD':
                self.optim = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9)
            elif self.args.optim == 'Adamax':
                self.optim = torch.optim.Adamax(self.parameters(), lr=self.args.lr)
            elif self.args.optim == 'AdamW':
                self.optim = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=0.00001)
            else:
                self.optim = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.00001)
            
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint["last_epoch"])
            self.current_epoch = checkpoint["last_epoch"]
            
    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)     # 進行梯度裁剪
        self.optim.step()   # update optimizer
        
    def process_image(self, img):  # 處理 NumPy、PIL 圖像或文件路徑
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'))
        elif isinstance(img, str):  # 如果是文件路徑
            img = Image.open(img).convert('RGB')
        elif not isinstance(img, Image.Image):
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # 調整圖像大小
        img = F.resize(img, (self.args.frame_H, self.args.frame_W))
        
        # 手動轉換為張量
        img = np.array(img)
        img = img.transpose((2, 0, 1))  # 將 HWC 轉換為 CHW
        img = torch.Tensor(img).float() / 255.0
        return img

        
def main(args):
    model = VAE_model(args).to(args.device)
    model.load_checkpoint()
    
    if args.test:
        _, PSNR, psnr_frame = model.eval_val()
        np.save(f"PSNR_per_frame.npy", np.asarray(psnr_frame))
    else:
        PSNR = model.training_stage()
    return PSNR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size', type=int,    default=2)
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--device', type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim', type=str, choices=["Adam", "AdamW", "SGD", "Adamax"], default="Adamax")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--test', action='store_true') # 是否在測試階段, 當在command line中提供這個flag時, args.test將會設置為true
    parser.add_argument('--store_visualization', action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR', type=str, default="Dataset", help="Your Dataset Path")
    parser.add_argument('--save_root', type=str, default="ckpt", help="The path to save your data")
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_epoch', type=int, default=15, help="number of total epoch")
    parser.add_argument('--per_save', type=int, default=1, help="Save checkpoint every seted epoch")
    parser.add_argument('--partial', type=float, default=1.0, help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len', type=int, default=16, help="Training video length")
    parser.add_argument('--val_vi_len', type=int, default=630, help="valdation video length")
    parser.add_argument('--frame_H', type=int, default=32, help="Height input image to be resize")
    parser.add_argument('--frame_W', type=int, default=64,  help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim', type=int, default=128, help="Dimension of feature human frame")
    parser.add_argument('--L_dim', type=int, default=32, help="Dimension of feature label frame")
    parser.add_argument('--N_dim', type=int, default=12, help="Dimension of the Noise")
    parser.add_argument('--D_out_dim', type=int, default=192, help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr', type=float, default=0.75, help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde', type=int,   default=6, help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step', type=float, default=0.2, help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path', type=str, default=None, help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train', action='store_true')
    parser.add_argument('--fast_partial', type=float, default=0.3, help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch', type=int, default=9, help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type', type=str, default='Cyclical', help="")
    parser.add_argument('--kl_anneal_cycle', type=int, default=2, help="")
    parser.add_argument('--kl_anneal_ratio', type=float, default=1.25, help="")
    
    args = parser.parse_args()
    main(args)


""" 
    python3 Lab4_template/Trainer.py --DR "./LAB4_Dataset" --save_root "./saved_model/Cyclical" --fast_train --kl_anneal_type "Cyclical"
    python3 Lab4_template/Trainer.py --DR "./LAB4_Dataset" --save_root "./saved_model/Monotonic" --fast_train --kl_anneal_type "Monotonic"
    python3 Lab4_template/Trainer.py --DR "./LAB4_Dataset" --save_root "./saved_model/None" --fast_train --kl_anneal_type "None"
    python3 Lab4_template/Trainer.py --DR "./LAB4_Dataset" --save_root "./saved_model/Full_KL" --fast_train --kl_anneal_type "Full_KL"

    tensorboard --logdir_spec Cyclical:log/log_Cyclical,Monotonic:log/log_Monotonic,Full_KL:log/log_Full_KL,None:log/log_None 
    tensorboard --logdir=logs 
"""