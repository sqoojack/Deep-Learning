import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils, transforms
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#TODO2 step1-4: design the transformer training strategy
class TrainTransfromer:
    def __init__(self, args, MaskGit_configs):
        self.args = args
        self.device = args.device 
        self.learning_rate = args.learning_rate
        self.model = VQGANTransformer(MaskGit_configs["model_param"]).to(self.device)   # 在設置模型時會用.to(device), 來確保model跟數據是在同一設備上
        self.optimizer, self.scheduler = self.configures_optimizers()   # configures_optimizers() 下面會定義
        self.prepare_training()
        self.writer = SummaryWriter("transformer_checkpoints/logs/")
        
    @staticmethod   # 靜態方法與實例無關
    def prepare_training(): # exist_ok: 當目錄存在時, 不會拋出錯誤
        os.makedirs("transformer_checkpoints", exist_ok=True)   # 創建 transformer_checkpoints 目錄, 裡面放checkpoint   

    def train_one_epoch(self, train_loader, epoch): # 在一個epoch裡所要做的train動作
        self.model.train()
        losses = []
        
        for x in tqdm(train_loader, ncols=90, leave=False, desc=f"Train Epoch {epoch}"):    # desc: 用來設定進度條前顯示的文字 -> 知道目前在哪個epoch
            x = x.to(self.device)
            self.optimizer.zero_grad()
            ratio = np.random.rand()
            y_pred, y = self.model(x, ratio)
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)   # 梯度裁減
            self.optimizer.step()
            
            losses.append(loss.detach().item())
        self.scheduler.step()
        return np.mean(losses)
    
    # 驗證階段
    def eval_one_epoch(self, val_loader, epoch):
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for x in tqdm(val_loader, ncols=90, leave=False, desc=f"Val Epoch {epoch}"):    # leave=False: 進度條完成後不會保留在終端上
                x = x.to(self.device)
                ratio = np.random.rand()
                y_pred, y = self.model(x, ratio)
                loss = F.cross_entropy(y_pred, y)
                losses.append(loss.detach().item())
        return np.mean(losses)
        
        
    def configures_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.8, weight_decay=1e-4)     # loss: 0.9 
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # (bad)
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.99)   # 對於噪聲梯度的問題 處理佳 (bad)
        # optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.learning_rate) # (bad)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))
        # optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0) # good
        # optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01) # not bad
        optimizer = torch.optim.Adadelta(self.model.parameters(), rho=0.9, eps=1e-6, weight_decay=1e-4)    # very good
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1)
        return optimizer, scheduler
    
    """ use fast_train """
    def fast_train(self, train_loader, val_loader, num_fast_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(1, num_fast_epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.eval_one_epoch(val_loader, epoch)
            
            # 在這裡使用 detach()
            train_loss = torch.tensor(train_loss).detach()
            val_loss = torch.tensor(val_loss).detach()
            
            self.writer.add_scalar('FastTrain/Loss/train', train_loss.item(), epoch)   # 記錄到 TensorBoard
            self.writer.add_scalar('FastTrain/Loss/val', val_loss.item(), epoch)
            self.writer.add_scalar('FastTrain/LR', self.scheduler.get_last_lr()[0], epoch)
            
            print(f"Fast Train Epoch: {epoch} |  Train Loss: {train_loss:.4f}   | Val Loss: {val_loss:.4f}  | lr: {train_transformer.scheduler.get_last_lr()[0]:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/fast_ckpt_{epoch}.pt")   # 保存model
            
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2: check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./transformer_checkpoints/fast_train_model.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for training.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
    parser.add_argument('--batch-size', type=int, default=14, help='Batch size.')
    parser.add_argument('--partial', type=float, default=1.0, help='Partial dataset to use.')
    
    # you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save checkpoint every ** epochs.')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--num-fast-epochs', type=int, default=2, help='Number of fast train epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--warmup-steps', type=int, default=1, help='Warmup steps.')
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Path to MaskGIT configuration file.')

    args = parser.parse_args()
    
    #   加載模型配置文件
    MaskGit_configs = yaml.safe_load(open(args.MaskGitConfig, 'r')) # args.MaskGitConfig: 指向配置文件的路徑
    train_transformer = TrainTransfromer(args, MaskGit_configs) # 創建一個TrainTransformer的實例
    
    train_loader = DataLoader(LoadTrainData(root=args.train_d_path, partial=args.partial),  # LoadTrainData: 從utils引入
                              batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,     # drop_last: 丟棄最後不完整的batch
                              pin_memory=True, shuffle=True)    # pin_memory: 在數據加載時, 將數據加載到內存的pinned memory -> 可以更快地從內存傳到memory上 
    
    val_loader = DataLoader(LoadTrainData(root=args.val_d_path, partial=args.partial),  # LoadTrainData: 從utils引入
                            batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
                            pin_memory=True, shuffle=False)
    
    # writer = SummaryWriter("transformer_checkpoints/logs/")     # 初始化SummaryWriter, 並將日記保存到路徑中
    
    fast_train_loader = DataLoader(LoadTrainData(root=args.train_d_path, partial=0.3),  # 使用部分數據
                              batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
                              pin_memory=True, shuffle=True)
    
    fast_val_loader = DataLoader(LoadTrainData(root=args.val_d_path, partial=0.3),  # 使用部分數據
                            batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
                            pin_memory=True, shuffle=False)
    
    # set fast_train mode
    train_transformer.fast_train(fast_train_loader, fast_val_loader, num_fast_epochs=args.num_fast_epochs)
    
    #TODO2 step1-5:
    for epoch in range(args.num_fast_epochs + 1, args.epochs + 1):
        best_val_loss = float('inf')
        train_loss = train_transformer.train_one_epoch(train_loader, epoch)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch)
        
        train_transformer.writer.add_scalar('Loss/train', train_loss, epoch)
        train_transformer.writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch}/{args.epochs}  |  Train Loss: {train_loss:.4f}   | Val Loss: {val_loss:.4f}  |  lr: {train_transformer.scheduler.get_last_lr()[0]:.4f}")
        
        if val_loss < best_val_loss : 
            best_val_loss = val_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/best_ckpt_{epoch}.pt")