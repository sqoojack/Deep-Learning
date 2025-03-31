

import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler

from dataset import iclevrDataset
from model.DDPM import ConditionalDDPM

class DDPMTrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.total_epoch = args.num_epochs
        self.learning_rate = args.lr
        self.model = ConditionalDDPM().to(self.device)
        self.total_timesteps = args.total_timesteps
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.total_timesteps, beta_schedule=args.beta_schedule)
        self.loss_function = nn.MSELoss()
        
        self.optimizer = self.choose_optim()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.8)
        
        self.batch_size = args.batch_size
        self.train_loader = DataLoader(iclevrDataset(args.dataset, mode='train'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.fast_train_loader = DataLoader(iclevrDataset(args.dataset, mode='train', partial=0.3), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        self.writer = SummaryWriter(os.path.join(args.log_dir, 'iclevr_ddpm'))
        
        if args.resume:
            self.load_checkpoint(args.checkpoint)
            print(f"Resumed training from checkpoint: {args.checkpoint}")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate  # 從自己指定的learning_rate繼續train

    def get_random_timesteps(self, batch_size):
        # 去噪過程是按照每一步timestep去做的, 在這邊會隨機選擇時間步, 從而模擬不同的噪聲
        return torch.randint(0, self.total_timesteps, (batch_size,)).long().to(self.device)   # 生成的隨機整數範圍是 [0, total_timesteps-1], 輸出張量形狀:(batch_size, )
    
    """ 保存model跟optim的檢查點 """
    def save_checkpoint(self, epoch):
            save_dir = os.path.join(self.args.save_dir, f"checkpoint_{epoch}.pth")
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': epoch
            }, save_dir)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint.get('epoch', 20)
    
    def train_one_epoch(self, epoch, train_loader):
        self.model.train()
        train_loss = []
        progress_bar = tqdm(train_loader, desc=f"Epoch: {epoch}", leave=False, ncols=90)  #  leave=False: 進度條完成後不會保留在終端上
        
        for x, label in progress_bar:
            x, label = x.to(self.device), label.to(self.device)
            batch_size = x.shape[0]
            noise = torch.randn_like(x)     # 生成與x形狀相同的隨機高斯噪聲, 並且是均值為0, std=1的normal distribution
            
            timesteps = self.get_random_timesteps(batch_size)
            timesteps = timesteps.squeeze() # 確保 timesteps 是 1 維數組
            
            noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)    # 經過noise後的圖像
            output = self.model(noisy_x, timesteps, label)
            
            loss = self.loss_function(output, noise)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss.append(loss.item())
            progress_bar.set_postfix({'Loss': np.mean(train_loss)})
        self.scheduler.step()
        return np.mean(train_loss)   

    """ 設置fast_train"""
    def fast_train(self):
        for epoch in range(1, self.args.num_fast_epochs + 1):
            train_loss = self.train_one_epoch(epoch, self.fast_train_loader)
            current_lr = self.scheduler.get_last_lr()[0]  # 獲取調度器最近設置的學習率
            print(f"Epoch {epoch}/{self.total_epoch}  |  Train Loss: {train_loss:.4f}  |  lr: {current_lr:.7f}")
    
    def train(self):
        start_epoch = 21
        if self.args.resume:
            checkpoint = torch.load(self.args.checkpoint, weights_only=True)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Resuming from epoch {start_epoch}")
            
            # 更新學習率並重置調度器
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.8)
        
        self.fast_train()
        
        for epoch in range(start_epoch, self.args.num_epochs + 1):  # 從訓練到一半的model繼續train
        # for epoch in range(self.args.num_fast_epochs + 1, self.args.num_epochs + 1):
            train_loss = self.train_one_epoch(epoch, self.train_loader)
            
            
            current_lr = self.optimizer.param_groups[0]['lr']  # 獲取調度器最近設置的學習率
            print(f"Epoch {epoch}/{self.args.num_epochs}  |  Train Loss: {train_loss:.4f}  |  lr: {current_lr:.7f}")
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(epoch)
        
        self.save_checkpoint(self.args.num_epochs)
    
    
    def choose_optim(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.8, weight_decay=1e-4)     # bad
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # bad
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.99)   # 對於噪聲梯度的問題 處理佳 bad
        # optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.learning_rate) # bad
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))    # good
        # optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0)  # bad
        # optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)  # good
        # optimizer = torch.optim.Adadelta(self.model.parameters(), rho=0.9, eps=1e-6, weight_decay=1e-4)   # good
        return optimizer
    
               
        
def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0007)     # 0.0006 is good
    # parser.add_argument('--lr', type=float, default=0.000001)    # 在載入checkpoint 之後 train的學習率
    
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--total_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='squaredcos_cap_v2')
    parser.add_argument('--batch_size', type=int, default=25)   # 25 is good
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--num-fast-epochs', type=int, default=5, help='Number of fast train epochs')
    
    
    parser.add_argument('--dataset', type=str, default='iclevr')    # dataset存在哪裡
    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--resume', action='store_true')
    # parser.add_argument('--resume', action='store_true', default=True)  # 將默認值設為 True
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_23(best).pth')  # 更新檢查點路徑
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_freq', type=int, default=1)
    return parser.parse_args()

def main():
    args = arg_parser()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    trainer = DDPMTrainer(args)
    trainer.train()
    
if __name__ == '__main__':
    main()
    
    
