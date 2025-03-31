import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder
from torchvision.utils import save_image
from torch import stack
from PIL import Image
from torchvision.transforms import functional as F

import imageio
from math import log10
from Trainer import VAE_model
import glob
import pandas as pd
import random


TA_ = """
 ██████╗ ██████╗ ███╗   ██╗ ██████╗ ██████╗  █████╗ ████████╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗    ██╗██╗██╗
██╔════╝██╔═══██╗████╗  ██║██╔════╝ ██╔══██╗██╔══██╗╚══██╔══╝██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝    ██║██║██║
██║     ██║   ██║██╔██╗ ██║██║  ███╗██████╔╝███████║   ██║   ██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗    ██║██║██║
██║     ██║   ██║██║╚██╗██║██║   ██║██╔══██╗██╔══██║   ██║   ██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║    ╚═╝╚═╝╚═╝
╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝██║  ██║██║  ██║   ██║   ╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║    ██╗██╗██╗
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝    ╚═╝╚═╝╚═╝                                                                                                                          
"""

def get_key(fp):
    filename = fp.split('/')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

from torch.utils.data import DataLoader
from glob import glob
from torch.utils.data import Dataset as torchData
from torchvision.datasets.folder import default_loader as imgloader

""" 使seed固定, 以便在demo時跟上傳kaggle的結果一樣  """
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Dataset_Dance(torchData):
    def __init__(self, root, transform, mode='test', video_len=7, partial=1.0):
        super().__init__()
        self.img_folder = []
        self.label_folder = []
        
        # data_num = len(glob('./Demo_Test/*'))
        data_num = 5
        for i in range(data_num):
            img_files = sorted(glob(os.path.join(root, f'test/test_img/{i}/*')), key=get_key)
            label_files = sorted(glob(os.path.join(root, f'test/test_label/{i}/*')), key=get_key)
            
            if img_files and label_files:
                self.img_folder.append(img_files)
                self.label_folder.append(label_files)
            else:
                print(f"Warning: 索引 {i} 未找到文件")     # 用來測試是否讀取到資料集
        
        if not self.img_folder or not self.label_folder:
            raise ValueError("在指定的目錄中未找到有效數據")
        
        self.transform = transform

    def __len__(self):
        return len(self.img_folder)

    def __getitem__(self, index):
        frame_seq = self.img_folder[index]
        label_seq = self.label_folder[index]
        
        imgs = []
        labels = []
        imgs.append(self.transform(imgloader(frame_seq[0])))
        for idx in range(len(label_seq)):
            labels.append(self.transform(imgloader(label_seq[idx])))
        return stack(imgs), stack(labels)


class Test_model(VAE_model):
    def __init__(self, args):
        super(VAE_model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
    """ 跟trainer那邊的forward差不多 """
    def forward(self, img, label): 
        # TODO
        img_features = self.frame_transformation(img)
        label_features = self.label_transformation(label)
        z, mu, logvar = self.Gaussian_Predictor(img_features, label_features)
        
        z = torch.randn(z.size()).to(self.args.device)  # 打亂z
        decoder_output = self.Decoder_Fusion(img_features, label_features, z)
        pred = self.Generator(decoder_output)
        return pred
            
            
    @torch.no_grad()
    def eval_val(self):
        self.eval()
        val_loader = self.val_dataloader()
        if len(val_loader) == 0:
            print("警告: val_loader 是空的!")
            return
        pred_seq_list = []
        for idx, (img, label) in enumerate(tqdm(val_loader, ncols=80)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            pred_seq = self.val_one_step(img, label, idx)
            if pred_seq is not None:
                pred_seq_list.append(pred_seq)
    
        if not pred_seq_list:
            print("警告: 沒有生成有效的預測!")
            return
        
        # submission.csv is the file you should submit to kaggle
        pred_to_int = (np.rint(torch.cat(pred_seq_list).numpy() * 255)).astype(int)
        df = pd.DataFrame(pred_to_int)
        df.insert(0, 'id', range(0, len(df)))
        df.to_csv(os.path.join(self.args.save_root, f'submission.csv'), header=True, index=False)
        
        
            
    
    def val_one_step(self, img, label, idx=0):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        assert label.shape[0] == 630, "Testing pose seqence should be 630"
        assert img.shape[0] == 1, "Testing video seqence should be 1"
        
        # decoded_frame_list is used to store the predicted frame seq
        # label_list is used to store the label seq
        # Both list will be used to make gif
        decoded_frame_list = [img[0].cpu()]
        label_list = []

        img_in = img[0]
        for i in range(629):
            pred_frame = self(img_in, label[i])
            img_in = pred_frame
            decoded_frame_list.append(pred_frame.cpu())
            label_list.append(label[i].cpu())
            
        # Please do not modify this part, it is used for visulization
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)
        
        assert generated_frame.shape == (1, 630, 3, 32, 64), f"The shape of output should be (1, 630, 3, 32, 64), but your output shape is {generated_frame.shape}"
        
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'pred_seq{idx}.gif'))
        
        # Reshape the generated frame to (630, 3 * 64 * 32)
        generated_frame = generated_frame.reshape(630, -1)
    
        if generated_frame.numel() == 0:
            print(f"警告: 索引 {idx} 的 generated_frame 是空的")
            return None
        
        return generated_frame
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=20, loop=0)
    
    def val_dataloader(self):
        transform = lambda x: self.process_image(x)
        
        
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, video_len=self.val_vi_len)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path, weights_only=True)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            
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
    set_seed(0)
    os.makedirs(args.save_root, exist_ok=True)
    model = Test_model(args).to(args.device)
    model.load_checkpoint()
    model.eval_val()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--no_sanity',     action='store_true')
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--make_gif',      action='store_true')
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    args = parser.parse_args()
    main(args)


"""
submission:
python3 Lab4_template/Tester.py --DR "./LAB4_Dataset" --ckpt_path "./saved_model/Full_KL/epoch=13_best.ckpt" --make_gif  --save_root "./Lab4_template/gif/Full_KL"
python3 Lab4_template/Tester.py --DR "./LAB4_Dataset" --ckpt_path "./saved_model/Full_KL/epoch=14_best.ckpt" --make_gif  --save_root "./Lab4_template/gif/Full_KL"

python3 Lab4_template/Tester.py --DR "./LAB4_Dataset" --ckpt_path "./saved_model/Cyclical/epoch=9_best.ckpt" --make_gif  --save_root "./Lab4_template/gif/Cyclical"
python3 Lab4_template/Tester.py --DR "./LAB4_Dataset" --ckpt_path "./saved_model/Monotonic/epoch=8_best.ckpt" --make_gif  --save_root "./Lab4_template/gif/Monotonic"
python3 Lab4_template/Tester.py --DR "./LAB4_Dataset" --ckpt_path "./saved_model/None/epoch=14_best.ckpt" --make_gif  --save_root "./Lab4_template/gif/None"

"""