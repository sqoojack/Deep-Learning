
""" Reference: https://github.com/hank891008/Deep-Learning """

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random

""" 定義圖像轉換函數 """
def transform_img(img):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),    # 調整圖像到 64x64
        transforms.ToTensor(),  # 轉換為Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化圖像 (0.5, 0.5, 0.5): 對RGB三個通道的均值 以及 標準差
    ])
    return transform(img)

class iclevrDataset(Dataset):
    def __init__(self, root=None, mode='train', partial=1.0):
        super().__init__()
        assert mode in ['train', 'test', 'new_test'], "mode should be 'train', 'test', 'new_test'"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))    # 獲取當前腳本所在的目錄
        
        self.json_data = self.load_json(os.path.join(current_dir, f"{mode}.json"))  # 讀取對應模式的json文件, 下面有_load_json()
        self.objects_dict = self.load_json(os.path.join(current_dir, 'objects.json'))
        
        """ 加入partial之後改動的部分   """
        if mode == 'train':    
            all_items = list(self.json_data.items())
            num_items = int(len(all_items) * partial)
            selected_items = random.sample(all_items, num_items)
            self.json_data = dict(selected_items)
        
        self.labels = list(self.json_data.values()) if mode == 'train' else self.json_data  # 處理標籤數據
        self.labels_one_hot = self.create_one_hot_labels()
        
        if mode == 'train':
            self.img_paths = list(self.json_data.keys())
        self.root = root
        self.mode = mode
        
    """ 用來讀取json文件 """
    def load_json(self, filename):
        with open(filename, 'r') as json_file:
            return json.load(json_file)
        
    """ 創造one-hot 標籤矩陣的函數 """
    def create_one_hot_labels(self):
        labels_one_hot = torch.zeros(len(self.labels), len(self.objects_dict))
        for i, label in enumerate(self.labels):
            labels_one_hot[i][[self.objects_dict[obj] for obj in label]] = 1
        return labels_one_hot
    
    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = os.path.join(self.root, self.img_paths[index])
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform_img(img)
            except Exception as e:  # 處理另外狀況
                print(f"Error loading image at {img_path}: {e}")
                return None, None
            label_one_hot = self.labels_one_hot[index]
            return img, label_one_hot
        else:
            return self.labels_one_hot[index]
        
if __name__ == '__main__':
    dataset = iclevrDataset(root='iclevr', mode='train', partial=1.0)
    print(f"訓練集大小: {len(dataset)}")
    
    item = dataset[0]
    if item[0] is not None and item[1] is not None:
        x, y = item
        print(f"圖像尺寸: {x.shape},  標籤尺寸: {y.shape}")
    else:
        print("無法載入第一個項目，請檢查數據集路徑和圖像文件。")
    
    test_dataset = iclevrDataset(root='iclevr', mode='test')
    y = test_dataset[0]
    print(f"測試集標籤尺寸: {y.shape}")