
# this file is to process dataset

import os
import torch
import shutil   # 用於文件操作
import numpy as np  

from PIL import Image   # 用於圖像處理
from tqdm import tqdm   # 顯示進度條
from urllib.request import urlretrieve  # 用來下載文件的函數
from torch.utils.data import DataLoader
from torchvision import transforms

# torch.utils.data.Dataset: 在torch.utils.data裡的Dataset, 要定義如何訪問dataset的方法
class OxfordPetDataset(torch.utils.data.Dataset):   # Oxford: 英國牛津大學
    def __init__(self, root, mode="train", transform=None): # root: 表示根目錄

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")   # 根目錄 -> images資料夾
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")    # annotations裡面的trimaps資料夾

        self.filenames = self._read_split()  # 根據self.mode值 來讀取不同的文件名列表(train, valid, test)

    def __len__(self):
        return len(self.filenames)  # self.filenames是一個list -> len(self.filenames)就是這個數據中的樣本總數

    # 取得數據與對應的標籤
    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg") # 開始一張一張讀入
        mask_path = os.path.join(self.masks_directory, filename + ".png")   # mask檔名皆為.png

        image = np.array(Image.open(image_path).convert("RGB")) # 打開圖像文件並轉為RGB模式, 再將圖像數據轉成 NumPy, 以便於數值數值操作
        trimap = np.array(Image.open(mask_path))  # 因為通常是灰度圖像 -> 不需要顏色信息
        mask = self._preprocess_mask(trimap)    # 調用私有方法_preprocess_mask 來對trimap做預處理

        sample = dict(image=image, mask=mask, trimap=trimap)
        # transform: 用來做數據增強或處理的東西
        if self.transform is not None:  # 如果transform不為空 -> 執行transform操作 
            sample = self.transform(sample)   # ** 將字典解包為關鍵字參數 -> 可將字典中的key value作為獨立的參數傳遞給function
            
        return sample

    @staticmethod   # 使其不用使用 self
    def _preprocess_mask(mask): # 做preprocess (只有0跟1 -> 方便計算dice score)
        mask = mask.astype(np.float32)  # 將mask數據轉換為 float32類型
        mask[mask == 2.0] = 0.0     # 將所有像素值為2的 轉換成0 (像素值2 表示不確定區域 -> 轉換為背景)
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0   # 像素值為 1 or 3的 轉換為1 (前景標籤)
        return mask

    def _read_split(self):  # 將數據做分割
        # test資料集在test.txt裡面, 只需將trainval.txt分割成trainset and valset即可
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"  
        split_filepath = os.path.join(self.root, "annotations", split_filename) # 創建資料路徑
        with open(split_filepath) as f:    # f = open(split_filepath)
            split_data = f.read().strip("\n").split("\n")
            
        filenames = [x.split(" ")[0] for x in split_data]   # 提取split_data中的第一部分 形成list (每行的分隔是 "空格")
        
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):
        # load images
        filepath = os.path.join(root, "images.tar.gz")  # 可發現這邊不是用self.root, 是用root
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,  # 下載後保存到 filepath裡
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

# SimpleOxfordPetDataset 是 OxfordPetDataset 的子類
class SimpleOxfordPetDataset(OxfordPetDataset): # 從父類(OxfordPetDataset)中繼承所有特性 
    def __getitem__(self, *args, **kwargs):
        # *args, **kwargs : 用來傳遞任意數量的位置參數和關鍵字參數
        
        sample = super().__getitem__(*args, **kwargs)   # 調用了父類 OxfordPetDataset 的 __getitem__ 方法來獲取樣本

        # resize images (在父類的基礎上去微調, 做進一步處理)
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))   # bilinear: 雙線性插值, 用於圖像的平滑施放
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))  # nearest: 最近鄰插值, 用於縮放標籤圖像
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))
        # print(image.shape)

        # channel: 例如RGB 有3個通道, 每個通道都是一個灰度圖像, 表示該特定顏色的強度, ex: (255, 0, 0)表示純紅色, (0, 255, 0)表示純綠色
        # convert to other format HWC -> CHW (pytorch期望的數據)    (H, W, C): Height, width, channel
        """ image: 原本的圖像數據, 是np.array. -1: 表示原本的軸位置在最後一個軸(即channel), 0: 要將原始軸移動到的新位置 -> 0表示第一個軸"""
        sample["image"] = np.moveaxis(image, -1, 0) # HW 依照原本的相對位置 往後挪
        sample["mask"] = np.expand_dims(mask, 0)    # 增加通道維度(並加到第一個軸), (H, W) -> (1, H, W)
        sample["trimap"] = np.expand_dims(trimap, 0)    
        # print(image.shape)
        # print(mask.shape)

        return sample


class TqdmUpTo(tqdm):   # 進度條
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    # os.path.abspath(filepath): 將相對路徑轉成絕對路徑
    directory = os.path.dirname(os.path.abspath(filepath))  # dirname: directory name 獲取給定路徑的目錄
    os.makedirs(directory, exist_ok=True)   # 創建目錄
    if os.path.exists(filepath):    # if文件已存在, 則直接返回(只下載一次)
        return

    with TqdmUpTo(
        unit="B",   # 單位為byte
        unit_scale=True,    # 啟用單位縮放, 可將Byte 縮放到 KB, GB
        unit_divisor=1024,
        miniters=1, # 進度條更新的最小間隔
        desc=os.path.basename(filepath),    # 進度條前面的描述文本
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)  # 用於從網路上下載文件, reportbook: 用來顯示下載進度的函數, data: 要發送到服務器的數據
        t.total = t.n


def extract_archive(filepath):  # 用於解壓縮
    extract_dir = os.path.dirname(os.path.abspath(filepath)) # extract_dir: 獲取解壓的目錄, 這邊把他轉成絕對路徑
    dst_dir = os.path.splitext(filepath)[0] # 目標的解壓縮目錄
    
    if not os.path.exists(dst_dir): # if目標目錄不存在，則進行解壓縮操作
        shutil.unpack_archive(filepath, extract_dir)   # 執行解壓

"""預處理: 對圖像和標籤進行隨機水平或垂直翻轉"""
def random_flip(sample):
    image = sample['image']
    mask = sample['mask']
    trimap = sample['trimap']

    if np.random.rand() > 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        trimap = np.fliplr(trimap).copy()
    if np.random.rand() > 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
        trimap = np.flipud(trimap).copy()

    return {'image': image, 'mask': mask, 'trimap': trimap}

def load_dataset(data_path, mode, batch_size=8, shuffle = True, num_workers=0): # num_workers: 加載數據時使用的子進程數量
    dataset = SimpleOxfordPetDataset(root=data_path, mode=mode, transform=random_flip)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

