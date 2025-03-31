
""" This file deals with model inference (making predictions) on unseen data. It
    includes functions to apply the trained model to new images.    """
# 用來將訓練好的模型生成盡量跟unseen data相似的圖？
""" 輸入command指令: python src/inference.py --model saved_models/UNet_best_model_90.pth --data_path
    ./dataset --output_path '.' --batch_size 16 --device cpu    """
    
import numpy as np
import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
from models.unet import UNet
from models.resnet34_unet import ResNet_UNet
from oxford_pet import load_dataset

def get_args():
    parser = argparse.ArgumentParser(description='Model inference on new images')
    parser.add_argument('--model', type=str, default='saved_models/UNet_best_model_90.pth', help='Path to the trained model file')
    # parser.add_argument('--model', type=str, default='saved_models/ResNet_UNet_best_model_90.pth', help='Path to the trained model file')
    parser.add_argument('--data_path', type=str, default='./dataset', help='Path to the input data folder')
    parser.add_argument('--output_path', type=str, default='.', help='Path to save the output results')   # 預設存在當前目錄
    parser.add_argument('--batch_size', '-b', type=int, default=10, help='Batch size for inference')
    parser.add_argument('--device', default='cuda', help='Device to use for inference (cuda or cpu)')
    return parser.parse_args()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension

def generate_output_image(preds, image_size):
    output_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # 假設preds是0或1的二元掩膜，0代表背景，1代表前景
    # 可以根據需要設置不同的顏色
    background_color = [0, 0, 0]  # 黑色
    foreground_color = [255, 255, 255]  # 白色
    
    output_image[preds == 0] = background_color
    output_image[preds == 1] = foreground_color
    
    return output_image

def inference(model, image_tensor, device):
    image_tensor = image_tensor.to(device)  # 將圖像張量移動到指定的設備（CPU 或 GPU）
    with torch.no_grad():  # 禁用梯度計算（在推理階段不需要計算梯度）
        outputs = model(image_tensor)  # 將圖像張量輸入模型，獲得輸出
        preds = torch.argmax(outputs, dim=1)  # 在通道維度上選取概率最大的類別作為預測結果
    return preds.cpu().numpy()  # 將預測結果移動到 CPU 並轉換為 NumPy 陣列

def save_output(output_image, output_path, image_name):
    output_image = Image.fromarray(output_image)
    output_image.save(os.path.join(output_path, image_name))

if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 使用命令列引數指定的模型
    model = UNet(n_channels=3, n_classes=2).to(device)
    # model = ResNet_UNet(num_classes=2).to(device)
    
    # 載入訓練好的權重，並確保載入到正確的設備
    model_weights = torch.load(args.model, map_location=device)
    model.load_state_dict(model_weights)

    # 使用 load_dataset 載入數據集
    dataset = load_dataset(args.data_path, mode='test', batch_size=args.batch_size, shuffle=False)
    
    max_batches = 2  # 設置要處理的批次數量
    
    for i, batch in enumerate(dataset):
        if i >= max_batches:
            break  # 停止處理更多批次
        images = batch['image'].to(device, dtype=torch.float32)  # 確保圖像是 float32 類型
        outputs = inference(model, images, device)
        
        for j in range(outputs.shape[0]):
            output_image = generate_output_image(outputs[j], images[j].shape[-2:])
            save_output(output_image, args.output_path, f"output_{i}_{j}.png")

    print("Inference complete. Results saved to:", args.output_path)