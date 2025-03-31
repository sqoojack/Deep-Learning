
""" This file probably handles model evaluation. It includes functions to assess
    model performance on validation.   """

# test階段
# Evaluation
import argparse
import torch
from utils import dice_score
from models.unet import UNet
from models.resnet34_unet import ResNet_UNet
from oxford_pet import load_dataset

def evaluate(net, data_loader, device):
    net.eval()
    total_loss = 0
    total_dice_score = 0
    num_batches = len(data_loader)

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in data_loader:
            images = data['image'].to(device, dtype=torch.float32)
            masks = data['mask'].to(device, dtype=torch.float32)
            
            # 將masks從[batch_size, 1, height, width] 變為[batch_size, height, width]
            masks = masks.squeeze(1).long()  # 確保masks是LongTensor
            
            outputs = net(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            dice = dice_score(outputs, masks)
            total_dice_score += dice

    avg_loss = total_loss / num_batches
    avg_dice_score = total_dice_score / num_batches

    return avg_loss, avg_dice_score

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # parser.add_argument('--data_path', type=str, default='./dataset/oxford-iiit-pet', help='path of the input data')
    parser.add_argument('--data_path', type=str, default='./dataset', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.00012, help='learning rate')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()  # 呼叫get_args()以獲取命令行參數

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    test_loader = load_dataset(args.data_path, mode='test', batch_size=args.batch_size, shuffle=False)  # 修正mode和shuffle
    
    # 初始化模型結構
    
    # model = UNet(n_channels=3, n_classes=2).to(device)
    model = ResNet_UNet(num_classes=2).to(device)
    
    # 載入訓練好的權重
    # model_weights = torch.load('saved_models/ResNet_UNet_best_model_90.pth', weights_only=True)
    # model_weights = torch.load('saved_models/UNet_best_model_90.pth', weights_only=True)
    
    # model_weights = torch.load('saved_models/DL_Lab3_UNet_313552049_鄭博涵.pth', weights_only=True)
    model_weights = torch.load('saved_models/DL_Lab3_ ResNet34_UNet_313552049_鄭博涵.pth', weights_only=True)
    
    model.load_state_dict(model_weights)

    avg_loss, avg_dice_score = evaluate(model, test_loader, device)
    
    print(f"Evaluation Loss: {avg_loss}, Evaluation Dice score: {avg_dice_score}")
    
    
    