
""" This file contains code for training the neural network models. It includes functions
    related to model training, optimization, and backpropagation.   """
    
""" HDD: 跟DSC一樣是評估結果的指標(DSC: dice score) 
    ResNet_UNet 優於 HR-Net和SegNet, """    
import argparse
from models.unet import UNet
from models.resnet34_unet import ResNet_UNet
import torch
import torch.optim as optim
import torch.nn as nn
from oxford_pet import load_dataset
import matplotlib.pyplot as plt
from utils import dice_score

def train(args):  # 從 get_args() 函數中獲得的命令行參數 有data_path, epochs, batch_size, learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = ResNet_UNet(2).to(device)
    # model = UNet(n_channels=3, n_classes=2).to(device)

    # set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.88)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = load_dataset(args.data_path, mode='train', batch_size=args.batch_size, shuffle = 'True')
    val_loader = load_dataset(args.data_path, mode='valid', batch_size=args.batch_size, shuffle = 'False')
    
    best_dice_score = 0.0   # 初始化最佳dice_score
    train_losses = []   # 用於儲存每個epoch的train_loss
    val_losses = []
    val_dice_scores = []
    
    model.train()   # set to train mode
    for epoch in range(args.epochs):
        running_loss = 0.0
        total_samples = 0
        for i, data in enumerate(train_loader):
            images = data['image'].to(device, dtype=torch.float32)  # data['image']: 包含了批次中的所有image 數據
            # if images.dim() == 3:  # 如果圖像是3維的 (channels, height, width)
            #     images = images.unsqueeze(0)  # 增加批次維度，使其變為 (1, channels, height, width)
            masks = data['mask'].to(device, dtype=torch.float32)    # data['mask'] 包含了批次中對應的mask 數據
            
            if i % 50 == 0:
                print(f"Current Batch Number: {i+1}")   # 顯示 (batch_size, height, width)

            optimizer.zero_grad()   # 梯度重置
            
            outputs = model(images) # 輸入是images
            masks = masks.squeeze(1).long()  # 去掉第2維度，即將形狀從 (N, 1, H, W) 變為 (N, H, W)
            loss = criterion(outputs, masks)    # masks 為Ground Truth
            
            loss.backward()
            optimizer.step()   # 更新模型的參數    
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size    # .item(): 將tensor改成float (loss 是tensor)
            total_samples += batch_size
            
        avg_epoch_loss = running_loss / total_samples
        train_losses.append(avg_epoch_loss)
    
        # 在每個 epoch 結束後進行驗證
        val_loss, avg_dice_score = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_dice_scores.append(avg_dice_score)
        
        print(f"Epoch: {epoch+1}, Train Loss: {avg_epoch_loss}, Validation Loss: {val_loss:.4f}, Dice Score: {avg_dice_score:.4f}")

        # 保存最好的dice_score model
        if avg_dice_score > best_dice_score:
            best_dice_score = avg_dice_score
            torch.save(model.state_dict(), 'saved_models/ResNet_UNet_best_model.pth')
            # torch.save(model.state_dict(), 'saved_models/UNet_best_model.pth')
            print(f"Model saved with Dice Score: {best_dice_score:.4f}")
            
    # 繪製損失曲線
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.savefig('ResNet_loss_curve.png')
    # plt.savefig('UNet_loss_curve.png')
    plt.show()
        
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    total_dice_score = 0.0
    with torch.no_grad():   # 在驗證時不需算梯度
        for data in val_loader: # 這裡不需要用索引i -> 不用enumetate()
            images = data["image"].to(device, dtype = torch.float32)
            masks = data["mask"].to(device, dtype = torch.float32)
            
            outputs = model(images)
            masks = masks.squeeze(1).long()  # 去掉第2維度，即將形狀從 (N, 1, H, W) 變為 (N, H, W)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)    # loss.item(): 該batch中的平均損失 image.size(0): images張量的第一個維度大小 -> 即batch size
            
            dice = dice_score(outputs, masks)
            total_dice_score += dice * images.size(0)
    val_loss /= len(val_loader.dataset)
    avg_dice_score = total_dice_score / len(val_loader.dataset) # 因為驗證集通常不進行梯度更新, 所以可以直接用 len()
    
    return val_loss, avg_dice_score



def get_args(): # args: arguments
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')   # 初始化一個 argparse.ArgumentParser 物件，用於處理命令行參數
    parser.add_argument('--data_path', type=str, default = './dataset', help='path of the input data') # 指定輸入數據的路徑
    parser.add_argument('--epochs', '-e', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=20, help='batch size')   # 設置批量大小(總共是3311)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.00012, help='learning rate')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)