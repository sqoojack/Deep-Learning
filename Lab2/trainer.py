""" Implement the code for training the SCCNet model, including functions 
    related to training, losses, optimizer, backpropagation, etc, remember to save 
    the model weight."""

import torch
import torch.nn as nn
import torch.optim as optim
# from model.SCCNet import model
from torch.utils.data import DataLoader
from tester import test_model, load_model
from Dataloader import MIBCI2aDataset
from torch.optim.lr_scheduler import StepLR
from utils import PlotLoss

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device, save_path, model_save_path):
    best_val_loss = float('inf')  # 初始化最佳驗證損失
    train_losses = []  # 儲存每個 epoch 的訓練損失
    val_losses = []  # 儲存每個 epoch 的驗證損失
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        # 打印訓練損失
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # 加入到train list中

        # 計算驗證損失
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)  # 加入驗證損失
        
        print(f"Epoch {epoch + 1},Training Loss: {avg_train_loss},  Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy:.2f}%")
        # 保存驗證損失最低時的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, model_save_path)
        
    # 獲取final test accuracy
    final_accuracy = test_model(model, test_loader, device)
    print(f"FT's Final accuracy: {final_accuracy:.2f}%")
    PlotLoss(train_losses, val_losses, save_path, final_accuracy)
        

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader)
    accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, accuracy

def save_model(model, path):
    torch.save(model.state_dict(), path)

num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "./model/SCCNet_model.pth"
save_path = "./result_plot.jpg"

model = load_model(model_save_path)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.88, weight_decay=1e-4)
# StepLR每隔10個epoch將學習率減半
scheduler = StepLR(optimizer, step_size=4, gamma=0.97)

train_dataset = MIBCI2aDataset(mode='FT_train')
val_dataset = MIBCI2aDataset(mode='FT_test')
test_dataset = MIBCI2aDataset(mode='FT_test')

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)



epochs = 200

train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device, save_path, model_save_path)

