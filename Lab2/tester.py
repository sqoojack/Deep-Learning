# Implement the code for testing, load the model, print out the accuracy for lab demo and report.

import torch
from model.SCCNet import SCCNet   # 從model資料夾引入SCCNet.py
from torch.utils.data import DataLoader
from Dataloader import MIBCI2aDataset

def load_model(model_path, num_classes = 4):
    # 加載已訓練的model
    model = SCCNet(numClasses = num_classes)
    model.load_state_dict(torch.load(model_path))   # 加載保存在路徑中的model parameter
    model.eval()    # set model to evaluate mode
    return model

def test_model(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():   # torch.no_grad: 表示不需計算梯度
        for data in test_loader:    # 逐批處理dataloader集 中的所有數據
            images, labels = data
            images, labels = images.to(device), labels.to(device)   # set to 指定的device
            outputs = model(images) 
            _, predicted = torch.max(outputs.data, 1)   # 1指的是在每一個輸出的向量中尋找最大值
            total += labels.size(0) # calculate總樣本數
            correct += (predicted == labels).sum().item()   # calculate正確預測數, sum(): 計算此陣列中True的總數, item(): tensor -> 轉成數字
    accuracy = 100 * correct / total
    # print(f"Accuracy of the model on the test images: {accuracy} % ")
    return accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model/SCCNet_model.pth"    # ./ : 是相對路徑, 表示從現在的目錄開始


# 加載模型並測試    

model = load_model(model_path)
model.to(device) 
""" 
test_dataset = MIBCI2aDataset(mode = 'FT_test')
test_loader = DataLoader(test_dataset, batch_size = 512, shuffle = False)

accuracy = test_model(model, test_loader, device)
print(f"Test accuracy: {accuracy:.2f}%")    """

# Test the model and print the accuracy
test_dataset = MIBCI2aDataset(mode='SD_test')
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
accuracy = test_model(model, test_loader, device)
print(f"Accuracy of the model on the test images: {accuracy} %")