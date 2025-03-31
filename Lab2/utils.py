# script for drawing figures, and more if needed


import matplotlib.pyplot as plt
import numpy as np

def PlotLoss(train_losses, val_losses, save_path, final_accuracy):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.text(0.5, 0.9, f'Final Accuracy: {final_accuracy:.2f}%', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.savefig(save_path)  # store result picture
    plt.close()  # 關閉當前圖形

