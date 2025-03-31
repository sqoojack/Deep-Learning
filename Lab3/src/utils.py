import torch

# dice_score: 衡量兩組數據相似度的指標, 等於兩倍交集大小除以兩個集合大小的總和
def dice_score(preds, masks):  # setA: predicted segmentation mask setB: ground truth mask
    # print(f"preds shape: {preds.shape}")
    # print(f"masks shape: {masks.shape}")
    # 轉換預測的多類別輸出為單類別
    preds = torch.argmax(preds, dim=1).float()  # 使preds跟masks形狀一樣
    
    smooth = 1e-10  # 用來避免除以0的情況
    preds = torch.sigmoid(preds)    # 對preds做預處理, 使其範圍在[0, 1]間 (masks已有做預處理, 值為0 or 1)
    # 使其值變成0 or 1  (if preds > 0.5 -> 轉成1 -> 加.float() -> 變成 1.0)
    preds = (preds > 0.5).float()   # 0是後(背)景, 1是前景
    
    intersection = (preds * masks).sum() # 只有preds 和masks等於1時 結果才為1 -> 再將其相加 -> 兩個交集的像素數量
    total_pixels = preds.sum() + masks.sum()
    dice = 2.0 * intersection / (total_pixels + smooth)
    return dice

