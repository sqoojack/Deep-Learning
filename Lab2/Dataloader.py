import torch
import numpy as np
import os
class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        features = []
        for filename in os.listdir(filePath):
            if filename.endswith('.npy'):
                feature = np.load(os.path.join(filePath, filename))
                features.append(feature)
                # print(feature.shape) debugging line
        if features:
            features = np.concatenate(features, axis=0)
            features = np.expand_dims(features, axis=1)  # Add channel dimension here
            features = torch.from_numpy(features).float()  # Convert to PyTorch tensor
            # print(features.shape)          
            return features
        else:
            raise RuntimeError(f"No feature files found in {filePath}")

    def _getLabels(self, filePath):
        # Assuming labels are stored the same way as features
        labels = self._getFeatures(filePath)
        return labels.long().squeeze()    # 將labels固定為long type

    
    def __init__(self, mode):
        assert mode in ['LOSO_train', 'LOSO_test', 'SD_train', 'SD_test', 'FT_train', 'FT_test']
        mode_path = {
            'LOSO_train': ('./dataset/LOSO_train/features/', './dataset/LOSO_train/labels/'),
            'LOSO_test': ('./dataset/LOSO_test/features/', './dataset/LOSO_test/labels/'),
            'SD_train': ('./dataset/SD_train/features/', './dataset/SD_train/labels/'),
            'SD_test': ('./dataset/SD_test/features/', './dataset/SD_test/labels/'),
            'FT_train': ('./dataset/FT/features/', './dataset/FT/labels/'),
            'FT_test': ('./dataset/LOSO_test/features/', './dataset/LOSO_test/labels/')
        }
        feature_path, label_path = mode_path[mode]
        self.features = self._getFeatures(feature_path)
        self.labels = self._getLabels(label_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


train_dataset = MIBCI2aDataset(mode='SD_train')

# 獲取數據集的長度
# print(f"Dataset length: {len(train_dataset)}")

# 取出第一個樣本並打印其形狀
# feature, label = train_dataset[0]
# print(f"Feature shape: {feature.shape}")
# print(f"Label shape: {label.shape}")
