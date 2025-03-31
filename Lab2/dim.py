
import numpy as np

# 替換下面的路徑為您的文件實際存儲路徑
file_path = './dataset/SD_train/features/s2T.npy'

# 加載.npy文件
data = np.load(file_path)

# 打印數據的形狀
print("Data dimensions:", data.shape)

# ./dataset/FT/features/s9T.npy is (288, 22, 438)
# ./dataset/FT/labels/s9T.npy is (288,)
# ./dataset/LOSO_test/features/s9E.npy is (288, 22, 438)
# ./dataset/LOSO_test/labels/s9E.npy is (288,)
# ./dataset/SD_train/features/s2T.npy is (288, 22, 438)

