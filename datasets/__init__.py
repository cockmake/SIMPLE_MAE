import os
from torch.utils.data import Dataset
import cv2 as cv
class LoadDataset(Dataset):
    def __init__(self, imgs_root, H=320, W=320, transform=None):
        super(LoadDataset, self).__init__()
        img_names = os.listdir(imgs_root)
        self.input_data = []
        # print('-' * 10, '加载数据中...', '-' * 10)
        for filename in img_names:
            img = cv.imread(os.path.join(imgs_root, filename))
            img = cv.resize(img, (W, H))
            if transform is not None:
                img = transform(img)
            self.input_data.append(img)
        # print('-' * 10, '加载完毕。', '-' * 10)
        self.input_data = self.input_data * 40


    def __getitem__(self, idx):
        return self.input_data[idx]

    def __len__(self):
        return len(self.input_data)