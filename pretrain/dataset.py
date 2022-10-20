import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
import os


class GoogleLandmarkDataset(Dataset):
    def __init__(self,
                 image_list,
                 class_ids,
                 resize_shape,
                 data_path,
                 transform=None):
        super().__init__()

        self.data_path = data_path
        self.image_list = image_list
        self.class_ids = class_ids
        self.transform = transform
        self.base_transforms = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),  # ImageNet Normalization
        ])

    def __getitem__(self, index):
        img_path = str(self.image_list[index])
        img_path = os.path.join(self.data_path, img_path[0], img_path[1], img_path[2], img_path + '.jpg')
        img = Image.open(img_path).convert("RGB")
        assert img is not None, f'path: {img_path} is invalid'
        if self.transform is not None:
            img = self.transform(img)
        img = self.base_transforms(img)  # Always apply the base_transforms
        label = torch.tensor(self.class_ids[index]).long()
        return img, label

    def __len__(self):
        return len(self.image_list)

