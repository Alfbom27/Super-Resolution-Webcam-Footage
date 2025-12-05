from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
import os

class WebcamSRDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, scale_factor=4, transforms=None):
        self.patch_size = patch_size
        self.random_crop = T.RandomCrop(patch_size)
        self.scale_factor = scale_factor
        self.transforms = transforms

        self.image_paths = []
        for filename in os.listdir(root_dir):
            self.image_paths.append(os.path.join(root_dir, filename))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.image_paths[idx]).convert('RGB')
        hr_patch = self.random_crop(hr_img)

        if self.transforms:
            lr_patch = self.transforms(hr_patch.copy())
        else:
            lr_patch = hr_patch.resize(
                (self.patch_size // self.scale_factor, self.patch_size // self.scale_factor),
                Image.BICUBIC
            )

        hr_tensor = F.to_tensor(hr_patch)
        lr_tensor = F.to_tensor(lr_patch)

        return lr_tensor, hr_tensor
